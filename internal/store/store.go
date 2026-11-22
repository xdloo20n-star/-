package store

import (
	"bytes"
	"context"
	"database/sql"
	_ "embed"
	"encoding/json"
	"errors"
	"fmt"
	"net/url"
	"path/filepath"
	"strings"

	_ "modernc.org/sqlite"

	"sverka/internal/model"
)

//go:embed 001_init.sql
var initialSchema string

type Store struct {
	db *sql.DB
}

func Open(ctx context.Context, path string) (*Store, error) {
	absolute, err := filepath.Abs(path)
	if err != nil {
		return nil, err
	}
	dsn := url.URL{Scheme: "file", Path: absolute}
	pragmas := url.Values{"_pragma": {"busy_timeout(5000)", "foreign_keys(ON)", "journal_mode(WAL)"}}
	dsn.RawQuery = pragmas.Encode()
	db, err := sql.Open("sqlite", dsn.String())
	if err != nil {
		return nil, err
	}
	db.SetMaxOpenConns(1)
	if err := migrate(ctx, db); err != nil {
		_ = db.Close()
		return nil, fmt.Errorf("migrate SQLite: %w", err)
	}
	return &Store{db: db}, nil
}

func (s *Store) Close() error { return s.db.Close() }

func (s *Store) Ping(ctx context.Context) error { return s.db.PingContext(ctx) }

func migrate(ctx context.Context, db *sql.DB) error {
	tx, err := db.BeginTx(ctx, nil)
	if err != nil {
		return err
	}
	defer tx.Rollback()
	var version int
	if err := tx.QueryRowContext(ctx, "PRAGMA user_version").Scan(&version); err != nil {
		return err
	}
	switch version {
	case 0:
		if _, err := tx.ExecContext(ctx, initialSchema); err != nil {
			return err
		}
	case 1:
		if err := refreshStoredBaselines(ctx, tx); err != nil {
			return err
		}
	case 2:
		return tx.Commit()
	default:
		return fmt.Errorf("unsupported database version %d", version)
	}
	if _, err := tx.ExecContext(ctx, "PRAGMA user_version = 2"); err != nil {
		return err
	}
	return tx.Commit()
}

const documentColumns = "document_id, original_filename, corpus_id, status, extraction_mode, failure_code, failure_message, uploaded_at"

func (s *Store) Enqueue(ctx context.Context, doc model.Document) error {
	_, err := s.db.ExecContext(ctx, `
		INSERT INTO documents (document_id, original_filename, corpus_id, status, extraction_mode, uploaded_at)
		VALUES (?, ?, ?, 'queued', ?, ?)`, doc.ID, doc.OriginalFilename, doc.CorpusID, doc.ExtractionMode, doc.UploadedAt)
	return err
}

func (s *Store) Recover(ctx context.Context) error {
	_, err := s.db.ExecContext(ctx, `
		UPDATE documents SET status = 'failed', failure_code = 'interrupted',
			failure_message = 'Обработка прервалась при перезапуске сервера.'
		WHERE status = 'processing'`)
	return err
}

func (s *Store) Claim(ctx context.Context) (model.Document, error) {
	query := `
		UPDATE documents SET status = 'processing'
		WHERE document_id = (
			SELECT document_id FROM documents WHERE status = 'queued'
			ORDER BY uploaded_at, document_id LIMIT 1
		) AND status = 'queued'
		RETURNING ` + documentColumns
	return scanDocument(s.db.QueryRowContext(ctx, query))
}

func (s *Store) Fail(ctx context.Context, docID string, failure model.Problem) error {
	result, err := s.db.ExecContext(ctx, `
		UPDATE documents SET status = 'failed', failure_code = ?, failure_message = ?
		WHERE document_id = ? AND status = 'processing'`, failure.Code, failure.Message, docID)
	return requireProcessing(result, err)
}

func (s *Store) Complete(ctx context.Context, docID string, data *model.DocumentData) error {
	snapshot, err := json.Marshal(data)
	if err != nil {
		return err
	}
	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return err
	}
	defer tx.Rollback()
	result, err := tx.ExecContext(ctx, `
		UPDATE documents SET status = 'completed', data_json = ?
		WHERE document_id = ? AND status = 'processing'`, string(snapshot), docID)
	if err := requireProcessing(result, err); err != nil {
		return err
	}

	stmt, err := tx.PrepareContext(ctx, `
		INSERT INTO results (result_id, document_id, dataset_id, task_id, config_id,
			metric_key, has_issues, search_text, summary_json)
		VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)`)
	if err != nil {
		return err
	}
	defer stmt.Close()
	for _, row := range data.Results {
		raw, err := json.Marshal(row)
		if err != nil {
			return err
		}
		record := row.Result
		search := strings.Join([]string{
			data.Publication.Text("title"), record.Text("metric_name"), value(row.DatasetName),
			value(row.TaskType), value(row.ConfigurationName),
		}, "\n")
		_, err = stmt.ExecContext(ctx, record.Text("result_id"), docID,
			record.TextPtr("dataset_id"), record.TextPtr("task_id"), record.TextPtr("config_id"),
			strings.ToLower(record.Text("metric_name")), row.IssueCount > 0, strings.ToLower(search), string(raw))
		if err != nil {
			return err
		}
	}
	return tx.Commit()
}

func requireProcessing(result sql.Result, err error) error {
	if err != nil {
		return err
	}
	n, err := result.RowsAffected()
	if err != nil {
		return err
	}
	if n != 1 {
		return errors.New("document no longer processing")
	}
	return nil
}

func value(text *string) string {
	if text == nil {
		return ""
	}
	return *text
}

type scanner interface {
	Scan(dest ...any) error
}

func scanDocument(row scanner, extra ...any) (model.Document, error) {
	var doc model.Document
	var failureCode, failureMessage sql.NullString
	dest := []any{
		&doc.ID, &doc.OriginalFilename, &doc.CorpusID, &doc.Status,
		&doc.ExtractionMode, &failureCode, &failureMessage, &doc.UploadedAt,
	}
	if err := row.Scan(append(dest, extra...)...); err != nil {
		return model.Document{}, err
	}
	if failureCode.Valid {
		doc.Failure = &model.Problem{Code: failureCode.String, Message: failureMessage.String}
	}
	doc.FileURL = "/api/v1/documents/" + doc.ID + "/file"
	return doc, nil
}

func (s *Store) Document(ctx context.Context, id string) (model.Document, error) {
	return scanDocument(s.db.QueryRowContext(ctx, "SELECT "+documentColumns+" FROM documents WHERE document_id = ?", id))
}

func scanDocumentDetail(row scanner) (model.DocumentDetail, error) {
	var snapshot sql.NullString
	doc, err := scanDocument(row, &snapshot)
	if err != nil {
		return model.DocumentDetail{}, err
	}
	detail := model.DocumentDetail{Document: doc}
	if snapshot.Valid {
		if err := decodeJSON([]byte(snapshot.String), &detail.Data); err != nil {
			return model.DocumentDetail{}, err
		}
	}
	return detail, nil
}

func (s *Store) DocumentDetail(ctx context.Context, id string) (model.DocumentDetail, error) {
	row := s.db.QueryRowContext(ctx, "SELECT "+documentColumns+", data_json FROM documents WHERE document_id = ?", id)
	return scanDocumentDetail(row)
}

func (s *Store) ResultDetail(ctx context.Context, id string) (model.ResultDetail, error) {
	var snapshot, summary []byte
	err := s.db.QueryRowContext(ctx, `
		SELECT d.data_json, r.summary_json FROM results r
		JOIN documents d ON d.document_id = r.document_id WHERE r.result_id = ?`, id).Scan(&snapshot, &summary)
	if err != nil {
		return model.ResultDetail{}, err
	}
	var data model.DocumentData
	if err := decodeJSON(snapshot, &data); err != nil {
		return model.ResultDetail{}, err
	}
	var row model.ResultSummary
	if err := decodeJSON(summary, &row); err != nil {
		return model.ResultDetail{}, err
	}
	return data.ResultDetail(row), nil
}

func decodeJSON(raw []byte, dst any) error {
	reader := json.NewDecoder(bytes.NewReader(raw))
	reader.UseNumber()
	return reader.Decode(dst)
}

func (s *Store) Export(ctx context.Context, id string) ([]model.DocumentDetail, error) {
	if id != "" {
		document, err := s.DocumentDetail(ctx, id)
		if err != nil {
			return nil, err
		}
		return []model.DocumentDetail{document}, nil
	}
	query := "SELECT " + documentColumns + ", data_json FROM documents ORDER BY uploaded_at DESC, document_id DESC"
	rows, err := s.db.QueryContext(ctx, query)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	documents := []model.DocumentDetail{}
	for rows.Next() {
		doc, err := scanDocumentDetail(rows)
		if err != nil {
			return nil, err
		}
		documents = append(documents, doc)
	}
	return documents, rows.Err()
}
