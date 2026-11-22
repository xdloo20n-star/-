package store

import (
	"context"
	"database/sql"
	"strings"

	"sverka/internal/model"
)

type Pagination struct {
	Page int
	Size int
}

type Page[T any] struct {
	Items    []T `json:"items"`
	Page     int `json:"page"`
	PageSize int `json:"page_size"`
	Total    int `json:"total"`
}

func (s *Store) ListDocuments(ctx context.Context, page Pagination, corpusID string) (Page[model.Document], error) {
	var where string
	var args []any
	if corpusID != "" {
		where = " WHERE corpus_id = ?"
		args = []any{corpusID}
	}
	return list(ctx, s.db, page, "documents", documentColumns, where,
		"uploaded_at DESC, document_id DESC", args, func(row scanner) (model.Document, error) {
			return scanDocument(row)
		})
}

type ResultFilter struct {
	DocumentID string
	DatasetID  string
	TaskID     string
	ConfigID   string
	Metric     string
	Query      string
	HasIssues  *bool
}

func (s *Store) ListResults(ctx context.Context, page Pagination, filter ResultFilter) (Page[model.ResultSummary], error) {
	clauses := []string{}
	args := []any{}
	for _, field := range []struct{ column, value string }{
		{"document_id", filter.DocumentID},
		{"dataset_id", filter.DatasetID},
		{"task_id", filter.TaskID},
		{"config_id", filter.ConfigID},
		{"metric_key", strings.ToLower(filter.Metric)},
	} {
		if field.value != "" {
			clauses = append(clauses, field.column+" = ?")
			args = append(args, field.value)
		}
	}
	if filter.HasIssues != nil {
		clauses = append(clauses, "has_issues = ?")
		args = append(args, *filter.HasIssues)
	}
	if filter.Query != "" {
		clauses = append(clauses, "instr(search_text, ?) > 0")
		args = append(args, strings.ToLower(filter.Query))
	}
	where := ""
	if len(clauses) > 0 {
		where = " WHERE " + strings.Join(clauses, " AND ")
	}
	return list(ctx, s.db, page, "results", "summary_json", where, "result_id ASC", args, scanResultSummary)
}

func scanResultSummary(row scanner) (model.ResultSummary, error) {
	var result model.ResultSummary
	var data []byte
	if err := row.Scan(&data); err != nil {
		return result, err
	}
	err := decodeJSON(data, &result)
	return result, err
}

func list[T any](ctx context.Context, db *sql.DB, page Pagination, table, columns, where, order string, args []any, decode func(scanner) (T, error)) (Page[T], error) {
	result := Page[T]{Items: []T{}, Page: page.Page, PageSize: page.Size}
	tx, err := db.BeginTx(ctx, &sql.TxOptions{ReadOnly: true})
	if err != nil {
		return result, err
	}
	defer tx.Rollback()
	from := " FROM " + table + where
	if err := tx.QueryRowContext(ctx, "SELECT count(*)"+from, args...).Scan(&result.Total); err != nil {
		return result, err
	}
	offset := (page.Page - 1) * page.Size
	if offset >= result.Total {
		return result, tx.Commit()
	}
	query := "SELECT " + columns + from + " ORDER BY " + order + " LIMIT ? OFFSET ?"
	rows, err := tx.QueryContext(ctx, query, append(args, page.Size, offset)...)
	if err != nil {
		return result, err
	}
	defer rows.Close()
	for rows.Next() {
		item, err := decode(rows)
		if err != nil {
			return result, err
		}
		result.Items = append(result.Items, item)
	}
	if err := rows.Err(); err != nil {
		return result, err
	}
	return result, tx.Commit()
}
