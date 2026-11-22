package store

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"

	"sverka/internal/model"
)

// Refresh the document snapshots and result index in the same migration transaction.
func refreshStoredBaselines(ctx context.Context, tx *sql.Tx) error {
	ids, err := documentsWithBaselines(ctx, tx)
	if err != nil {
		return err
	}
	for _, id := range ids {
		if err := refreshDocumentBaselines(ctx, tx, id); err != nil {
			return fmt.Errorf("refresh baselines for %s: %w", id, err)
		}
	}
	return nil
}

func documentsWithBaselines(ctx context.Context, tx *sql.Tx) ([]string, error) {
	rows, err := tx.QueryContext(ctx, `
		SELECT document_id FROM documents WHERE status = 'completed' AND EXISTS (
			SELECT 1 FROM json_each(documents.data_json, '$.results') AS result
			WHERE json_type(result.value, '$.result.baseline_result_id') = 'text'
		)`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var ids []string
	for rows.Next() {
		var id string
		if err := rows.Scan(&id); err != nil {
			return nil, err
		}
		ids = append(ids, id)
	}
	return ids, rows.Err()
}

func refreshDocumentBaselines(ctx context.Context, tx *sql.Tx, id string) error {
	var snapshot []byte
	if err := tx.QueryRowContext(ctx, "SELECT data_json FROM documents WHERE document_id = ?", id).Scan(&snapshot); err != nil {
		return err
	}
	var data model.DocumentData
	if err := decodeJSON(snapshot, &data); err != nil {
		return err
	}
	data.RefreshBaselines()
	updated, err := json.Marshal(&data)
	if err != nil {
		return err
	}
	if _, err := tx.ExecContext(ctx, "UPDATE documents SET data_json = ? WHERE document_id = ?", string(updated), id); err != nil {
		return err
	}
	for _, row := range data.Results {
		summary, err := json.Marshal(row)
		if err != nil {
			return err
		}
		if _, err := tx.ExecContext(ctx, `
			UPDATE results SET summary_json = ?, has_issues = ? WHERE document_id = ? AND result_id = ?`,
			string(summary), row.IssueCount > 0, id, row.Result.Text("result_id")); err != nil {
			return err
		}
	}
	return nil
}
