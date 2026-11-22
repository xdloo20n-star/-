package store

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"slices"

	"sverka/internal/model"
)

// Refresh the document snapshots and result index in the same migration transaction.
func refreshStoredResultChecks(ctx context.Context, tx *sql.Tx, legacyBaselines bool) error {
	ids, err := completedDocumentIDs(ctx, tx)
	if err != nil {
		return err
	}
	for _, id := range ids {
		if err := refreshDocumentResultChecks(ctx, tx, id, legacyBaselines); err != nil {
			return fmt.Errorf("refresh result checks for %s: %w", id, err)
		}
	}
	return nil
}

func completedDocumentIDs(ctx context.Context, tx *sql.Tx) ([]string, error) {
	rows, err := tx.QueryContext(ctx, "SELECT document_id FROM documents WHERE status = 'completed'")
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

func refreshDocumentResultChecks(ctx context.Context, tx *sql.Tx, id string, legacyBaselines bool) error {
	var snapshot []byte
	if err := tx.QueryRowContext(ctx, "SELECT data_json FROM documents WHERE document_id = ?", id).Scan(&snapshot); err != nil {
		return err
	}
	var data model.DocumentData
	if err := decodeJSON(snapshot, &data); err != nil {
		return err
	}
	changed := data.RefreshResultIntegrity()
	if !changed && !(legacyBaselines && slices.ContainsFunc(data.Results, func(row model.ResultSummary) bool {
		return row.Result.Text("baseline_result_id") != ""
	})) {
		return nil
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
