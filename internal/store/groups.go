package store

import (
	"context"

	"sverka/internal/model"
)

// GroupCandidates reads the comparison fields from completed snapshots.
func (s *Store) GroupCandidates(ctx context.Context) ([]model.ResultDetail, error) {
	rows, err := s.db.QueryContext(ctx, `
		SELECT json_remove(data_json, '$.source_document', '$.configurations', '$.evidence', '$.issues')
		FROM documents WHERE status = 'completed'`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var details []model.ResultDetail
	for rows.Next() {
		var raw []byte
		if err := rows.Scan(&raw); err != nil {
			return nil, err
		}
		var data model.DocumentData
		if err := decodeJSON(raw, &data); err != nil {
			return nil, err
		}
		details = data.AppendGroupCandidates(details)
	}
	return details, rows.Err()
}
