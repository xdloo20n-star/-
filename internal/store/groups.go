package store

import (
	"context"

	"sverka/internal/model"
)

// GroupCandidates reads the comparison fields from completed snapshots.
func (s *Store) GroupCandidates(ctx context.Context) (model.GroupCandidates, error) {
	var candidates model.GroupCandidates
	rows, err := s.exportDB.QueryContext(ctx, `
		SELECT extraction_mode, json_remove(data_json, '$.source_document', '$.configurations', '$.evidence', '$.issues')
		FROM documents WHERE status = 'completed'`)
	if err != nil {
		return candidates, err
	}
	defer rows.Close()
	for rows.Next() {
		var mode string
		var raw []byte
		if err := rows.Scan(&mode, &raw); err != nil {
			return model.GroupCandidates{}, err
		}
		var data model.DocumentData
		if err := decodeJSON(raw, &data); err != nil {
			return model.GroupCandidates{}, err
		}
		candidates.AddDocument(mode, data)
	}
	return candidates, rows.Err()
}
