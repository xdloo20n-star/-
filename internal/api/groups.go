package api

import (
	"net/http"

	"sverka/internal/model"
)

func (s *Server) comparableGroups(w http.ResponseWriter, r *http.Request) {
	documents, err := s.store.Export(r.Context(), "")
	if err != nil {
		s.storeError(w, r, err)
		return
	}
	details := make([]model.ResultDetail, 0)
	for _, document := range documents {
		if document.Data == nil || document.Document.Status != model.Completed {
			continue
		}
		for _, result := range document.Data.Results {
			details = append(details, document.Data.ResultDetail(result))
		}
	}
	writeJSON(w, http.StatusOK, struct {
		Groups []model.ComparableGroup `json:"groups"`
	}{Groups: model.ComparableGroups(details)})
}
