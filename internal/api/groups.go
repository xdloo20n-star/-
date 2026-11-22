package api

import (
	"net/http"

	"sverka/internal/model"
)

func (s *Server) comparableGroups(w http.ResponseWriter, r *http.Request) {
	if _, ok := requestQuery(w, r); !ok {
		return
	}
	candidates, err := s.store.GroupCandidates(r.Context())
	if err != nil {
		s.storeError(w, r, err)
		return
	}
	writeJSON(w, http.StatusOK, struct {
		Groups []model.ComparableGroup `json:"groups"`
	}{Groups: candidates.Groups()})
}
