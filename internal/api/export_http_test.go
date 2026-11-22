package api

import (
	"bytes"
	"encoding/json"
	"net/http"
	"testing"
	"time"

	"sverka"
	"sverka/internal/extract"
	"sverka/internal/model"
)

func TestStreamedExportShapeAndRawSnapshot(t *testing.T) {
	h := newHarness(t, extract.MockExtractor{Fixture: sverka.MockExtraction}, time.Second)
	endpoint := h.url + "/api/v1/export"
	type export struct {
		SchemaVersion string `json:"schema_version"`
		Documents     []struct {
			Document model.Document  `json:"document"`
			Data     json.RawMessage `json:"data"`
		} `json:"documents"`
	}
	empty := decode[export](t, request(t, http.MethodGet, endpoint, "", nil, http.StatusOK))
	if empty.Documents == nil || len(empty.Documents) != 0 {
		t.Fatalf("empty export documents = %+v", empty.Documents)
	}
	const pdf = "%PDF-1.7\n"
	first := decode[model.Document](t, postPDF(t, h, "", "D001.pdf", pdf, http.StatusAccepted))
	if _, err := h.app.store.Claim(t.Context()); err != nil {
		t.Fatal(err)
	}
	snapshot := &model.DocumentData{Publication: model.Record{"precise": json.Number("6.789e+123")}}
	if err := h.app.store.Complete(t.Context(), first.ID, snapshot); err != nil {
		t.Fatal(err)
	}
	second := decode[model.Document](t, postPDF(t, h, "", "P001.pdf", pdf, http.StatusAccepted))
	want, err := json.Marshal(snapshot)
	if err != nil {
		t.Fatal(err)
	}
	result := decode[export](t, request(t, http.MethodGet, endpoint, "", nil, http.StatusOK))
	equal(t, "schema version", model.SchemaVersion, result.SchemaVersion)
	equal(t, "exported document count", 2, len(result.Documents))
	equal(t, "newest document ID", second.ID, result.Documents[0].Document.ID)
	equal(t, "queued document data", "null", string(result.Documents[0].Data))
	equal(t, "oldest document ID", first.ID, result.Documents[1].Document.ID)
	if got := result.Documents[1].Data; !bytes.Equal(got, want) {
		t.Fatalf("raw snapshot = %s, want %s", got, want)
	}
	body := request(t, http.MethodGet, endpoint+"?document_id=doc_missing", "", nil, http.StatusNotFound)
	equal(t, "missing export document code", "not_found", decode[errorResponse](t, body).Error.Code)
}
