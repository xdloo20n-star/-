package store

import (
	"path/filepath"
	"testing"

	"sverka/internal/model"
)

func TestAtomicImportAndRecovery(t *testing.T) {
	ctx := t.Context()
	path := filepath.Join(t.TempDir(), "recovery.sqlite")
	db, err := Open(ctx, path)
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	for _, id := range []string{"first", "second"} {
		if err := db.Enqueue(ctx, model.Document{
			ID: id, OriginalFilename: "D019.pdf", ExtractionMode: "mock", UploadedAt: "2026-11-20T13:00:00Z",
		}); err != nil {
			t.Fatal(err)
		}
	}
	job, err := db.Claim(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if job.ID != "first" || job.Status != model.Processing {
		t.Fatalf("claim expected: first/processing, got: %s/%s", job.ID, job.Status)
	}
	summary := model.ResultSummary{
		DocumentID: job.ID, ExtractionMode: "mock",
		Result: model.Record{"result_id": "repeated-id", "metric_name": "Recall@5"},
	}
	err = db.Complete(ctx, job.ID, &model.DocumentData{Results: []model.ResultSummary{summary, summary}})
	if err == nil {
		t.Fatal("import error expected: duplicate result ID, got: nil")
	}
	doc, err := db.DocumentDetail(ctx, job.ID)
	if err != nil {
		t.Fatal(err)
	}
	if doc.Document.Status != model.Processing || doc.Data != nil {
		t.Fatalf("rolled-back state expected: processing with no data, got: %+v", doc)
	}
	rows, err := db.ListResults(ctx, Pagination{Page: 1, Size: 25}, ResultFilter{})
	if err != nil {
		t.Fatal(err)
	}
	if rows.Total != 0 {
		t.Fatalf("rolled-back result count expected: 0, got: %d", rows.Total)
	}

	if err := db.Close(); err != nil {
		t.Fatal(err)
	}
	db, err = Open(ctx, path)
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	if err := db.Recover(ctx); err != nil {
		t.Fatal(err)
	}
	first, err := db.Document(ctx, "first")
	if err != nil {
		t.Fatal(err)
	}
	if first.Status != model.Failed || first.Failure == nil || first.Failure.Code != "interrupted" {
		t.Fatalf("recovered job expected: failed/interrupted, got: %+v", first)
	}
	second, err := db.Claim(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if second.ID != "second" || second.Status != model.Processing {
		t.Fatalf("pending job expected: second/processing, got: %s/%s", second.ID, second.Status)
	}
	if err := db.Fail(ctx, second.ID, model.Problem{Code: "test", Message: "test"}); err != nil {
		t.Fatal(err)
	}
	if err := db.Complete(ctx, second.ID, &model.DocumentData{}); err == nil {
		t.Fatal("transition error expected: failure is final, got: nil")
	}
}
