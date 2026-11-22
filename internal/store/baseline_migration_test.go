package store

import (
	"path/filepath"
	"strings"
	"testing"

	"sverka"
	"sverka/internal/extract"
	"sverka/internal/model"
)

func TestBackfillBaselinesUpdatesStoredSnapshotsAndResultIndex(t *testing.T) {
	ctx := t.Context()
	path := filepath.Join(t.TempDir(), "legacy.sqlite")
	db, err := Open(ctx, path)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = db.Close() })
	if _, err := db.db.ExecContext(ctx, "PRAGMA user_version = 1"); err != nil {
		t.Fatal(err)
	}
	doc := model.Document{ID: "legacy", OriginalFilename: "D001.pdf", ExtractionMode: "mock"}
	if err := db.Enqueue(ctx, doc); err != nil {
		t.Fatal(err)
	}
	if _, err := db.Claim(ctx); err != nil {
		t.Fatal(err)
	}
	importer, err := extract.NewImporter(sverka.ExtractorSchema)
	if err != nil {
		t.Fatal(err)
	}
	data, err := importer.Prepare(sverka.MockExtraction, doc)
	if err != nil {
		t.Fatal(err)
	}
	// A v1 database stores the rejected baseline without an issue or indexed warnings.
	data.Results[0].Result["aggregation"] = "median"
	data.Results[1].Baseline = nil
	if err := db.Complete(ctx, doc.ID, data); err != nil {
		t.Fatal(err)
	}
	if _, err := db.db.ExecContext(ctx, `CREATE TRIGGER fail_index_update BEFORE UPDATE ON results
		BEGIN SELECT RAISE(ABORT, 'test index update failure'); END`); err != nil {
		t.Fatal(err)
	}
	if err := migrate(ctx, db.db); err == nil || !strings.Contains(err.Error(), "test index update failure") {
		t.Fatalf("migration expected: index update failure, got: %v", err)
	}
	version := 0
	if err := db.db.QueryRowContext(ctx, "PRAGMA user_version").Scan(&version); err != nil {
		t.Fatal(err)
	}
	rolledBack, err := db.DocumentDetail(ctx, doc.ID)
	if err != nil {
		t.Fatal(err)
	}
	if version != 1 || len(rolledBack.Data.Issues) != 0 {
		t.Fatalf("rollback expected: version 1 and no warnings, got version %d and %+v", version, rolledBack.Data.Issues)
	}
	if _, err := db.db.ExecContext(ctx, "DROP TRIGGER fail_index_update"); err != nil {
		t.Fatal(err)
	}
	if err := db.Close(); err != nil {
		t.Fatal(err)
	}
	upgraded, err := Open(ctx, path)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = upgraded.Close() })
	exported, err := upgraded.Export(ctx, doc.ID)
	if err != nil {
		t.Fatal(err)
	}
	issues := exported[0].Data.Issues
	if len(issues) != 1 || issues[0].CheckName != "baseline_unavailable" || len(issues[0].EvidenceIDs) < 2 {
		t.Fatalf("export expected: one baseline warning with both sources, got: %+v", issues)
	}
	hasIssues := true
	page, err := upgraded.ListResults(ctx, Pagination{Page: 1, Size: 25}, ResultFilter{HasIssues: &hasIssues})
	if err != nil {
		t.Fatal(err)
	}
	if page.Total != 2 || len(page.Items) != 2 {
		t.Fatalf("filtered results expected: two results with warnings, got: %+v", page)
	}
	for _, row := range page.Items {
		detail, err := upgraded.ResultDetail(ctx, row.Result.Text("result_id"))
		if err != nil {
			t.Fatal(err)
		}
		if row.IssueCount != 1 || len(detail.Issues) != 1 || detail.Issues[0].ID != issues[0].ID {
			t.Fatalf("result details expected: the same indexed warning as in export, got: %+v", detail)
		}
	}
}
