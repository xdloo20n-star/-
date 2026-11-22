package store

import (
	"context"
	"path/filepath"
	"slices"
	"testing"
	"time"

	"sverka/internal/model"
)

func TestReadExportKeepsWriterAvailable(t *testing.T) {
	db, err := Open(t.Context(), filepath.Join(t.TempDir(), "export.sqlite"))
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	for _, id := range []string{"doc_first", "doc_second"} {
		if err := db.Enqueue(t.Context(), model.Document{ID: id, ExtractionMode: "mock", UploadedAt: "2026-11-20T12:00:00Z"}); err != nil {
			t.Fatal(err)
		}
	}
	ctx, cancel := context.WithCancel(t.Context())
	defer cancel()
	started := make(chan struct{})
	release := make(chan struct{})
	done := make(chan error, 1)
	var exported []string
	go func() {
		done <- db.ReadExport(ctx, "", func(doc model.Document, _ []byte) error {
			exported = append(exported, doc.ID)
			if len(exported) == 1 {
				close(started)
				select {
				case <-release:
				case <-ctx.Done():
					return ctx.Err()
				}
			}
			return nil
		})
	}()
	select {
	case <-started:
	case err := <-done:
		t.Fatalf("export returned before visiting documents: %v", err)
	case <-time.After(5 * time.Second):
		t.Fatal("export did not visit the first document")
	}
	writeCtx, cancelWrite := context.WithTimeout(t.Context(), 2*time.Second)
	defer cancelWrite()
	err = db.Enqueue(writeCtx, model.Document{ID: "doc_third", ExtractionMode: "mock", UploadedAt: "2026-11-20T13:00:00Z"})
	close(release)
	if err != nil {
		t.Fatalf("enqueue while export was blocked: %v", err)
	}
	if err := <-done; err != nil {
		t.Fatal(err)
	}
	if want := []string{"doc_second", "doc_first"}; !slices.Equal(exported, want) {
		t.Fatalf("exported IDs = %v, want snapshot %v", exported, want)
	}
}
