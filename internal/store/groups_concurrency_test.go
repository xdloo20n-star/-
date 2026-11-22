package store

import (
	"context"
	"path/filepath"
	"testing"
	"time"

	"sverka/internal/model"
)

func TestGroupReadDoesNotBlockQueueWrites(t *testing.T) {
	db, err := Open(t.Context(), filepath.Join(t.TempDir(), "groups.sqlite"))
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	conn, err := db.exportDB.Conn(t.Context())
	if err != nil {
		t.Fatal(err)
	}
	done := make(chan error, 1)
	go func() {
		_, err := db.GroupCandidates(t.Context())
		done <- err
	}()
	waitUntil := time.Now().Add(2 * time.Second)
	for db.exportDB.Stats().WaitCount == 0 && time.Now().Before(waitUntil) {
		time.Sleep(time.Millisecond)
	}
	if db.exportDB.Stats().WaitCount == 0 {
		_ = conn.Close()
		t.Fatal("group read did not wait for the read connection")
	}
	ctx, cancel := context.WithTimeout(t.Context(), 2*time.Second)
	defer cancel()
	err = db.Enqueue(ctx, model.Document{ID: "waiting", ExtractionMode: "mock"})
	if closeErr := conn.Close(); closeErr != nil {
		t.Fatal(closeErr)
	}
	if err != nil {
		t.Fatalf("enqueue while group read waits: %v", err)
	}
	if err := <-done; err != nil {
		t.Fatal(err)
	}
}
