package api

import (
	"context"
	"database/sql"
	"errors"
	"net/http"
	"path/filepath"
	"testing"
	"time"

	"sverka"
	"sverka/internal/model"
	"sverka/internal/store"
)

func TestWorkerRetriesLockedTerminalWrites(t *testing.T) {
	for _, test := range []struct {
		name, want string
		extractErr error
	}{
		{"completion", model.Completed, nil},
		{"failure", model.Failed, errors.New("stub extractor failure")},
	} {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()
			entered := make(chan struct{})
			release := make(chan struct{})
			h := newHarness(t, testExtractor(func(ctx context.Context, _ string) ([]byte, error) {
				close(entered)
				select {
				case <-release:
					return sverka.MockExtraction, test.extractErr
				case <-ctx.Done():
					return nil, ctx.Err()
				}
			}), 20*time.Second)
			doc := decode[model.Document](t, upload(t, h, "%PDF-1.7\n", http.StatusAccepted))
			h.startWorker(t)
			waitForWorkerEvent(t, entered)

			locker, err := sql.Open("sqlite", filepath.Join(h.app.pdfDir, "test.sqlite"))
			if err != nil {
				t.Fatal(err)
			}
			defer locker.Close()
			conn, err := locker.Conn(t.Context())
			if err != nil {
				t.Fatal(err)
			}
			defer conn.Close()
			if _, err := conn.ExecContext(t.Context(), "BEGIN IMMEDIATE"); err != nil {
				t.Fatal(err)
			}
			close(release)
			// Hold the writer lock beyond the store's five-second busy timeout.
			time.Sleep(6 * time.Second)
			if _, err := conn.ExecContext(t.Context(), "ROLLBACK"); err != nil {
				t.Fatal(err)
			}

			terminal := waitFor(t, h, doc.ID, test.want)
			results := get[store.Page[model.ResultSummary]](t, h, "/api/v1/results")
			if test.extractErr != nil {
				if terminal.Document.Failure == nil || terminal.Document.Failure.Code != "extraction_failed" || results.Total != 0 {
					t.Fatalf("failure = %+v, results = %d", terminal.Document.Failure, results.Total)
				}
			} else if terminal.Data == nil || results.Total != 2 {
				t.Fatalf("completion has data = %t, results = %d", terminal.Data != nil, results.Total)
			}
		})
	}
}

func TestWorkerStopsClaimsAndGrantsActiveJobGrace(t *testing.T) {
	for _, test := range []struct {
		name, want string
		grace      time.Duration
	}{
		{"active job completes", model.Completed, 2 * time.Second},
		{"active job exceeds grace", model.Failed, 150 * time.Millisecond},
	} {
		t.Run(test.name, func(t *testing.T) {
			entered := make(chan struct{}, 2)
			release := make(chan struct{})
			h := newHarness(t, testExtractor(func(ctx context.Context, _ string) ([]byte, error) {
				entered <- struct{}{}
				select {
				case <-release:
					return sverka.MockExtraction, nil
				case <-ctx.Done():
					return nil, ctx.Err()
				}
			}), time.Minute)
			first := decode[model.Document](t, upload(t, h, "%PDF-1.7\n", http.StatusAccepted))
			ctx, stop := context.WithCancel(t.Context())
			done := make(chan struct{})
			go func() {
				defer close(done)
				h.app.RunWorkerWithGrace(ctx, test.grace)
			}()
			t.Cleanup(func() {
				stop()
				waitForWorkerEvent(t, done)
			})
			waitForWorkerEvent(t, entered)
			second := decode[model.Document](t, upload(t, h, "%PDF-1.7\n", http.StatusAccepted))
			stop()
			if test.want == model.Completed {
				close(release)
			}
			waitForWorkerEvent(t, done)

			active := waitFor(t, h, first.ID, test.want)
			if test.want == model.Failed && (active.Document.Failure == nil || active.Document.Failure.Code != "interrupted") {
				t.Fatalf("failure = %+v, want interrupted", active.Document.Failure)
			}
			waiting := get[model.DocumentDetail](t, h, "/api/v1/documents/"+second.ID)
			if waiting.Document.Status != model.Queued || len(entered) != 0 {
				t.Fatalf("second job status = %q, additional extraction starts = %d", waiting.Document.Status, len(entered))
			}
		})
	}
}

func waitForWorkerEvent(t *testing.T, event <-chan struct{}) {
	t.Helper()
	select {
	case <-event:
	case <-time.After(5 * time.Second):
		t.Fatal("timed out waiting for worker")
	}
}
