package main

import (
	"io"
	"log/slog"
	"net"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"sverka/internal/model"
	"sverka/internal/store"
)

func TestInvalidExtractorStartupPreservesQueue(t *testing.T) {
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	defer listener.Close()
	host, port, err := net.SplitHostPort(listener.Addr().String())
	if err != nil {
		t.Fatal(err)
	}

	for _, test := range []struct {
		name, mode, want string
		script, binFile  bool
	}{
		{"queued python needs script", "mock", "PYTHON_SCRIPT is required for queued python jobs", false, false},
		{"queued python needs executable", "mock", "PYTHON_BIN must resolve", true, false},
		{"selected python needs executable", "python", "PYTHON_BIN must resolve", true, false},
		{"python binary lacks execute permission", "python", "PYTHON_BIN must resolve", true, true},
	} {
		t.Run(test.name, func(t *testing.T) {
			dir := t.TempDir()
			t.Setenv("DATA_DIR", dir)
			t.Setenv("HOST", host)
			t.Setenv("PORT", port)
			t.Setenv("EXTRACTION_MODE", test.mode)
			t.Setenv("EXTRACT_TIMEOUT", "1m")
			t.Setenv("MAX_PDF_BYTES", "1024")
			bin := filepath.Join(dir, "python-bin")
			if test.binFile {
				if err := os.WriteFile(bin, []byte("#!/bin/sh\n"), 0600); err != nil {
					t.Fatal(err)
				}
			}
			t.Setenv("PYTHON_BIN", bin)
			t.Setenv("PYTHON_SCRIPT", "")
			if test.script {
				script := filepath.Join(dir, "stub.py")
				if err := os.WriteFile(script, []byte("pass\n"), 0600); err != nil {
					t.Fatal(err)
				}
				t.Setenv("PYTHON_SCRIPT", script)
			}

			db, err := store.Open(t.Context(), filepath.Join(dir, "sverka.sqlite"))
			if err != nil {
				t.Fatal(err)
			}
			defer db.Close()
			processing := model.Document{ID: "active", OriginalFilename: "active.pdf", ExtractionMode: "mock", UploadedAt: "2026-01-01T00:00:00Z"}
			queued := model.Document{ID: "waiting", OriginalFilename: "waiting.pdf", ExtractionMode: "python", UploadedAt: "2026-01-01T00:00:01Z"}
			for _, doc := range []model.Document{processing, queued} {
				if err := db.Enqueue(t.Context(), doc); err != nil {
					t.Fatal(err)
				}
			}
			if _, err := db.Claim(t.Context()); err != nil {
				t.Fatal(err)
			}

			logger := slog.New(slog.NewTextHandler(io.Discard, nil))
			if err := runServer(t.Context(), logger); err == nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("startup error = %v, want %q before checking the occupied port", err, test.want)
			}
			for id, want := range map[string]string{processing.ID: model.Processing, queued.ID: model.Queued} {
				doc, err := db.Document(t.Context(), id)
				if err != nil || doc.Status != want || doc.Failure != nil {
					t.Fatalf("document %s = %+v, err = %v, want status %s and no failure", id, doc, err, want)
				}
			}
		})
	}
}
