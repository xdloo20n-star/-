package api

import (
	"context"
	"crypto/sha256"
	"database/sql"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"regexp"
	"strings"

	"sverka/internal/model"
)

const uploadTempPrefix = ".sverka-upload-"

var (
	stagedUploadName = regexp.MustCompile(`^` + regexp.QuoteMeta(uploadTempPrefix) + `[0-9]+$`)
	storedUploadName = regexp.MustCompile(`^doc_[a-z2-7]{26,}\.pdf$`)
)

// RecoverFiles removes abandoned upload files. Call once before serving HTTP or
// starting workers, with exclusive access to the PDF directory and database.
func (s *Server) RecoverFiles(ctx context.Context) error {
	entries, err := os.ReadDir(s.pdfDir)
	if err != nil {
		return err
	}
	for _, entry := range entries {
		if err := ctx.Err(); err != nil {
			return err
		}
		if !entry.Type().IsRegular() {
			continue
		}
		name := entry.Name()
		switch {
		case stagedUploadName.MatchString(name):
		case storedUploadName.MatchString(name):
			id := strings.TrimSuffix(name, ".pdf")
			if _, err := s.store.Document(ctx, id); err == nil {
				continue
			} else if !errors.Is(err, sql.ErrNoRows) {
				return fmt.Errorf("look up PDF %s: %w", name, err)
			}
		default:
			continue
		}
		if err := os.Remove(filepath.Join(s.pdfDir, name)); err != nil && !errors.Is(err, os.ErrNotExist) {
			return fmt.Errorf("remove abandoned upload %s: %w", name, err)
		}
	}
	return nil
}

func (s *Server) publishUpload(ctx context.Context, doc model.Document, file *os.File) error {
	if err := file.Sync(); err != nil {
		return err
	}
	if err := file.Close(); err != nil {
		return err
	}
	final := s.pdfPath(doc.ID)
	// Publish atomically and leave an existing destination intact on ID collision.
	if err := os.Link(file.Name(), final); err != nil {
		return err
	}
	dir, err := os.Open(s.pdfDir)
	if err == nil {
		err = dir.Sync()
		_ = dir.Close()
	}
	if err != nil {
		s.removeUpload(final)
		return err
	}
	if err := s.store.Enqueue(ctx, doc); err != nil {
		// SQLite cancellation can race with commit. Let startup reconcile the
		// final file if the outcome was uncertain.
		if ctx.Err() == nil {
			s.removeUpload(final)
		}
		return err
	}
	return nil
}

func (s *Server) removeUpload(filePath string) {
	if err := os.Remove(filePath); err != nil && !errors.Is(err, os.ErrNotExist) {
		s.log.Error("remove incomplete upload", "path", filePath, "error", err)
	}
}

func (s *Server) pdfSHA256(id string) (string, error) {
	file, err := os.Open(s.pdfPath(id))
	if err != nil {
		return "", err
	}
	defer file.Close()
	digest := sha256.New()
	if _, err := io.Copy(digest, file); err != nil {
		return "", err
	}
	return hex.EncodeToString(digest.Sum(nil)), nil
}

func (s *Server) documentDigest(w http.ResponseWriter, r *http.Request) {
	doc, err := s.store.Document(r.Context(), r.PathValue("id"))
	if err != nil {
		s.storeError(w, r, err)
		return
	}
	digest, err := s.pdfSHA256(doc.ID)
	if errors.Is(err, os.ErrNotExist) {
		problem(w, http.StatusNotFound, "file_not_found", "Исходный PDF не найден.")
		return
	}
	if err != nil {
		s.storeError(w, r, err)
		return
	}
	writeJSON(w, http.StatusOK, map[string]string{"sha256": digest})
}
