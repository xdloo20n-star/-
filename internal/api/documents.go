package api

import (
	"context"
	"crypto/rand"
	"errors"
	"io"
	"mime"
	"net/http"
	"os"
	"path"
	"regexp"
	"strings"
	"time"

	"sverka/internal/model"
)

const multipartOverhead = 64 << 10

var corpusFilename = regexp.MustCompile(`^([DP][0-9]{3})\.pdf$`)

func (s *Server) uploadDocument(w http.ResponseWriter, r *http.Request) {
	r.Body = http.MaxBytesReader(w, r.Body, s.maxPDFBytes+multipartOverhead)
	parts, err := r.MultipartReader()
	if err != nil {
		invalidUpload(w, err)
		return
	}
	part, err := parts.NextPart()
	if err != nil {
		invalidUpload(w, err)
		return
	}
	filename := path.Base(strings.ReplaceAll(part.FileName(), "\\", "/"))
	if part.FormName() != "file" || filename == "." || filename == "/" || filename == ".." {
		invalidUpload(w, nil)
		return
	}
	var signature [5]byte
	if _, err := io.ReadFull(part, signature[:]); err != nil && !errors.Is(err, io.EOF) && !errors.Is(err, io.ErrUnexpectedEOF) {
		invalidUpload(w, err)
		return
	}
	if string(signature[:]) != "%PDF-" {
		problem(w, http.StatusUnsupportedMediaType, "invalid_pdf", "Файл не распознан как PDF.")
		return
	}

	id := "doc_" + strings.ToLower(rand.Text())
	filePath := s.pdfPath(id)
	file, err := os.OpenFile(filePath, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0600)
	if err != nil {
		s.storeError(w, r, err)
		return
	}
	queued := false
	defer func() {
		_ = file.Close()
		if !queued {
			if err := os.Remove(filePath); err != nil && !errors.Is(err, os.ErrNotExist) {
				s.log.Error("remove incomplete upload", "document_id", id, "error", err)
			}
		}
	}()
	if _, err := file.Write(signature[:]); err != nil {
		s.storeError(w, r, err)
		return
	}
	remaining := s.maxPDFBytes - int64(len(signature))
	written, err := io.Copy(file, io.LimitReader(part, remaining+1))
	if written > remaining {
		uploadTooLarge(w)
		return
	}
	if err != nil {
		var pathErr *os.PathError
		if errors.As(err, &pathErr) {
			s.storeError(w, r, err)
		} else {
			invalidUpload(w, err)
		}
		return
	}
	if _, err := parts.NextPart(); err != io.EOF {
		invalidUpload(w, err)
		return
	}
	if err := file.Sync(); err != nil {
		s.storeError(w, r, err)
		return
	}
	if err := file.Close(); err != nil {
		s.storeError(w, r, err)
		return
	}

	doc := model.Document{
		ID: id, OriginalFilename: filename, Status: model.Queued, ExtractionMode: s.mode,
		FileURL:    "/api/v1/documents/" + id + "/file",
		UploadedAt: time.Now().UTC().Format("2006-01-02T15:04:05.000000000Z"),
	}
	if match := corpusFilename.FindStringSubmatch(filename); match != nil {
		doc.CorpusID = &match[1]
	}
	ctx, cancel := context.WithTimeout(context.WithoutCancel(r.Context()), 5*time.Second)
	defer cancel()
	if err := s.store.Enqueue(ctx, doc); err != nil {
		s.storeError(w, r, err)
		return
	}
	queued = true
	select {
	case s.wake <- struct{}{}:
	default:
	}
	w.Header().Set("Location", "/api/v1/documents/"+id)
	writeJSON(w, http.StatusAccepted, doc)
}

func invalidUpload(w http.ResponseWriter, err error) {
	var exceeded *http.MaxBytesError
	if errors.As(err, &exceeded) {
		uploadTooLarge(w)
		return
	}
	problem(w, http.StatusBadRequest, "invalid_upload", "Передайте один PDF в multipart-поле file.")
}

func uploadTooLarge(w http.ResponseWriter) {
	problem(w, http.StatusRequestEntityTooLarge, "upload_too_large", "Файл превышает допустимый размер.")
}

func (s *Server) listDocuments(w http.ResponseWriter, r *http.Request) {
	page, err := pagination(r.URL.Query())
	if err != nil {
		badQuery(w, err)
		return
	}
	result, err := s.store.ListDocuments(r.Context(), page)
	if err != nil {
		s.storeError(w, r, err)
		return
	}
	writeJSON(w, http.StatusOK, result)
}

func (s *Server) document(w http.ResponseWriter, r *http.Request) {
	doc, err := s.store.DocumentDetail(r.Context(), r.PathValue("id"))
	if err != nil {
		s.storeError(w, r, err)
		return
	}
	writeJSON(w, http.StatusOK, doc)
}

func (s *Server) documentFile(w http.ResponseWriter, r *http.Request) {
	doc, err := s.store.Document(r.Context(), r.PathValue("id"))
	if err != nil {
		s.storeError(w, r, err)
		return
	}
	file, err := os.Open(s.pdfPath(doc.ID))
	if errors.Is(err, os.ErrNotExist) {
		problem(w, http.StatusNotFound, "file_not_found", "Исходный PDF не найден.")
		return
	}
	if err != nil {
		s.storeError(w, r, err)
		return
	}
	defer file.Close()
	info, err := file.Stat()
	if err != nil {
		s.storeError(w, r, err)
		return
	}
	w.Header().Set("Content-Type", "application/pdf")
	w.Header().Set("Content-Disposition", mime.FormatMediaType("inline", map[string]string{"filename": doc.OriginalFilename}))
	w.Header().Set("X-Content-Type-Options", "nosniff")
	content := &pdfResponse{ResponseWriter: w}
	http.ServeContent(content, r, doc.OriginalFilename, info.ModTime(), file)
	if content.errorStatus != 0 {
		w.Header().Del("Content-Length")
		w.Header().Del("Content-Disposition")
		if content.errorStatus == http.StatusRequestedRangeNotSatisfiable {
			problem(w, content.errorStatus, "invalid_range", "Запрошенный диапазон выходит за пределы PDF.")
		} else {
			problem(w, content.errorStatus, "file_unavailable", "Не удалось выдать PDF для этого запроса.")
		}
	}
}

// Keep error responses from ServeContent in the API's JSON format.
type pdfResponse struct {
	http.ResponseWriter
	errorStatus int
}

func (r *pdfResponse) WriteHeader(status int) {
	if status >= 400 {
		r.errorStatus = status
		return
	}
	r.ResponseWriter.WriteHeader(status)
}

func (r *pdfResponse) Write(data []byte) (int, error) {
	if r.errorStatus != 0 {
		return len(data), nil
	}
	return r.ResponseWriter.Write(data)
}
