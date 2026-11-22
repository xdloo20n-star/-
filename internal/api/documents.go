package api

import (
	"context"
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
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
	"sverka/internal/store"
)

const multipartOverhead = 64 << 10

var corpusIDPattern = regexp.MustCompile(`^[DP][0-9]{3}$`)

func (s *Server) uploadDocument(w http.ResponseWriter, r *http.Request) {
	query, ok := requestQuery(w, r)
	if !ok {
		return
	}
	var reuse bool
	if values, present := query["reuse"]; present {
		if len(values) != 1 || (values[0] != "true" && values[0] != "false") {
			badQuery(w, errors.New("reuse должен быть true или false"))
			return
		}
		reuse = values[0] == "true"
	}
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
	corpusID, pdfSuffix := strings.CutSuffix(filename, ".pdf")
	isCorpus := pdfSuffix && corpusIDPattern.MatchString(corpusID)
	if reuse && !isCorpus {
		badQuery(w, errors.New("reuse=true требует PDF с именем корпуса, например D001.pdf"))
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

	file, err := os.CreateTemp(s.pdfDir, uploadTempPrefix)
	if err != nil {
		s.storeError(w, r, err)
		return
	}
	defer func() {
		_ = file.Close()
		s.removeUpload(file.Name())
	}()
	digest := sha256.New()
	var target io.Writer = file
	if reuse {
		target = io.MultiWriter(file, digest)
	}
	if _, err := target.Write(signature[:]); err != nil {
		s.storeError(w, r, err)
		return
	}
	remaining := s.maxPDFBytes - int64(len(signature))
	written, err := io.Copy(target, io.LimitReader(part, remaining+1))
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
	id := "doc_" + strings.ToLower(rand.Text())
	doc := model.Document{
		ID: id, OriginalFilename: filename, Status: model.Queued, ExtractionMode: s.mode,
		FileURL:    "/api/v1/documents/" + id + "/file",
		UploadedAt: time.Now().UTC().Format("2006-01-02T15:04:05.000000000Z"),
	}
	if isCorpus {
		doc.CorpusID = &corpusID
	}
	ctx, cancel := context.WithTimeout(context.WithoutCancel(r.Context()), 5*time.Second)
	defer cancel()
	var sha256Hex string
	if reuse {
		sha256Hex = hex.EncodeToString(digest.Sum(nil))
	}
	doc, queued, err := s.queueUpload(ctx, doc, file, sha256Hex)
	var conflict reuseConflict
	if errors.As(err, &conflict) {
		problem(w, http.StatusConflict, "reuse_conflict", conflict.Error())
		return
	}
	if err != nil {
		s.storeError(w, r, err)
		return
	}
	if queued {
		select {
		case s.wake <- struct{}{}:
		default:
		}
	}
	w.Header().Set("Location", "/api/v1/documents/"+doc.ID)
	writeJSON(w, http.StatusAccepted, doc)
}

type reuseConflict string

func (e reuseConflict) Error() string { return string(e) }

func (s *Server) queueUpload(ctx context.Context, doc model.Document, file *os.File, sha256Hex string) (model.Document, bool, error) {
	if doc.CorpusID != nil {
		s.corpusUploads.Lock()
		defer s.corpusUploads.Unlock()
	}
	if sha256Hex != "" {
		matches, err := s.store.ListDocuments(ctx, store.Pagination{Page: 1, Size: 1}, *doc.CorpusID)
		if err != nil {
			return model.Document{}, false, err
		}
		if matches.Total > 1 {
			return model.Document{}, false, reuseConflict("Для этого ID корпуса найдено несколько заданий.")
		}
		if matches.Total == 1 {
			existing := matches.Items[0]
			if existing.ExtractionMode != doc.ExtractionMode {
				return model.Document{}, false, reuseConflict("Режим извлечения существующего задания отличается.")
			}
			existingDigest, err := s.pdfSHA256(existing.ID)
			if errors.Is(err, os.ErrNotExist) {
				return model.Document{}, false, reuseConflict("Исходный PDF существующего задания не найден.")
			}
			if err != nil {
				return model.Document{}, false, err
			}
			if existingDigest != sha256Hex {
				return model.Document{}, false, reuseConflict("Содержимое PDF существующего задания отличается.")
			}
			return existing, false, nil
		}
	}
	if err := ctx.Err(); err != nil {
		return model.Document{}, false, err
	}
	if err := s.publishUpload(ctx, doc, file); err != nil {
		return model.Document{}, false, err
	}
	return doc, true, nil
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
	query, ok := requestQuery(w, r)
	if !ok {
		return
	}
	page, err := pagination(query)
	if err != nil {
		badQuery(w, err)
		return
	}
	corpusID := query.Get("corpus_id")
	if values, present := query["corpus_id"]; present && (len(values) != 1 || !corpusIDPattern.MatchString(corpusID)) {
		badQuery(w, errors.New("некорректный идентификатор corpus_id"))
		return
	}
	result, err := s.store.ListDocuments(r.Context(), page, corpusID)
	if err != nil {
		s.storeError(w, r, err)
		return
	}
	writeJSON(w, http.StatusOK, struct {
		store.Page[model.Document]
		CorpusID string `json:"corpus_id,omitempty"`
	}{Page: result, CorpusID: corpusID})
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
