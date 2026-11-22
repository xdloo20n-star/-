package api

import (
	"bytes"
	"crypto/rand"
	"crypto/sha256"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"sverka"
	"sverka/internal/extract"
	"sverka/internal/model"
	"sverka/internal/store"
)

const uploadPath = "/api/v1/documents"

func uploadRequest(t *testing.T, query, filename, pdf string) *http.Request {
	t.Helper()
	var body bytes.Buffer
	writer := multipart.NewWriter(&body)
	part, err := writer.CreateFormFile("file", filename)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := io.WriteString(part, pdf); err != nil {
		t.Fatal(err)
	}
	if err := writer.Close(); err != nil {
		t.Fatal(err)
	}
	req := httptest.NewRequestWithContext(t.Context(), http.MethodPost, uploadPath+query, &body)
	req.Header.Set("Content-Type", writer.FormDataContentType())
	return req
}

func postPDF(t *testing.T, h harness, query, filename, pdf string, status int) []byte {
	t.Helper()
	result := httptest.NewRecorder()
	h.app.Handler().ServeHTTP(result, uploadRequest(t, query, filename, pdf))
	if result.Code != status {
		t.Fatalf("POST %s status = %d, want %d, body %s", query, result.Code, status, result.Body)
	}
	return result.Body.Bytes()
}

func TestConcurrentReuseUpload(t *testing.T) {
	h := newHarness(t, extract.MockExtractor{Fixture: sverka.MockExtraction}, time.Second)
	handler := h.app.Handler()
	const pdf = "%PDF-1.7\nresume test\n"
	responses := make(chan *httptest.ResponseRecorder, 12)
	begin := make(chan struct{})
	for range cap(responses) {
		req := uploadRequest(t, "?reuse=true", "D001.pdf", pdf)
		go func() {
			<-begin
			result := httptest.NewRecorder()
			handler.ServeHTTP(result, req)
			responses <- result
		}()
	}
	close(begin)
	var first model.Document
	for range cap(responses) {
		result := <-responses
		if result.Code != http.StatusAccepted {
			t.Errorf("status = %d, body %s", result.Code, result.Body)
			continue
		}
		doc := decode[model.Document](t, result.Body.Bytes())
		if first.ID == "" {
			first = doc
		}
		equal(t, "document ID", first.ID, doc.ID)
		equal(t, "Location", uploadPath+"/"+doc.ID, result.Header().Get("Location"))
	}
	page := get[store.Page[model.Document]](t, h, uploadPath)
	equal(t, "queued job count", 1, page.Total)
	entries, err := os.ReadDir(h.app.pdfDir)
	if err != nil {
		t.Fatal(err)
	}
	for _, entry := range entries {
		if entry.Name() != first.ID+".pdf" && (storedUploadName.MatchString(entry.Name()) || stagedUploadName.MatchString(entry.Name())) {
			t.Fatalf("unexpected upload file %s", entry.Name())
		}
	}
	doc, err := h.app.store.Claim(t.Context())
	if err != nil {
		t.Fatal(err)
	}
	if err := h.app.store.Fail(t.Context(), doc.ID, model.Problem{Code: "test", Message: "test failure"}); err != nil {
		t.Fatal(err)
	}
	reused := decode[model.Document](t, postPDF(t, h, "?reuse=true", "D001.pdf", pdf, http.StatusAccepted))
	equal(t, "reused ID", first.ID, reused.ID)
	equal(t, "current status", model.Failed, reused.Status)
	if reused.Failure == nil || reused.Failure.Code != "test" {
		t.Fatalf("unexpected reused failure: %+v", reused.Failure)
	}
}

func TestReuseConflictsAndHistoricalDigest(t *testing.T) {
	h := newHarness(t, extract.MockExtractor{Fixture: sverka.MockExtraction}, time.Second)
	const pdf = "%PDF-1.7\nprevious upload\n"
	first := decode[model.Document](t, postPDF(t, h, "", "D001.pdf", pdf, http.StatusAccepted))
	digest := get[map[string]string](t, h, uploadPath+"/"+first.ID+"/digest")
	equal(t, "digest fields", 1, len(digest))
	equal(t, "stored digest", fmt.Sprintf("%x", sha256.Sum256([]byte(pdf))), digest["sha256"])
	body := postPDF(t, h, "?reuse=true", "D001.pdf", pdf+"different bytes", http.StatusConflict)
	equal(t, "PDF conflict code", "reuse_conflict", decode[errorResponse](t, body).Error.Code)
	second := decode[model.Document](t, postPDF(t, h, "?reuse=false", "D001.pdf", pdf, http.StatusAccepted))
	if first.ID == second.ID {
		t.Fatal("ordinary upload reused an existing job")
	}
	body = postPDF(t, h, "?reuse=true", "D001.pdf", pdf, http.StatusConflict)
	equal(t, "multiple jobs conflict code", "reuse_conflict", decode[errorResponse](t, body).Error.Code)

	corpusID := "P003"
	historical := model.Document{ID: "doc_historical", CorpusID: &corpusID, OriginalFilename: "P003.pdf", ExtractionMode: "python"}
	if err := h.app.store.Enqueue(t.Context(), historical); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(h.app.pdfPath(historical.ID), []byte(pdf), 0600); err != nil {
		t.Fatal(err)
	}
	pinDigest := get[map[string]string](t, h, uploadPath+"/"+historical.ID+"/digest")
	equal(t, "historical pin digest", digest["sha256"], pinDigest["sha256"])
	body = postPDF(t, h, "?reuse=true", "P003.pdf", pdf, http.StatusConflict)
	equal(t, "mode conflict code", "reuse_conflict", decode[errorResponse](t, body).Error.Code)
	missing := decode[model.Document](t, postPDF(t, h, "", "D002.pdf", pdf, http.StatusAccepted))
	if err := os.Remove(h.app.pdfPath(missing.ID)); err != nil {
		t.Fatal(err)
	}
	body = request(t, http.MethodGet, h.url+uploadPath+"/"+missing.ID+"/digest", "", nil, http.StatusNotFound)
	equal(t, "missing PDF code", "file_not_found", decode[errorResponse](t, body).Error.Code)
	body = request(t, http.MethodGet, h.url+uploadPath+"/doc_unknown/digest", "", nil, http.StatusNotFound)
	equal(t, "missing document code", "not_found", decode[errorResponse](t, body).Error.Code)
	body = postPDF(t, h, "?reuse=true", "D002.pdf", pdf, http.StatusConflict)
	equal(t, "missing original conflict code", "reuse_conflict", decode[errorResponse](t, body).Error.Code)
	equal(t, "job count after conflicts", 4, get[store.Page[model.Document]](t, h, uploadPath).Total)
}

func TestCorpusFilterAndMalformedQueries(t *testing.T) {
	h := newHarness(t, extract.MockExtractor{Fixture: sverka.MockExtraction}, time.Second)
	const pdf = "%PDF-1.7\n"
	for _, filename := range []string{"D001.pdf", "P001.pdf", "D001.pdf", "article.pdf"} {
		postPDF(t, h, "", filename, pdf, http.StatusAccepted)
	}
	type filteredPage struct {
		store.Page[model.Document]
		CorpusID *string `json:"corpus_id"`
	}
	for _, test := range []struct {
		query string
		id    string
		total int
		items int
	}{
		{"?corpus_id=D001&page_size=1&page=2", "D001", 2, 1},
		{"?corpus_id=D001&page_size=1&page=3", "D001", 2, 0},
		{"?corpus_id=P999&page_size=100", "P999", 0, 0},
	} {
		page := get[filteredPage](t, h, uploadPath+test.query)
		if page.CorpusID == nil || *page.CorpusID != test.id {
			t.Fatalf("%s did not echo %s", test.query, test.id)
		}
		equal(t, "filtered total", test.total, page.Total)
		if page.Items == nil || len(page.Items) != test.items {
			t.Fatalf("%s item count = %d, want %d", test.query, len(page.Items), test.items)
		}
		for _, doc := range page.Items {
			if doc.CorpusID == nil || *doc.CorpusID != test.id {
				t.Fatalf("%s returned nonmatching document %+v", test.query, doc)
			}
		}
	}
	page := get[filteredPage](t, h, uploadPath)
	equal(t, "unfiltered corpus field", (*string)(nil), page.CorpusID)
	equal(t, "unfiltered default size", 25, page.PageSize)
	for _, suffix := range []string{
		uploadPath + "?corpus_id=", uploadPath + "?corpus_id=d001", uploadPath + "?corpus_id=D01",
		uploadPath + "?corpus_id=D001&corpus_id=P001", uploadPath + "?corpus_id=D001&page_size=101",
		uploadPath + "?corpus_id=D001&ignored=%ZZ", "/api/v1/results?page=1;q=x",
		"/api/v1/groups?ignored=%", "/api/v1/export?document_id=doc_missing&ignored=%ZZ",
	} {
		body := request(t, http.MethodGet, h.url+suffix, "", nil, http.StatusBadRequest)
		equal(t, "invalid query code", "invalid_query", decode[errorResponse](t, body).Error.Code)
	}
	for _, query := range []string{"?reuse=true&ignored=%ZZ", "?reuse=1", "?reuse=", "?reuse=true&reuse=false"} {
		body := postPDF(t, h, query, "D003.pdf", pdf, http.StatusBadRequest)
		equal(t, "upload query error", "invalid_query", decode[errorResponse](t, body).Error.Code)
	}
	body := postPDF(t, h, "?reuse=true", "article.pdf", pdf, http.StatusBadRequest)
	equal(t, "non-corpus reuse error", "invalid_query", decode[errorResponse](t, body).Error.Code)
	equal(t, "job count after invalid requests", 4, get[store.Page[model.Document]](t, h, uploadPath).Total)
}

func TestUploadStaysStagedUntilMultipartValidated(t *testing.T) {
	h := newHarness(t, extract.MockExtractor{Fixture: sverka.MockExtraction}, time.Second)
	reader, writer := io.Pipe()
	defer reader.Close()
	defer writer.Close()
	multipartWriter := multipart.NewWriter(writer)
	req := httptest.NewRequestWithContext(t.Context(), http.MethodPost, uploadPath, reader)
	req.Header.Set("Content-Type", multipartWriter.FormDataContentType())
	done := make(chan *httptest.ResponseRecorder, 1)
	go func() {
		response := httptest.NewRecorder()
		h.app.Handler().ServeHTTP(response, req)
		done <- response
	}()
	part, err := multipartWriter.CreateFormFile("file", "D001.pdf")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := io.WriteString(part, "%PDF-1.7\n"); err != nil {
		t.Fatal(err)
	}
	deadline := time.Now().Add(5 * time.Second)
	for {
		files, err := os.ReadDir(h.app.pdfDir)
		if err != nil {
			t.Fatal(err)
		}
		staged := false
		for _, file := range files {
			if storedUploadName.MatchString(file.Name()) {
				t.Fatalf("published %s before validating the body", file.Name())
			}
			staged = staged || stagedUploadName.MatchString(file.Name())
		}
		if staged {
			break
		}
		if time.Now().After(deadline) {
			t.Fatal("upload did not reach the staging directory")
		}
		time.Sleep(5 * time.Millisecond)
	}
	// The second part makes the request invalid after the file has been staged.
	if _, err := multipartWriter.CreateFormField("extra"); err != nil {
		t.Fatal(err)
	}
	// Close the pipe directly since the handler stops reading when it sees an
	// extra part and a multipart closing boundary could block.
	_ = writer.Close()
	result := <-done
	equal(t, "multipart response", http.StatusBadRequest, result.Code)
	files, err := os.ReadDir(h.app.pdfDir)
	if err != nil {
		t.Fatal(err)
	}
	for _, file := range files {
		if stagedUploadName.MatchString(file.Name()) || storedUploadName.MatchString(file.Name()) {
			t.Fatalf("invalid upload left file %s", file.Name())
		}
	}
	equal(t, "jobs after invalid upload", 0, get[store.Page[model.Document]](t, h, uploadPath).Total)
}

func TestRecoverFilesRemovesAbandonedUploads(t *testing.T) {
	h := newHarness(t, extract.MockExtractor{Fixture: sverka.MockExtraction}, time.Second)
	const pdf = "%PDF-1.7\n"
	doc := decode[model.Document](t, postPDF(t, h, "", "D001.pdf", pdf, http.StatusAccepted))
	staged, err := os.CreateTemp(h.app.pdfDir, uploadTempPrefix)
	if err != nil {
		t.Fatal(err)
	}
	if err := staged.Close(); err != nil {
		t.Fatal(err)
	}
	orphan := h.app.pdfPath("doc_" + strings.ToLower(rand.Text()))
	unrelated := filepath.Join(h.app.pdfDir, "archived.pdf")
	for _, file := range []string{orphan, unrelated} {
		if err := os.WriteFile(file, []byte(pdf), 0600); err != nil {
			t.Fatal(err)
		}
	}
	for range 2 {
		if err := h.app.RecoverFiles(t.Context()); err != nil {
			t.Fatal(err)
		}
	}
	for _, name := range []string{staged.Name(), orphan} {
		if _, err := os.Stat(name); !os.IsNotExist(err) {
			t.Fatalf("expected removal of %s, stat error = %v", name, err)
		}
	}
	for _, name := range []string{h.app.pdfPath(doc.ID), unrelated} {
		if content, err := os.ReadFile(name); err != nil || string(content) != pdf {
			t.Fatalf("expected %s preserved, content = %q, error = %v", name, content, err)
		}
	}
}
