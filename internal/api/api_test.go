package api

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"sverka"
	"sverka/internal/extract"
	"sverka/internal/model"
	"sverka/internal/store"
)

type testExtractor func(context.Context, string) ([]byte, error)

func (testExtractor) Mode() string { return "mock" }

func (e testExtractor) Extract(ctx context.Context, path string) ([]byte, error) {
	return e(ctx, path)
}

type harness struct {
	app *Server
	url string
}

func newHarness(t *testing.T, source extract.Extractor, timeout time.Duration) harness {
	t.Helper()
	dir := t.TempDir()
	db, err := store.Open(t.Context(), filepath.Join(dir, "test.sqlite"))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = db.Close() })
	importer, err := extract.NewImporter(sverka.ExtractorSchema)
	if err != nil {
		t.Fatal(err)
	}
	s := New(Config{
		PDFDir: dir, ExtractionMode: source.Mode(), MaxPDFBytes: 512,
		ExtractionTimeout: timeout, CORSOrigin: "http://localhost:5173",
	}, db, importer, slog.New(slog.NewTextHandler(io.Discard, nil)), source)
	web := httptest.NewServer(s.Handler())
	t.Cleanup(web.Close)
	return harness{app: s, url: web.URL}
}

func (h harness) startWorker(t *testing.T) {
	t.Helper()
	ctx, cancel := context.WithCancel(t.Context())
	done := make(chan struct{})
	go func() {
		defer close(done)
		h.app.RunWorker(ctx)
	}()
	t.Cleanup(func() {
		cancel()
		<-done
	})
}

func request(t *testing.T, method, endpoint, contentType string, body io.Reader, status int, headers ...[2]string) []byte {
	t.Helper()
	req, err := http.NewRequestWithContext(t.Context(), method, endpoint, body)
	if err != nil {
		t.Fatal(err)
	}
	if contentType != "" {
		req.Header.Set("Content-Type", contentType)
	}
	for _, header := range headers {
		req.Header.Set(header[0], header[1])
	}
	response, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer response.Body.Close()
	data, err := io.ReadAll(response.Body)
	if err != nil {
		t.Fatal(err)
	}
	if response.StatusCode != status {
		t.Fatalf("%s %s status expected: %d, got: %d, body: %s", method, endpoint, status, response.StatusCode, data)
	}
	return data
}

func upload(t *testing.T, h harness, pdf string, status int) []byte {
	t.Helper()
	var body bytes.Buffer
	writer := multipart.NewWriter(&body)
	part, err := writer.CreateFormFile("file", "D019.pdf")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := io.WriteString(part, pdf); err != nil {
		t.Fatal(err)
	}
	if err := writer.Close(); err != nil {
		t.Fatal(err)
	}
	return request(t, http.MethodPost, h.url+"/api/v1/documents", writer.FormDataContentType(), &body, status)
}

func decode[T any](t *testing.T, raw []byte) T {
	t.Helper()
	var value T
	if err := json.Unmarshal(raw, &value); err != nil {
		t.Fatal(err)
	}
	return value
}

func equal[T comparable](t *testing.T, field string, expected, got T) {
	t.Helper()
	if expected != got {
		t.Fatalf("%s expected: %v, got: %v", field, expected, got)
	}
}

func get[T any](t *testing.T, h harness, path string) T {
	t.Helper()
	return decode[T](t, request(t, http.MethodGet, h.url+path, "", nil, http.StatusOK))
}

func waitFor(t *testing.T, h harness, id, status string) model.DocumentDetail {
	t.Helper()
	path := "/api/v1/documents/" + id
	deadline := time.Now().Add(5 * time.Second)
	for {
		doc := get[model.DocumentDetail](t, h, path)
		if doc.Document.Status == status {
			return doc
		}
		if doc.Document.Status == model.Completed || doc.Document.Status == model.Failed || time.Now().After(deadline) {
			t.Fatalf("document status expected: %s, got: %s (%+v)", status, doc.Document.Status, doc.Document.Failure)
		}
		time.Sleep(5 * time.Millisecond)
	}
}

func TestHTTPUploadWorkflow(t *testing.T) {
	release := make(chan struct{})
	h := newHarness(t, testExtractor(func(ctx context.Context, _ string) ([]byte, error) {
		select {
		case <-release:
			return sverka.MockExtraction, nil
		case <-ctx.Done():
			return nil, ctx.Err()
		}
	}), 3*time.Second)

	request(t, http.MethodGet, h.url+"/healthz", "", nil, http.StatusOK)
	const pdf = "%PDF-1.7\nsubmission fixture\n"
	accepted := decode[model.Document](t, upload(t, h, pdf, http.StatusAccepted))
	equal(t, "accepted status", model.Queued, accepted.Status)
	equal(t, "accepted mode", "mock", accepted.ExtractionMode)
	if accepted.CorpusID == nil || *accepted.CorpusID != "D019" {
		t.Fatalf("corpus_id expected: D019, got: %v", accepted.CorpusID)
	}
	waitFor(t, h, accepted.ID, model.Queued)

	h.startWorker(t)
	processing := waitFor(t, h, accepted.ID, model.Processing)
	equal(t, "processing data", (*model.DocumentData)(nil), processing.Data)
	close(release)
	completed := waitFor(t, h, accepted.ID, model.Completed)
	if completed.Data == nil {
		t.Fatal("completed data expected: present, got: nil")
	}
	equal(t, "document result count", 2, len(completed.Data.Results))

	fullPDF := request(t, http.MethodGet, h.url+accepted.FileURL, "", nil, http.StatusOK)
	equal(t, "original PDF bytes", pdf, string(fullPDF))
	rangePDF := request(t, http.MethodGet, h.url+accepted.FileURL, "", nil,
		http.StatusPartialContent, [2]string{"Range", "bytes=0-4"})
	equal(t, "PDF range bytes", "%PDF-", string(rangePDF))
	rangeErr := request(t, http.MethodGet, h.url+accepted.FileURL, "", nil,
		http.StatusRequestedRangeNotSatisfiable, [2]string{"Range", "bytes=999-1000"})
	equal(t, "range error", "invalid_range", decode[errorResponse](t, rangeErr).Error.Code)

	filter := url.Values{"document_id": {accepted.ID}, "metric_name": {"recall@5"}, "q": {"syntheticqa"}, "has_issues": {"false"}, "page_size": {"1"}}
	page := get[store.Page[model.ResultSummary]](t, h, "/api/v1/results?"+filter.Encode())
	equal(t, "filtered total", 2, page.Total)
	equal(t, "filtered page length", 1, len(page.Items))
	filter.Set("page", "3")
	equal(t, "empty page length", 0, len(get[store.Page[model.ResultSummary]](t, h, "/api/v1/results?"+filter.Encode()).Items))

	left, right := completed.Data.Results[0].Result.Text("result_id"), completed.Data.Results[1].Result.Text("result_id")
	detail := get[model.ResultDetail](t, h, "/api/v1/results/"+left)
	equal(t, "result mode", "mock", detail.ExtractionMode)
	equal(t, "result source count", 2, len(detail.Evidence))
	equal(t, "source result link", left, detail.Evidence[0].Text("entity_id"))
	equal(t, "related config link", detail.Result.Text("config_id"), detail.Configuration.Text("config_id"))
	compare := strings.NewReader(`{"left_result_id":"` + left + `","right_result_id":"` + right + `"}`)
	decision := decode[model.Comparison](t, request(t, http.MethodPost, h.url+"/api/v1/comparisons", "application/json", compare, http.StatusOK))
	equal(t, "comparison status", model.Comparable, decision.Status)
	if decision.Numeric == nil {
		t.Fatal("numeric comparison expected: present, got: nil")
	}
	equal(t, "comparison preferred side", "right", decision.Numeric.Preferred)

	other := decode[model.Document](t, upload(t, h, pdf, http.StatusAccepted))
	second := waitFor(t, h, other.ID, model.Completed)
	if second.Data.Results[0].Result.Text("result_id") == left || second.Data.Configurations[0].Text("config_id") == detail.Configuration.Text("config_id") {
		t.Fatal("IDs expected: scoped to separate documents, got: overlapping IDs")
	}
	export := get[struct {
		SchemaVersion string                 `json:"schema_version"`
		Documents     []model.DocumentDetail `json:"documents"`
	}](t, h, "/api/v1/export?document_id="+accepted.ID)
	equal(t, "export version", "1.0", export.SchemaVersion)
	equal(t, "export document count", 1, len(export.Documents))
	equal(t, "export mode", "mock", export.Documents[0].Document.ExtractionMode)
	equal(t, "export result count", 2, len(export.Documents[0].Data.Results))
}

type errorResponse struct {
	Error model.Problem `json:"error"`
}

func TestInvalidUploads(t *testing.T) {
	h := newHarness(t, extract.MockExtractor{Fixture: sverka.MockExtraction}, time.Second)
	data := upload(t, h, "not a PDF", http.StatusUnsupportedMediaType)
	equal(t, "invalid PDF error", "invalid_pdf", decode[errorResponse](t, data).Error.Code)
	data = upload(t, h, "%PDF-"+strings.Repeat("a", 508), http.StatusRequestEntityTooLarge)
	equal(t, "upload limit error", "upload_too_large", decode[errorResponse](t, data).Error.Code)
	equal(t, "document count", 0, get[store.Page[model.Document]](t, h, "/api/v1/documents").Total)
}

func TestWorkerFailuresDoNotPublishPartialResults(t *testing.T) {
	for _, test := range []struct {
		name string
		code string
		run  testExtractor
	}{
		{"extractor error", "extraction_failed", func(context.Context, string) ([]byte, error) {
			return nil, errors.New("test failure")
		}},
		{"timeout", "extraction_timeout", func(ctx context.Context, _ string) ([]byte, error) {
			<-ctx.Done()
			return nil, ctx.Err()
		}},
		{"malformed JSON", "invalid_extraction", func(context.Context, string) ([]byte, error) {
			return []byte("{"), nil
		}},
		{"broken link after valid result", "invalid_extraction", func(context.Context, string) ([]byte, error) {
			return bytes.Replace(sverka.MockExtraction, []byte(`"baseline_result_id": "recall-dense"`), []byte(`"baseline_result_id": "missing"`), 1), nil
		}},
	} {
		t.Run(test.name, func(t *testing.T) {
			h := newHarness(t, test.run, 40*time.Millisecond)
			h.startWorker(t)
			accepted := decode[model.Document](t, upload(t, h, "%PDF-1.7\n", http.StatusAccepted))
			failed := waitFor(t, h, accepted.ID, model.Failed)
			equal(t, "failure code", test.code, failed.Document.Failure.Code)
			equal(t, "failed data", (*model.DocumentData)(nil), failed.Data)
			equal(t, "published results", 0, get[store.Page[model.ResultSummary]](t, h, "/api/v1/results").Total)
		})
	}
}

func TestPythonExtractorJob(t *testing.T) {
	python, err := exec.LookPath("python3")
	if err != nil {
		t.Skip("python3 unavailable")
	}
	fixture := bytes.Replace(sverka.MockExtraction, []byte(`"extraction_mode": "mock"`), []byte(`"extraction_mode": "python"`), 1)
	script := filepath.Join(t.TempDir(), "extractor with spaces.py")
	body := fmt.Sprintf(`
from pathlib import Path
import sys
import time

assert len(sys.argv) == 3 and sys.argv[1] == "--pdf"
pdf = Path(sys.argv[2])
assert pdf.is_absolute() and pdf.read_bytes().startswith(b"%%PDF-")
content = pdf.read_bytes()
if b"fail" in content:
    print("private diagnostic", file=sys.stderr)
    sys.exit(7)
if b"delay" in content:
    time.sleep(10)
print(%q)
`, string(fixture))
	if err := os.WriteFile(script, []byte(body), 0600); err != nil {
		t.Fatal(err)
	}
	h := newHarness(t, extract.PythonExtractor{Executable: python, Script: script}, 750*time.Millisecond)
	h.startWorker(t)

	success := decode[model.Document](t, upload(t, h, "%PDF-1.7\n", http.StatusAccepted))
	completed := waitFor(t, h, success.ID, model.Completed)
	equal(t, "Python document mode", "python", completed.Document.ExtractionMode)
	equal(t, "Python result mode", "python", completed.Data.Results[0].ExtractionMode)
	equal(t, "source review rule", "source_unverified", completed.Data.Issues[0].CheckName)

	for _, test := range []struct{ content, code string }{
		{"fail", "extraction_failed"},
		{"delay", "extraction_timeout"},
	} {
		accepted := decode[model.Document](t, upload(t, h, "%PDF-1.7\n"+test.content, http.StatusAccepted))
		failed := waitFor(t, h, accepted.ID, model.Failed)
		equal(t, "Python failure mode", "python", failed.Document.ExtractionMode)
		equal(t, "Python failure code", test.code, failed.Document.Failure.Code)
		equal(t, "Python failed data", (*model.DocumentData)(nil), failed.Data)
	}
}

func TestComparableGroupsEndpointReturnsEmptyListWithoutEligibleCrossPublicationPairs(t *testing.T) {
	h := newHarness(t, testExtractor(func(context.Context, string) ([]byte, error) {
		return sverka.MockExtraction, nil
	}), time.Second)
	response := decode[struct {
		Groups []model.ComparableGroup `json:"groups"`
	}](t, request(t, http.MethodGet, h.url+"/api/v1/groups", "", nil, http.StatusOK))
	equal(t, "empty comparable groups", 0, len(response.Groups))
	if response.Groups == nil {
		t.Fatal("expected groups to be an empty JSON array")
	}
}

func TestComparableGroupsEndpointGroupsResultsFromDistinctPublications(t *testing.T) {
	source := testExtractor(func(_ context.Context, path string) ([]byte, error) {
		content, err := os.ReadFile(path)
		if err != nil {
			return nil, err
		}
		title := "Paper A"
		if bytes.Contains(content, []byte("paper-b")) {
			title = "Paper B"
		}
		return bytes.Replace(sverka.MockExtraction,
			[]byte("Сверка — синтетическая оценка поиска"), []byte(title), 1), nil
	})
	h := newHarness(t, source, time.Second)
	h.startWorker(t)
	first := decode[model.Document](t, upload(t, h, "%PDF-1.7\npaper-a", http.StatusAccepted))
	second := decode[model.Document](t, upload(t, h, "%PDF-1.7\npaper-b", http.StatusAccepted))
	waitFor(t, h, first.ID, model.Completed)
	waitFor(t, h, second.ID, model.Completed)
	response := decode[struct {
		Groups []model.ComparableGroup `json:"groups"`
	}](t, request(t, http.MethodGet, h.url+"/api/v1/groups", "", nil, http.StatusOK))
	if len(response.Groups) != 1 || len(response.Groups[0].Results) != 4 {
		t.Fatalf("two publications with two matching configurations expected one four-result group, got: %+v", response.Groups)
	}
	group := response.Groups[0]
	equal(t, "group extraction mode", "mock", group.ExtractionMode)
	if group.Results[0].DocumentID == group.Results[1].DocumentID {
		t.Fatal("expected the first two group results to be from distinct publications")
	}
	configurations := map[string]bool{}
	for _, result := range group.Results {
		configurations[result.ConfigurationName] = true
		if result.Normalized == nil {
			t.Fatalf("expected a normalized value for group result %s", result.ResultID)
		}
	}
	if !configurations["Dense"] || !configurations["Hybrid"] {
		t.Fatalf("group configuration names expected Dense and Hybrid, got: %+v", configurations)
	}
}
