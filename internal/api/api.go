package api

import (
	"database/sql"
	"encoding/json"
	"errors"
	"log/slog"
	"net/http"
	"path/filepath"
	"slices"
	"strings"
	"sync"
	"time"

	"sverka/internal/extract"
	"sverka/internal/model"
	"sverka/internal/store"
)

type Config struct {
	PDFDir            string
	ExtractionMode    string
	MaxPDFBytes       int64
	ExtractionTimeout time.Duration
	CORSOrigin        string
}

type Server struct {
	store       *store.Store
	importer    *extract.Importer
	extractors  map[string]extract.Extractor
	log         *slog.Logger
	pdfDir      string
	mode        string
	maxPDFBytes int64
	timeout     time.Duration
	corsOrigin  string
	wake        chan struct{}
	// Serialize corpus uploads from lookup through enqueue, including ordinary POSTs.
	corpusUploads sync.Mutex
}

func New(config Config, db *store.Store, importer *extract.Importer, logger *slog.Logger, extractors ...extract.Extractor) *Server {
	byMode := make(map[string]extract.Extractor, len(extractors))
	for _, extractor := range extractors {
		byMode[extractor.Mode()] = extractor
	}
	return &Server{
		store: db, importer: importer, extractors: byMode, log: logger,
		pdfDir: config.PDFDir, maxPDFBytes: config.MaxPDFBytes,
		mode: config.ExtractionMode, timeout: config.ExtractionTimeout,
		corsOrigin: config.CORSOrigin, wake: make(chan struct{}, 1),
	}
}

func (s *Server) Handler() http.Handler {
	mux := http.NewServeMux()
	get := func(path string, handler http.HandlerFunc) {
		route(mux, path, map[string]http.HandlerFunc{http.MethodGet: handler})
	}
	get("/healthz", s.health)
	route(mux, "/api/v1/documents", map[string]http.HandlerFunc{
		http.MethodGet: s.listDocuments, http.MethodPost: s.uploadDocument,
	})
	get("/api/v1/documents/{id}", s.document)
	get("/api/v1/documents/{id}/file", s.documentFile)
	get("/api/v1/documents/{id}/digest", s.documentDigest)
	get("/api/v1/results", s.listResults)
	get("/api/v1/results/{id}", s.result)
	get("/api/v1/groups", s.comparableGroups)
	route(mux, "/api/v1/comparisons", map[string]http.HandlerFunc{http.MethodPost: s.comparison})
	get("/api/v1/export", s.export)
	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		problem(w, http.StatusNotFound, "not_found", "Маршрут не найден.")
	})
	return s.cors(mux)
}

func route(mux *http.ServeMux, path string, handlers map[string]http.HandlerFunc) {
	var allowed []string
	for method := range handlers {
		allowed = append(allowed, method)
	}
	if handlers[http.MethodGet] != nil {
		allowed = append(allowed, http.MethodHead)
	}
	slices.Sort(allowed)
	allow := strings.Join(allowed, ", ")
	mux.HandleFunc(path, func(w http.ResponseWriter, r *http.Request) {
		method := r.Method
		if method == http.MethodHead {
			method = http.MethodGet
		}
		if handler := handlers[method]; handler != nil {
			handler(w, r)
			return
		}
		w.Header().Set("Allow", allow)
		problem(w, http.StatusMethodNotAllowed, "method_not_allowed", "Метод не поддерживается для этого маршрута.")
	})
}

func (s *Server) cors(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if s.corsOrigin != "" {
			w.Header().Add("Vary", "Origin")
		}
		origin := r.Header.Get("Origin")
		allowed := origin != "" && origin == s.corsOrigin
		if allowed {
			w.Header().Set("Access-Control-Allow-Origin", origin)
			w.Header().Set("Access-Control-Expose-Headers", "Accept-Ranges, Content-Disposition, Content-Range")
		}
		if r.Method == http.MethodOptions && origin != "" && r.Header.Get("Access-Control-Request-Method") != "" {
			if !allowed {
				problem(w, http.StatusForbidden, "origin_not_allowed", "Этот источник запроса не разрешён.")
				return
			}
			w.Header().Set("Access-Control-Allow-Methods", "GET, HEAD, POST, OPTIONS")
			w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Range")
			w.WriteHeader(http.StatusNoContent)
			return
		}
		next.ServeHTTP(w, r)
	})
}

func (s *Server) health(w http.ResponseWriter, r *http.Request) {
	if err := s.store.Ping(r.Context()); err != nil {
		s.log.Error("database health check", "error", err)
		problem(w, http.StatusServiceUnavailable, "unavailable", "База данных недоступна.")
		return
	}
	writeJSON(w, http.StatusOK, map[string]string{"status": "ok"})
}

func (s *Server) pdfPath(id string) string {
	return filepath.Join(s.pdfDir, id+".pdf")
}

func (s *Server) storeError(w http.ResponseWriter, r *http.Request, err error) {
	if errors.Is(err, sql.ErrNoRows) {
		problem(w, http.StatusNotFound, "not_found", "Запись не найдена.")
		return
	}
	s.log.Error("request failed", "method", r.Method, "path", r.URL.Path, "error", err)
	problem(w, http.StatusInternalServerError, "internal_error", "Не удалось выполнить запрос.")
}

func problem(w http.ResponseWriter, status int, code, message string) {
	writeJSON(w, status, struct {
		Error model.Problem `json:"error"`
	}{Error: model.Problem{Code: code, Message: message}})
}

func writeJSON(w http.ResponseWriter, status int, value any) {
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(value)
}
