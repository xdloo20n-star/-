package api

import (
	"encoding/json"
	"fmt"
	"io"
	"math"
	"net/http"
	"net/url"
	"regexp"
	"strconv"

	"sverka/internal/model"
	"sverka/internal/store"
)

var idPattern = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9_-]*$`)

func pagination(query url.Values) (store.Pagination, error) {
	page := store.Pagination{Page: 1, Size: 25}
	for _, field := range []struct {
		name string
		dst  *int
	}{
		{"page", &page.Page},
		{"page_size", &page.Size},
	} {
		if !query.Has(field.name) {
			continue
		}
		value, err := strconv.Atoi(query.Get(field.name))
		if err != nil || value < 1 {
			return page, fmt.Errorf("%s должен быть положительным целым числом", field.name)
		}
		*field.dst = value
	}
	if page.Size > 100 || page.Page-1 > math.MaxInt/page.Size {
		return page, fmt.Errorf("параметры пагинации выходят за допустимый диапазон")
	}
	return page, nil
}

func queryID(query url.Values, name string) (string, error) {
	value := query.Get(name)
	if query.Has(name) && !idPattern.MatchString(value) {
		return "", fmt.Errorf("некорректный идентификатор %s", name)
	}
	return value, nil
}

func resultFilter(query url.Values) (store.ResultFilter, error) {
	filter := store.ResultFilter{Metric: query.Get("metric_name"), Query: query.Get("q")}
	for _, field := range []struct {
		name string
		dst  *string
	}{
		{"document_id", &filter.DocumentID}, {"dataset_id", &filter.DatasetID},
		{"task_id", &filter.TaskID}, {"config_id", &filter.ConfigID},
	} {
		value, err := queryID(query, field.name)
		if err != nil {
			return filter, err
		}
		*field.dst = value
	}
	if (query.Has("metric_name") && filter.Metric == "") || (query.Has("q") && filter.Query == "") {
		return filter, fmt.Errorf("параметры metric_name и q должны содержать текст")
	}
	if query.Has("has_issues") {
		raw := query.Get("has_issues")
		if raw != "true" && raw != "false" {
			return filter, fmt.Errorf("has_issues должен быть true или false")
		}
		value := raw == "true"
		filter.HasIssues = &value
	}
	return filter, nil
}

func (s *Server) listResults(w http.ResponseWriter, r *http.Request) {
	query, ok := requestQuery(w, r)
	if !ok {
		return
	}
	page, err := pagination(query)
	if err != nil {
		badQuery(w, err)
		return
	}
	filter, err := resultFilter(query)
	if err != nil {
		badQuery(w, err)
		return
	}
	result, err := s.store.ListResults(r.Context(), page, filter)
	if err != nil {
		s.storeError(w, r, err)
		return
	}
	writeJSON(w, http.StatusOK, result)
}

func (s *Server) result(w http.ResponseWriter, r *http.Request) {
	row, err := s.store.ResultDetail(r.Context(), r.PathValue("id"))
	if err != nil {
		s.storeError(w, r, err)
		return
	}
	writeJSON(w, http.StatusOK, row)
}

func (s *Server) comparison(w http.ResponseWriter, r *http.Request) {
	var request struct {
		LeftID  string `json:"left_result_id"`
		RightID string `json:"right_result_id"`
	}
	decoder := json.NewDecoder(http.MaxBytesReader(w, r.Body, 16<<10))
	decoder.DisallowUnknownFields()
	err := decoder.Decode(&request)
	var extra any
	if err != nil || decoder.Decode(&extra) != io.EOF ||
		!idPattern.MatchString(request.LeftID) || !idPattern.MatchString(request.RightID) || request.LeftID == request.RightID {
		problem(w, http.StatusBadRequest, "invalid_comparison", "Укажите два разных ID результатов.")
		return
	}
	left, err := s.store.ResultDetail(r.Context(), request.LeftID)
	if err != nil {
		s.storeError(w, r, err)
		return
	}
	right, err := s.store.ResultDetail(r.Context(), request.RightID)
	if err != nil {
		s.storeError(w, r, err)
		return
	}
	writeJSON(w, http.StatusOK, model.Compare(left, right))
}

func (s *Server) export(w http.ResponseWriter, r *http.Request) {
	query, ok := requestQuery(w, r)
	if !ok {
		return
	}
	id, err := queryID(query, "document_id")
	if err != nil {
		badQuery(w, err)
		return
	}
	started := false
	begin := func() error {
		started = true
		w.Header().Set("Content-Type", "application/json; charset=utf-8")
		_, err := io.WriteString(w, `{"schema_version":`+strconv.Quote(model.SchemaVersion)+`,"documents":[`)
		return err
	}
	err = s.store.ReadExport(r.Context(), id, func(doc model.Document, snapshot []byte) error {
		if snapshot == nil {
			snapshot = []byte("null")
		} else if !json.Valid(snapshot) {
			return fmt.Errorf("invalid stored snapshot for %s", doc.ID)
		}
		metadata, err := json.Marshal(doc)
		if err != nil {
			return err
		}
		if !started {
			if err := begin(); err != nil {
				return err
			}
		} else if _, err := io.WriteString(w, ","); err != nil {
			return err
		}
		return writeExportDocument(w, metadata, snapshot)
	})
	if err != nil {
		if !started {
			s.storeError(w, r, err)
			return
		}
		s.log.Error("stream export", "error", err)
		panic(http.ErrAbortHandler)
	}
	if !started {
		if err := begin(); err != nil {
			return
		}
	}
	_, _ = io.WriteString(w, "]}\n")
}

func writeExportDocument(w io.Writer, metadata, snapshot []byte) error {
	for _, chunk := range [][]byte{[]byte(`{"document":`), metadata, []byte(`,"data":`), snapshot, []byte("}")} {
		if _, err := w.Write(chunk); err != nil {
			return err
		}
	}
	return nil
}

func requestQuery(w http.ResponseWriter, r *http.Request) (url.Values, bool) {
	query, err := url.ParseQuery(r.URL.RawQuery)
	if err != nil {
		badQuery(w, fmt.Errorf("некорректные параметры запроса: %w", err))
		return nil, false
	}
	return query, true
}

func badQuery(w http.ResponseWriter, err error) {
	problem(w, http.StatusBadRequest, "invalid_query", err.Error())
}
