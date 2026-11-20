package api

import (
	"context"
	"database/sql"
	"errors"
	"time"

	"papertrace/internal/model"
)

// RunWorker claims jobs from SQLite. Call Recover before starting the worker.
func (s *Server) RunWorker(ctx context.Context) {
	ticker := time.NewTicker(time.Second)
	defer ticker.Stop()
	for ctx.Err() == nil {
		doc, err := s.store.Claim(ctx)
		if err == nil {
			s.process(ctx, doc)
			continue
		}
		if !errors.Is(err, sql.ErrNoRows) && ctx.Err() == nil {
			s.log.Error("claim extraction job", "error", err)
		}
		select {
		case <-ctx.Done():
		case <-ticker.C:
		case <-s.wake:
		}
	}
}

func (s *Server) process(ctx context.Context, doc model.Document) {
	extractor := s.extractors[doc.ExtractionMode]
	if extractor == nil {
		s.failJob(doc.ID, "extractor_unavailable", "Извлекатель для этого режима не настроен.", nil)
		return
	}
	extractionCtx, cancel := context.WithTimeout(ctx, s.timeout)
	raw, err := extractor.Extract(extractionCtx, s.pdfPath(doc.ID))
	extractionErr := extractionCtx.Err()
	cancel()
	switch {
	case ctx.Err() != nil:
		s.failJob(doc.ID, "interrupted", "Обработка прервана при остановке сервера.", ctx.Err())
		return
	case errors.Is(extractionErr, context.DeadlineExceeded) || errors.Is(err, context.DeadlineExceeded):
		s.failJob(doc.ID, "extraction_timeout", "Извлечение превысило лимит времени.", err)
		return
	case err != nil:
		s.failJob(doc.ID, "extraction_failed", "Извлекатель завершился с ошибкой.", err)
		return
	}

	data, err := s.importer.Prepare(raw, doc)
	if err != nil {
		s.failJob(doc.ID, "invalid_extraction", "Ответ извлекателя не соответствует контракту.", err)
		return
	}
	if err := s.store.Complete(ctx, doc.ID, data); err != nil {
		if ctx.Err() != nil {
			s.failJob(doc.ID, "interrupted", "Обработка прервана при остановке сервера.", ctx.Err())
		} else {
			s.failJob(doc.ID, "import_failed", "Не удалось сохранить результаты.", err)
		}
		return
	}
	s.log.Info("extraction completed", "document_id", doc.ID, "mode", doc.ExtractionMode, "results", len(data.Results))
}

func (s *Server) failJob(id, code, message string, cause error) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	s.log.Warn("extraction failed", "document_id", id, "code", code, "error", cause)
	if err := s.store.Fail(ctx, id, model.Problem{Code: code, Message: message}); err != nil {
		s.log.Error("persist job failure", "document_id", id, "error", err)
	}
}
