package api

import (
	"context"
	"database/sql"
	"errors"
	"time"

	"modernc.org/sqlite"
	sqlite3 "modernc.org/sqlite/lib"

	"sverka/internal/model"
)

// RunWorker claims jobs from SQLite. Call Recover before starting the worker.
func (s *Server) RunWorker(ctx context.Context) {
	s.runWorker(ctx, ctx)
}

// RunWorkerWithGrace stops claiming on cancellation and lets the active job
// finish for up to grace before interrupting it.
func (s *Server) RunWorkerWithGrace(ctx context.Context, grace time.Duration) {
	workCtx, cancelWork := context.WithCancel(context.WithoutCancel(ctx))
	defer cancelWork()
	stopTimer := context.AfterFunc(ctx, func() {
		timer := time.NewTimer(grace)
		defer timer.Stop()
		select {
		case <-timer.C:
			cancelWork()
		case <-workCtx.Done():
		}
	})
	defer stopTimer()
	s.runWorker(ctx, workCtx)
}

func (s *Server) runWorker(claimCtx, workCtx context.Context) {
	ticker := time.NewTicker(time.Second)
	defer ticker.Stop()
	for claimCtx.Err() == nil {
		doc, err := s.store.Claim(claimCtx)
		if err == nil {
			s.process(workCtx, doc)
			continue
		}
		if !errors.Is(err, sql.ErrNoRows) && claimCtx.Err() == nil {
			s.log.Error("claim extraction job", "error", err)
		}
		select {
		case <-claimCtx.Done():
		case <-ticker.C:
		case <-s.wake:
		}
	}
}

func (s *Server) process(ctx context.Context, doc model.Document) {
	extractor := s.extractors[doc.ExtractionMode]
	if extractor == nil {
		s.failJob(ctx, doc.ID, "extractor_unavailable", "Извлекатель для этого режима не настроен.", nil)
		return
	}
	extractionCtx, cancel := context.WithTimeout(ctx, s.timeout)
	raw, err := extractor.Extract(extractionCtx, s.pdfPath(doc.ID))
	extractionErr := extractionCtx.Err()
	cancel()
	switch {
	case ctx.Err() != nil:
		s.failJob(ctx, doc.ID, "interrupted", "Обработка прервана при остановке сервера.", ctx.Err())
		return
	case errors.Is(extractionErr, context.DeadlineExceeded) || errors.Is(err, context.DeadlineExceeded):
		s.failJob(ctx, doc.ID, "extraction_timeout", "Извлечение превысило лимит времени.", err)
		return
	case err != nil:
		s.failJob(ctx, doc.ID, "extraction_failed", "Извлекатель завершился с ошибкой.", err)
		return
	}

	data, err := s.importer.Prepare(raw, doc)
	if err != nil {
		s.failJob(ctx, doc.ID, "invalid_extraction", "Ответ извлекателя не соответствует контракту.", err)
		return
	}
	err = retryTerminalWrite(ctx, func(ctx context.Context) error {
		return s.store.Complete(ctx, doc.ID, data)
	})
	if err != nil {
		if ctx.Err() != nil {
			s.failJob(ctx, doc.ID, "interrupted", "Обработка прервана при остановке сервера.", ctx.Err())
		} else {
			s.failJob(ctx, doc.ID, "import_failed", "Не удалось сохранить результаты.", err)
		}
		return
	}
	s.log.Info("extraction completed", "document_id", doc.ID, "mode", doc.ExtractionMode, "results", len(data.Results))
}

func (s *Server) failJob(ctx context.Context, id, code, message string, cause error) {
	s.log.Warn("extraction failed", "document_id", id, "code", code, "error", cause)
	write := func(ctx context.Context) error {
		return s.store.Fail(ctx, id, model.Problem{Code: code, Message: message})
	}
	err := retryTerminalWrite(ctx, write)
	if err != nil && ctx.Err() != nil {
		// An interrupted job still needs a terminal status after work is canceled.
		persistCtx, cancel := context.WithTimeout(context.WithoutCancel(ctx), 5*time.Second)
		defer cancel()
		err = retryTerminalWrite(persistCtx, write)
	}
	if err != nil {
		s.log.Error("persist job failure", "document_id", id, "error", err)
	}
}

func retryTerminalWrite(ctx context.Context, write func(context.Context) error) error {
	for ctx.Err() == nil {
		err := write(ctx)
		var sqliteErr *sqlite.Error
		if !errors.As(err, &sqliteErr) {
			return err
		}
		// Extended SQLite result codes retain the primary code in the low byte.
		switch sqliteErr.Code() & 0xff {
		case sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED:
		default:
			return err
		}
		timer := time.NewTimer(100 * time.Millisecond)
		select {
		case <-ctx.Done():
			timer.Stop()
			return ctx.Err()
		case <-timer.C:
		}
	}
	return ctx.Err()
}
