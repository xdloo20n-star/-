package main

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"net"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"syscall"
	"time"

	"papertrace"
	"papertrace/internal/api"
	"papertrace/internal/extract"
	"papertrace/internal/store"
)

func main() {
	logger := slog.New(slog.NewTextHandler(os.Stderr, nil))
	if err := run(logger); err != nil {
		logger.Error("PaperTrace stopped", "error", err)
		os.Exit(1)
	}
}

func run(logger *slog.Logger) error {
	cfg, err := settingsFromEnvironment()
	if err != nil {
		return err
	}
	if err := os.MkdirAll(cfg.api.PDFDir, 0700); err != nil {
		return fmt.Errorf("create PDF directory: %w", err)
	}
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	listener, err := net.Listen("tcp", cfg.address)
	if err != nil {
		return fmt.Errorf("listen: %w", err)
	}
	defer listener.Close()
	db, err := store.Open(ctx, filepath.Join(cfg.dataDir, "papertrace.sqlite"))
	if err != nil {
		return err
	}
	defer db.Close()
	if err := db.Recover(ctx); err != nil {
		return fmt.Errorf("recover interrupted jobs: %w", err)
	}
	importer, err := extract.NewImporter(papertrace.ExtractorSchema)
	if err != nil {
		return fmt.Errorf("prepare extractor schema: %w", err)
	}
	extractors := []extract.Extractor{extract.MockExtractor{Fixture: papertrace.MockExtraction}}
	if cfg.python.Script != "" {
		extractors = append(extractors, cfg.python)
	}
	app := api.New(cfg.api, db, importer, logger, extractors...)
	server := &http.Server{
		Handler:           app.Handler(),
		ReadHeaderTimeout: 5 * time.Second,
		ReadTimeout:       60 * time.Second,
		WriteTimeout:      60 * time.Second,
		IdleTimeout:       60 * time.Second,
	}

	workerCtx, cancelWorker := context.WithCancel(context.Background())
	workerDone := make(chan struct{})
	go func() {
		defer close(workerDone)
		app.RunWorker(workerCtx)
	}()
	serveError := make(chan error, 1)
	go func() { serveError <- server.Serve(listener) }()
	logger.Info("PaperTrace listening", "address", listener.Addr(), "extraction_mode", cfg.api.ExtractionMode)

	select {
	case <-ctx.Done():
	case err = <-serveError:
		if errors.Is(err, http.ErrServerClosed) {
			err = nil
		}
	}
	shutdownCtx, cancelShutdown := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancelShutdown()
	if shutdownErr := server.Shutdown(shutdownCtx); shutdownErr != nil {
		_ = server.Close()
		err = errors.Join(err, shutdownErr)
	}
	cancelWorker()
	<-workerDone
	return err
}
