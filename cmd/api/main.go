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
	"syscall"
	"time"

	"sverka"
	"sverka/internal/api"
	"sverka/internal/extract"
	"sverka/internal/store"
)

func main() {
	logger := slog.New(slog.NewTextHandler(os.Stderr, nil))
	if err := run(logger); err != nil {
		logger.Error("Сверка остановлена", "error", err)
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
	dbPath, err := databasePath(cfg.dataDir)
	if err != nil {
		return err
	}
	db, err := store.Open(ctx, dbPath)
	if err != nil {
		return err
	}
	defer db.Close()
	if err := db.Recover(ctx); err != nil {
		return fmt.Errorf("recover interrupted jobs: %w", err)
	}
	importer, err := extract.NewImporter(sverka.ExtractorSchema)
	if err != nil {
		return fmt.Errorf("prepare extractor schema: %w", err)
	}
	extractors := []extract.Extractor{extract.MockExtractor{Fixture: sverka.MockExtraction}}
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
	logger.Info("Сверка запущена", "address", listener.Addr(), "extraction_mode", cfg.api.ExtractionMode)

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
