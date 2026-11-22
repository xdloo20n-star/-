package main

import (
	"errors"
	"fmt"
	"math"
	"net"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"time"

	"sverka/internal/api"
	"sverka/internal/extract"
)

type settings struct {
	dataDir string
	address string
	api     api.Config
	python  extract.PythonExtractor
}

func settingsFromEnvironment() (settings, error) {
	var cfg settings
	port, err := strconv.Atoi(env("PORT", "8080"))
	if err != nil || port < 0 || port > 65535 {
		return cfg, fmt.Errorf("PORT must be between 0 and 65535")
	}
	cfg.address = net.JoinHostPort(env("HOST", "127.0.0.1"), strconv.Itoa(port))
	cfg.dataDir, err = filepath.Abs(env("DATA_DIR", "data"))
	if err != nil {
		return cfg, err
	}
	cfg.api.PDFDir = filepath.Join(cfg.dataDir, "pdf")
	cfg.api.CORSOrigin = env("CORS_ORIGIN", "http://localhost:5173")
	cfg.api.MaxPDFBytes, err = strconv.ParseInt(env("MAX_PDF_BYTES", "33554432"), 10, 64)
	if err != nil || cfg.api.MaxPDFBytes < 5 || cfg.api.MaxPDFBytes > math.MaxInt64-(64<<10) {
		return cfg, fmt.Errorf("MAX_PDF_BYTES must be a positive PDF size in bytes")
	}
	cfg.api.ExtractionTimeout, err = time.ParseDuration(env("EXTRACT_TIMEOUT", "5m"))
	if err != nil || cfg.api.ExtractionTimeout <= 0 {
		return cfg, fmt.Errorf("EXTRACT_TIMEOUT must be a positive duration such as 5m")
	}
	cfg.api.ExtractionMode = env("EXTRACTION_MODE", "mock")
	if cfg.api.ExtractionMode != "mock" && cfg.api.ExtractionMode != "python" {
		return cfg, fmt.Errorf("EXTRACTION_MODE must be mock or python")
	}
	cfg.python.Executable = env("PYTHON_BIN", "python3")
	if script := os.Getenv("PYTHON_SCRIPT"); script != "" {
		cfg.python.Script, err = filepath.Abs(script)
		if err != nil {
			return cfg, err
		}
		info, err := os.Stat(cfg.python.Script)
		if err != nil || !info.Mode().IsRegular() {
			return cfg, fmt.Errorf("PYTHON_SCRIPT must point to an existing regular file")
		}
		cfg.python.Executable, err = exec.LookPath(cfg.python.Executable)
		if err != nil {
			return cfg, fmt.Errorf("PYTHON_BIN must resolve to an executable: %w", err)
		}
	}
	if cfg.api.ExtractionMode == "python" && cfg.python.Script == "" {
		return cfg, fmt.Errorf("PYTHON_SCRIPT is required in python mode")
	}
	return cfg, nil
}

func (cfg settings) validateQueuedModes(modes []string) error {
	for _, mode := range modes {
		switch mode {
		case "mock":
		case "python":
			if cfg.python.Script == "" {
				return errors.New("PYTHON_SCRIPT is required for queued python jobs")
			}
		default:
			return fmt.Errorf("queued jobs require unsupported extraction mode %q", mode)
		}
	}
	return nil
}

func env(key, fallback string) string {
	if value, ok := os.LookupEnv(key); ok {
		return value
	}
	return fallback
}

func databasePath(dir string) (string, error) {
	current := filepath.Join(dir, "sverka.sqlite")
	// Use existing SQLite files in place, including their WAL sidecars.
	legacy := filepath.Join(dir, "papertrace.sqlite")
	for _, candidate := range []string{current, legacy} {
		_, err := os.Stat(candidate)
		if err == nil {
			return candidate, nil
		}
		if !errors.Is(err, os.ErrNotExist) {
			return "", fmt.Errorf("check database %q: %w", candidate, err)
		}
	}
	return current, nil
}
