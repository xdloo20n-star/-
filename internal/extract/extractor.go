package extract

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"os/exec"
	"strings"
	"time"
)

type Extractor interface {
	Mode() string
	Extract(ctx context.Context, pdfPath string) ([]byte, error)
}

type MockExtractor struct {
	Fixture []byte
}

func (MockExtractor) Mode() string { return "mock" }

func (m MockExtractor) Extract(ctx context.Context, _ string) ([]byte, error) {
	return m.Fixture, ctx.Err()
}

type PythonExtractor struct {
	Executable string
	Script     string
}

func (PythonExtractor) Mode() string { return "python" }

func (p PythonExtractor) Extract(ctx context.Context, pdfPath string) ([]byte, error) {
	const maxOutput = 32 << 20
	stdout := boundedOutput{limit: maxOutput}
	stderr := boundedOutput{limit: 4 << 10}
	cmd := exec.CommandContext(ctx, p.Executable, p.Script, "--pdf", pdfPath)
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	cmd.WaitDelay = 2 * time.Second
	err := cmd.Run()
	if ctx.Err() != nil {
		return nil, ctx.Err()
	}
	if stdout.truncated {
		return nil, fmt.Errorf("extractor stdout exceeds %d bytes", maxOutput)
	}
	if err != nil {
		diagnostic := strings.TrimSpace(stderr.buffer.String())
		if diagnostic == "" {
			return nil, fmt.Errorf("python extractor: %w", err)
		}
		return nil, fmt.Errorf("python extractor: %w: %s", err, diagnostic)
	}
	if stdout.buffer.Len() == 0 {
		return nil, errors.New("extractor stdout is empty")
	}
	return stdout.buffer.Bytes(), nil
}

type boundedOutput struct {
	buffer    bytes.Buffer
	limit     int
	truncated bool
}

func (output *boundedOutput) Write(p []byte) (int, error) {
	n := len(p)
	remaining := output.limit - output.buffer.Len()
	if n > remaining {
		output.truncated = true
		p = p[:remaining]
	}
	_, err := output.buffer.Write(p)
	return n, err
}

var _ io.Writer = (*boundedOutput)(nil)
