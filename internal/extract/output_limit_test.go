package extract

import (
	"os"
	"os/exec"
	"path/filepath"
	"testing"
)

func TestPythonExtractorAcceptsOutputAbovePreviousLimit(t *testing.T) {
	python, err := exec.LookPath("python3")
	if err != nil {
		t.Skip("python3 is unavailable")
	}
	script := filepath.Join(t.TempDir(), "large.py")
	if err := os.WriteFile(script, []byte("import sys\nsys.stdout.buffer.write(b'x' * (33 * 1024 * 1024))\n"), 0600); err != nil {
		t.Fatal(err)
	}
	output, err := (PythonExtractor{Executable: python, Script: script}).Extract(t.Context(), "unused.pdf")
	if err != nil {
		t.Fatal(err)
	}
	if len(output) != 33<<20 {
		t.Fatalf("output length = %d, want %d", len(output), 33<<20)
	}
}
