package extract

import (
	"context"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"syscall"
	"testing"
	"time"
)

func TestPythonExtractorKillsProcessGroup(t *testing.T) {
	python, err := exec.LookPath("python3")
	if err != nil {
		t.Skip("python3 unavailable")
	}
	for _, test := range []struct {
		name       string
		leader     string
		deadline   time.Duration
		contextErr error
	}{
		{"deadline after leader exit", "exit", 1500 * time.Millisecond, context.DeadlineExceeded},
		{"cancellation after leader exit", "exit", 0, context.Canceled},
		{"cancellation with leader running", "wait", 0, context.Canceled},
		{"pipe wait limit after leader error", "error", 8 * time.Second, nil},
	} {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()
			dir := t.TempDir()
			script := filepath.Join(dir, "stub.py")
			body := fmt.Sprintf(`
import os
import subprocess
import sys
import time
from pathlib import Path

behavior = %q
child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(12)"])
Path(sys.argv[2]).write_text(f"{os.getpid()} {child.pid}")
if behavior == "wait":
    time.sleep(12)
if behavior == "error":
    sys.exit(7)
`, test.leader)
			if err := os.WriteFile(script, []byte(body), 0600); err != nil {
				t.Fatal(err)
			}
			ctx, cancel := context.WithCancel(t.Context())
			defer cancel()
			if test.deadline != 0 {
				var cancelDeadline context.CancelFunc
				ctx, cancelDeadline = context.WithTimeout(ctx, test.deadline)
				defer cancelDeadline()
			}
			pids := filepath.Join(dir, "pids")
			done := make(chan error, 1)
			go func() {
				_, err := (PythonExtractor{Executable: python, Script: script}).Extract(ctx, pids)
				done <- err
			}()
			var leader, child int
			waitForExtractorCondition(t, "stub process IDs", func() bool {
				raw, err := os.ReadFile(pids)
				if err != nil {
					return false
				}
				_, err = fmt.Sscan(string(raw), &leader, &child)
				return err == nil && leader > 0 && child > 0
			})
			t.Cleanup(func() {
				if !extractorProcessExited(leader) || !extractorProcessExited(child) {
					_ = syscall.Kill(-leader, syscall.SIGKILL)
				}
			})
			if test.leader != "wait" {
				waitForExtractorCondition(t, "leader exit", func() bool { return extractorProcessExited(leader) })
				if extractorProcessExited(child) {
					t.Fatal("stub child exited before the pipe wait")
				}
			}
			if test.contextErr == context.Canceled {
				cancel()
			}

			select {
			case err := <-done:
				if test.contextErr != nil {
					if !errors.Is(err, test.contextErr) {
						t.Fatalf("extraction error = %v, want %v", err, test.contextErr)
					}
				} else {
					var exitErr *exec.ExitError
					if !errors.As(err, &exitErr) || exitErr.ExitCode() != 7 {
						t.Fatalf("extraction error = %v, want exit status 7", err)
					}
				}
			case <-time.After(5 * time.Second):
				t.Fatal("extractor did not return")
			}
			waitForExtractorCondition(t, "child exit", func() bool { return extractorProcessExited(child) })
		})
	}
}

func extractorProcessExited(pid int) bool {
	raw, err := os.ReadFile(fmt.Sprintf("/proc/%d/stat", pid))
	if err != nil {
		return errors.Is(err, os.ErrNotExist)
	}
	// A killed child may remain a zombie until the process that adopted it reaps it.
	_, state, _ := strings.Cut(string(raw), ") ")
	return strings.HasPrefix(state, "Z ") || strings.HasPrefix(state, "X ")
}

func waitForExtractorCondition(t *testing.T, description string, condition func() bool) {
	t.Helper()
	deadline := time.Now().Add(2 * time.Second)
	for !condition() {
		if time.Now().After(deadline) {
			t.Fatalf("timed out waiting for %s", description)
		}
		time.Sleep(10 * time.Millisecond)
	}
}
