package main

import (
	"os"
	"path/filepath"
	"testing"
)

func TestDatabasePathUsesExistingFile(t *testing.T) {
	dir := t.TempDir()
	newPath := filepath.Join(dir, "sverka.sqlite")
	legacyPath := filepath.Join(dir, "papertrace.sqlite")

	check := func(expected string) {
		t.Helper()
		got, err := databasePath(dir)
		if err != nil {
			t.Fatal(err)
		}
		if got != expected {
			t.Fatalf("database path expected: %q, got: %q", expected, got)
		}
	}
	check(newPath)
	if err := os.WriteFile(legacyPath, nil, 0600); err != nil {
		t.Fatal(err)
	}
	check(legacyPath)
	if err := os.WriteFile(newPath, nil, 0600); err != nil {
		t.Fatal(err)
	}
	check(newPath)
}
