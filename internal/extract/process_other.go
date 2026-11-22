//go:build !linux

package extract

import (
	"context"
	"os/exec"
)

func runExtractorProcess(_ context.Context, cmd *exec.Cmd) error { return cmd.Run() }
