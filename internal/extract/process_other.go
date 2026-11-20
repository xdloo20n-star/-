//go:build !linux

package extract

import "os/exec"

func configureExtractorProcess(_ *exec.Cmd) {}
