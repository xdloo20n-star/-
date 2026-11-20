package extract

import (
	"errors"
	"os"
	"os/exec"
	"syscall"
)

// Stop Poppler and Tesseract descendants together with the Python orchestrator.
func configureExtractorProcess(cmd *exec.Cmd) {
	cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
	cmd.Cancel = func() error {
		err := syscall.Kill(-cmd.Process.Pid, syscall.SIGKILL)
		if errors.Is(err, syscall.ESRCH) {
			return os.ErrProcessDone
		}
		return err
	}
}
