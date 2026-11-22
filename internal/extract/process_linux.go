package extract

import (
	"context"
	"os/exec"
	"syscall"
)

func runExtractorProcess(ctx context.Context, cmd *exec.Cmd) error {
	cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
	cmd.Cancel = nil
	if err := cmd.Start(); err != nil {
		return err
	}
	killGroup := func() { _ = syscall.Kill(-cmd.Process.Pid, syscall.SIGKILL) }
	// os/exec stops watching cancellation when the leader exits, even while
	// Wait is still draining pipes that Poppler or Tesseract may have inherited.
	killed := make(chan struct{})
	stop := context.AfterFunc(ctx, func() {
		killGroup()
		close(killed)
	})
	err := cmd.Wait()
	if !stop() {
		<-killed
	} else if err != nil || ctx.Err() != nil {
		// A nonzero exit status can mask ErrWaitDelay when the pipes time out.
		killGroup()
	}
	return err
}
