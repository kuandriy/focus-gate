//go:build !windows

package persist

import (
	"os"
	"path/filepath"
	"syscall"
)

// Lock is an advisory file lock acquired with syscall.Flock. The lock is
// released automatically when the underlying file descriptor is closed.
type Lock struct {
	f *os.File
}

// Acquire takes an exclusive advisory lock on path, creating the file if it
// does not exist. The call blocks until the lock is granted. The parent
// directory is created if necessary so the first-ever invocation of the hook
// succeeds without requiring external setup.
//
// Concurrent UserPromptSubmit hooks (e.g. two prompts arriving back-to-back)
// would otherwise race on state-file reads and writes, silently dropping the
// earlier prompt from the forest. Acquiring this lock at the start of the
// mutating path serializes them.
func Acquire(path string) (*Lock, error) {
	if err := os.MkdirAll(filepath.Dir(path), 0755); err != nil {
		return nil, err
	}
	f, err := os.OpenFile(path, os.O_CREATE|os.O_RDWR, 0644)
	if err != nil {
		return nil, err
	}
	if err := syscall.Flock(int(f.Fd()), syscall.LOCK_EX); err != nil {
		_ = f.Close()
		return nil, err
	}
	return &Lock{f: f}, nil
}

// Release drops the lock and closes the underlying file.
func (l *Lock) Release() error {
	if l == nil || l.f == nil {
		return nil
	}
	// Closing the fd releases the flock.
	err := l.f.Close()
	l.f = nil
	return err
}
