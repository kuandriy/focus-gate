//go:build windows

package persist

// Lock is a no-op on Windows. The stdlib does not expose a portable advisory
// lock primitive, and Focus Gate intentionally avoids external dependencies.
// Concurrent hook invocations on Windows are rare in practice; if they become
// a real problem, add a LockFileEx-based implementation here guarded by the
// windows build tag.
type Lock struct{}

// Acquire returns a zero-value Lock. It never blocks and never fails.
func Acquire(path string) (*Lock, error) {
	_ = path
	return &Lock{}, nil
}

// Release is a no-op.
func (l *Lock) Release() error { return nil }
