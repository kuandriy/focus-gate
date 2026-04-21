//go:build !windows

package persist

import (
	"path/filepath"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

func TestLockSerializesConcurrentAcquirers(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, ".lock")

	var active int32
	var maxActive int32
	var wg sync.WaitGroup

	for i := 0; i < 8; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			lock, err := Acquire(path)
			if err != nil {
				t.Errorf("Acquire failed: %v", err)
				return
			}
			defer lock.Release()

			now := atomic.AddInt32(&active, 1)
			for {
				old := atomic.LoadInt32(&maxActive)
				if now <= old || atomic.CompareAndSwapInt32(&maxActive, old, now) {
					break
				}
			}
			time.Sleep(20 * time.Millisecond)
			atomic.AddInt32(&active, -1)
		}()
	}

	wg.Wait()
	if maxActive > 1 {
		t.Errorf("lock did not serialize: observed %d concurrent holders", maxActive)
	}
}

func TestLockReleaseAllowsReacquire(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, ".lock")

	lock1, err := Acquire(path)
	if err != nil {
		t.Fatal(err)
	}
	if err := lock1.Release(); err != nil {
		t.Fatal(err)
	}

	done := make(chan error, 1)
	go func() {
		lock2, err := Acquire(path)
		if err != nil {
			done <- err
			return
		}
		_ = lock2.Release()
		done <- nil
	}()

	select {
	case err := <-done:
		if err != nil {
			t.Errorf("reacquire failed: %v", err)
		}
	case <-time.After(time.Second):
		t.Error("reacquire blocked after release")
	}
}
