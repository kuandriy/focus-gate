package persist

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
)

// SaveAtomic writes v as indented JSON to a temporary file, then renames it
// to the target path. On Unix, os.Rename is atomic (POSIX guarantee). On
// Windows, Rename can fail if the target exists, so we remove it first. That
// creates a brief window where neither file exists; RecoverTmpFiles handles
// this on the next startup.
func SaveAtomic(path string, v any) error {
	dir := filepath.Dir(path)
	if err := os.MkdirAll(dir, 0755); err != nil {
		return err
	}

	data, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		return err
	}

	tmp := path + ".tmp"
	if err := os.WriteFile(tmp, data, 0644); err != nil {
		return err
	}

	// On Windows, os.Rename fails when the target already exists.
	if runtime.GOOS == "windows" {
		_ = os.Remove(path)
	}

	return os.Rename(tmp, path)
}

// RecoverTmpFiles restores data from stale .tmp files left by interrupted
// SaveAtomic calls. For each path: if .tmp exists but the target is missing,
// the .tmp is promoted; if both exist, the stale .tmp is removed. Should be
// called before any Load to ensure the most recent complete data is available.
func RecoverTmpFiles(paths ...string) {
	for _, path := range paths {
		tmp := path + ".tmp"
		tmpExists := Exists(tmp)
		targetExists := Exists(path)

		if !tmpExists {
			continue
		}

		if !targetExists {
			// .tmp without target — interrupted save. Promote .tmp to target.
			if err := os.Rename(tmp, path); err != nil {
				fmt.Fprintf(os.Stderr, "focus-gate: recover %s: %v\n", path, err)
			} else {
				fmt.Fprintf(os.Stderr, "focus-gate: recovered %s from tmp\n", path)
			}
		} else {
			// Both exist — target is authoritative, remove stale .tmp.
			if err := os.Remove(tmp); err != nil {
				fmt.Fprintf(os.Stderr, "focus-gate: cleanup %s.tmp: %v\n", path, err)
			}
		}
	}
}

// Load reads a JSON file and unmarshals it into v.
// If the file does not exist, v is left unchanged and no error is returned.
func Load(path string, v any) error {
	data, err := os.ReadFile(path)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return nil // Graceful: missing file = empty state
		}
		return err
	}
	return json.Unmarshal(data, v)
}

// Remove deletes a file. No error if the file doesn't exist.
func Remove(path string) error {
	err := os.Remove(path)
	if err != nil && !errors.Is(err, os.ErrNotExist) {
		return err
	}
	return nil
}

// Exists returns true if the file exists.
func Exists(path string) bool {
	_, err := os.Stat(path)
	return err == nil
}
