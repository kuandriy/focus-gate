package persist

import (
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
)

// SaveAtomic marshals v to JSON and writes it atomically (tmp file + rename).
// This prevents corruption if the process is killed mid-write.
func SaveAtomic(path string, v interface{}) error {
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

	return os.Rename(tmp, path)
}

// Load reads a JSON file and unmarshals it into v.
// If the file does not exist, v is left unchanged and no error is returned.
func Load(path string, v interface{}) error {
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
