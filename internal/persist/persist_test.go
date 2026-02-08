package persist

import (
	"os"
	"path/filepath"
	"testing"
)

type testData struct {
	Name  string `json:"name"`
	Value int    `json:"value"`
}

func TestSaveAndLoad(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "test.json")

	original := testData{Name: "focus", Value: 42}
	if err := SaveAtomic(path, original); err != nil {
		t.Fatalf("SaveAtomic failed: %v", err)
	}

	var loaded testData
	if err := Load(path, &loaded); err != nil {
		t.Fatalf("Load failed: %v", err)
	}

	if loaded.Name != original.Name || loaded.Value != original.Value {
		t.Errorf("loaded = %+v, want %+v", loaded, original)
	}
}

func TestSaveCreatesDirectory(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "nested", "deep", "test.json")

	if err := SaveAtomic(path, testData{Name: "test"}); err != nil {
		t.Fatalf("SaveAtomic with nested dirs failed: %v", err)
	}

	if !Exists(path) {
		t.Error("file should exist after save")
	}
}

func TestLoadMissingFile(t *testing.T) {
	var data testData
	err := Load("/nonexistent/path/file.json", &data)
	if err != nil {
		t.Errorf("Load of missing file should not error, got: %v", err)
	}
	if data.Name != "" {
		t.Error("data should be zero value when file is missing")
	}
}

func TestSaveAtomicNoPartialWrite(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "test.json")

	// Save initial data
	if err := SaveAtomic(path, testData{Name: "original", Value: 1}); err != nil {
		t.Fatal(err)
	}

	// Save updated data
	if err := SaveAtomic(path, testData{Name: "updated", Value: 2}); err != nil {
		t.Fatal(err)
	}

	// No .tmp file should remain
	tmp := path + ".tmp"
	if Exists(tmp) {
		t.Error(".tmp file should not exist after successful save")
	}

	// Data should be the updated version
	var loaded testData
	if err := Load(path, &loaded); err != nil {
		t.Fatal(err)
	}
	if loaded.Name != "updated" || loaded.Value != 2 {
		t.Errorf("loaded = %+v, want {updated, 2}", loaded)
	}
}

func TestRemove(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "test.json")

	os.WriteFile(path, []byte("{}"), 0644)
	if !Exists(path) {
		t.Fatal("file should exist before removal")
	}

	if err := Remove(path); err != nil {
		t.Fatalf("Remove failed: %v", err)
	}
	if Exists(path) {
		t.Error("file should not exist after removal")
	}

	// Removing nonexistent file should not error
	if err := Remove(path); err != nil {
		t.Errorf("Remove of nonexistent file should not error: %v", err)
	}
}
