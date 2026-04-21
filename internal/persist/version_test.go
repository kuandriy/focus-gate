package persist

import (
	"errors"
	"os"
	"path/filepath"
	"testing"
)

type versionedData struct {
	SchemaVersion string `json:"schemaVersion"`
	Name          string `json:"name"`
	Value         int    `json:"value"`
}

func (v *versionedData) SetSchemaVersion(s string) { v.SchemaVersion = s }

func TestLoadVersionedMissingFile(t *testing.T) {
	var data versionedData
	err := LoadVersioned("/nonexistent/path/file.json", &data, "1")
	if err != nil {
		t.Errorf("missing file should not error, got: %v", err)
	}
	if data.SchemaVersion != "" {
		t.Error("schema version should not be stamped on missing file")
	}
}

func TestLoadVersionedMatch(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "t.json")

	original := versionedData{SchemaVersion: "1", Name: "focus", Value: 42}
	if err := SaveAtomic(path, original); err != nil {
		t.Fatal(err)
	}

	var loaded versionedData
	if err := LoadVersioned(path, &loaded, "1"); err != nil {
		t.Fatalf("LoadVersioned failed: %v", err)
	}
	if loaded.Name != original.Name || loaded.Value != original.Value {
		t.Errorf("loaded = %+v, want %+v", loaded, original)
	}
}

func TestLoadVersionedLegacyFileGetsStamped(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "legacy.json")

	// Write legacy JSON without a schemaVersion field.
	if err := os.WriteFile(path, []byte(`{"name":"legacy","value":7}`), 0644); err != nil {
		t.Fatal(err)
	}

	var loaded versionedData
	if err := LoadVersioned(path, &loaded, "1"); err != nil {
		t.Fatalf("legacy file should load without error, got: %v", err)
	}
	if loaded.Name != "legacy" || loaded.Value != 7 {
		t.Errorf("unmarshal failed: %+v", loaded)
	}
	if loaded.SchemaVersion != "1" {
		t.Errorf("SetSchemaVersion should stamp current version on legacy data, got %q", loaded.SchemaVersion)
	}
}

func TestLoadVersionedMismatchRejected(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "future.json")

	// File declares a newer schema than the binary supports.
	if err := os.WriteFile(path, []byte(`{"schemaVersion":"2","name":"future","value":99}`), 0644); err != nil {
		t.Fatal(err)
	}

	var loaded versionedData
	err := LoadVersioned(path, &loaded, "1")
	if err == nil {
		t.Fatal("expected ErrSchemaMismatch, got nil")
	}
	if !errors.Is(err, ErrSchemaMismatch) {
		t.Errorf("expected ErrSchemaMismatch, got %v", err)
	}
	if loaded.Name != "" || loaded.Value != 0 {
		t.Errorf("data should not be populated on version mismatch, got %+v", loaded)
	}
}
