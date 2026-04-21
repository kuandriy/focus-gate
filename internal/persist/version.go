package persist

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
)

// SchemaVersioner is implemented by persistable types that carry an explicit
// schemaVersion field. Loaders use SetSchemaVersion to stamp the current
// version onto legacy files (no version on disk) after a successful load, so
// the next SaveAtomic writes the expected version.
type SchemaVersioner interface {
	SetSchemaVersion(string)
}

// ErrSchemaMismatch is returned by LoadVersioned when the file on disk
// declares a schemaVersion different from the one the caller expects. The
// caller should log and proceed with empty state rather than risk a partial
// unmarshal into an incompatible struct.
var ErrSchemaMismatch = errors.New("schema version mismatch")

// LoadVersioned reads a JSON file and unmarshals it into v, guarding against
// incompatible schema upgrades.
//
// Behaviour:
//   - Missing file: v is left unchanged, nil error (same as Load).
//   - File has no schemaVersion (legacy): unmarshal proceeds. If v implements
//     SchemaVersioner, its version is stamped to expected so the next save is
//     forward-compatible.
//   - File has schemaVersion matching expected: unmarshal proceeds.
//   - File has schemaVersion different from expected: returns ErrSchemaMismatch
//     wrapped with the observed/expected values. v is left unchanged.
func LoadVersioned(path string, v any, expected string) error {
	data, err := os.ReadFile(path)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return nil
		}
		return err
	}

	var peek struct {
		SchemaVersion string `json:"schemaVersion"`
	}
	// Ignore peek errors — if the JSON is malformed the main Unmarshal below
	// will surface the real problem.
	_ = json.Unmarshal(data, &peek)

	if peek.SchemaVersion != "" && peek.SchemaVersion != expected {
		return fmt.Errorf("%w: file=%q expected=%q", ErrSchemaMismatch, peek.SchemaVersion, expected)
	}

	if err := json.Unmarshal(data, v); err != nil {
		return err
	}

	// Stamp the current version so the next save is consistent even when
	// loading from a legacy (versionless) file.
	if sv, ok := v.(SchemaVersioner); ok {
		sv.SetSchemaVersion(expected)
	}

	return nil
}
