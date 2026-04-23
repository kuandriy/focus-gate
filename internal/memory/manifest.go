package memory

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/kuandriy/focus-gate/internal/persist"
)

// IndexEntry is one row in the manifest. Carries exactly the data Surface
// needs for cosine lookup — keeping the manifest small lets the hook
// load the whole thing once at startup and avoid per-prompt disk IO.
type IndexEntry struct {
	ID          string             `json:"id"`
	Title       string             `json:"title"`
	Path        string             `json:"path"` // relative to project data dir
	TopTerms    []string           `json:"topTerms"`
	Fingerprint map[string]float64 `json:"fingerprint"`
	Created     time.Time          `json:"created"`
	Updated     time.Time          `json:"updated"`
	TouchedBy   int                `json:"touchedBy"`
	Sources     []string           `json:"sources,omitempty"`
	Refs        []string           `json:"refs,omitempty"`
}

// Manifest is the on-disk index at <memoriesDir>/index.json. Cheap to
// load, cheap to scan, cheap to write — everything the hook needs for
// per-prompt surfacing sits here, so the full memory bodies stay on
// disk and aren't parsed again unless the user explicitly reads one.
type Manifest struct {
	SchemaVersion string       `json:"schemaVersion"`
	VocabHash     string       `json:"vocabHash"`
	RebuiltAt     time.Time    `json:"rebuiltAt"`
	Entries       []IndexEntry `json:"entries"`

	// dirty is in-memory only — set by touch increments so callers can
	// debounce manifest writes (at most once per prompt, not once per
	// surfaced entry).
	dirty bool `json:"-"`
}

// NewManifest returns an empty manifest stamped with the current schema.
func NewManifest() *Manifest {
	return &Manifest{
		SchemaVersion: SchemaVersion,
		RebuiltAt:     time.Now().UTC().Truncate(time.Second),
	}
}

// IndexPath is the canonical filename for the manifest inside a memories
// directory.
func IndexPath(dir string) string {
	return filepath.Join(dir, "index.json")
}

// Load reads the manifest from disk. A missing file is not an error —
// an empty manifest is returned so first-run callers don't special-case.
func Load(dir string) (*Manifest, error) {
	path := IndexPath(dir)
	data, err := os.ReadFile(path)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return NewManifest(), nil
		}
		return nil, fmt.Errorf("read manifest: %w", err)
	}
	var m Manifest
	if err := json.Unmarshal(data, &m); err != nil {
		return nil, fmt.Errorf("parse manifest: %w", err)
	}
	if m.SchemaVersion != "" && m.SchemaVersion != SchemaVersion {
		return NewManifest(), fmt.Errorf("manifest schemaVersion %q, binary speaks %q — starting fresh",
			m.SchemaVersion, SchemaVersion)
	}
	if m.SchemaVersion == "" {
		m.SchemaVersion = SchemaVersion // legacy file with no version — accept and stamp
	}
	return &m, nil
}

// Save persists the manifest to disk atomically. Clears the dirty flag.
func (m *Manifest) Save(dir string) error {
	if err := os.MkdirAll(dir, 0755); err != nil {
		return err
	}
	m.SchemaVersion = SchemaVersion
	m.dirty = false
	return persist.SaveAtomic(IndexPath(dir), m)
}

// SaveIfDirty writes only when something changed since the last Save or
// Rebuild. Hook path calls this at the end of each prompt so per-prompt
// touch increments batch into one write.
func (m *Manifest) SaveIfDirty(dir string) error {
	if !m.dirty {
		return nil
	}
	return m.Save(dir)
}

// Get returns an entry by ID and whether it was found.
func (m *Manifest) Get(id string) (*IndexEntry, bool) {
	for i := range m.Entries {
		if m.Entries[i].ID == id {
			return &m.Entries[i], true
		}
	}
	return nil, false
}

// Upsert adds or replaces an entry. Matching is by ID. Marks dirty.
func (m *Manifest) Upsert(entry IndexEntry) {
	for i := range m.Entries {
		if m.Entries[i].ID == entry.ID {
			m.Entries[i] = entry
			m.dirty = true
			return
		}
	}
	m.Entries = append(m.Entries, entry)
	m.dirty = true
}

// Remove deletes the entry with the given ID. Returns true if removed.
// Marks dirty.
func (m *Manifest) Remove(id string) bool {
	for i := range m.Entries {
		if m.Entries[i].ID == id {
			m.Entries = append(m.Entries[:i], m.Entries[i+1:]...)
			m.dirty = true
			return true
		}
	}
	return false
}

// Touch increments the touchedBy counter for an entry. Marks dirty so
// SaveIfDirty persists the change at the end of the prompt. Missing IDs
// are silently ignored — a race between manifest rebuild and touch
// shouldn't crash the hook.
func (m *Manifest) Touch(id string) {
	for i := range m.Entries {
		if m.Entries[i].ID == id {
			m.Entries[i].TouchedBy++
			m.dirty = true
			return
		}
	}
}

// Rebuild scans the memories directory for .md files and rebuilds the
// manifest from scratch. Called when vocabHash is stale, when a user
// hand-edits a file outside the binary, or via fg: memory health.
//
// Files that fail to parse are reported in errs but do not abort the
// rebuild — one broken file shouldn't invalidate the whole index.
func (m *Manifest) Rebuild(dir string, vocab VocabSnapshot) (errs []error) {
	entries, err := os.ReadDir(dir)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			// First run — fresh manifest stays as-is.
			m.Entries = nil
			m.VocabHash = vocab.Hash
			m.RebuiltAt = time.Now().UTC().Truncate(time.Second)
			m.dirty = true
			return nil
		}
		return []error{fmt.Errorf("read memories dir: %w", err)}
	}

	var fresh []IndexEntry
	for _, e := range entries {
		if e.IsDir() || !strings.HasSuffix(e.Name(), ".md") {
			continue
		}
		path := filepath.Join(dir, e.Name())
		mem, err := ReadFile(path)
		if err != nil {
			errs = append(errs, fmt.Errorf("%s: %w", e.Name(), err))
			continue
		}
		// Preserve touch count from the existing manifest if present.
		touched := 0
		if prev, ok := m.Get(mem.ID); ok {
			touched = prev.TouchedBy
		}
		// Re-derive lookup metadata so it stays consistent with the body
		// even if the user hand-edited the file.
		RefreshDerived(mem, vocab)
		fresh = append(fresh, IndexEntry{
			ID:          mem.ID,
			Title:       mem.Title,
			Path:        e.Name(),
			TopTerms:    mem.TopTerms,
			Fingerprint: mem.Fingerprint,
			Created:     mem.Created,
			Updated:     mem.Updated,
			TouchedBy:   touched,
			Sources:     mem.Sources,
			Refs:        mem.Refs,
		})
	}
	sort.Slice(fresh, func(i, j int) bool {
		return fresh[i].Updated.After(fresh[j].Updated)
	})

	m.Entries = fresh
	m.VocabHash = vocab.Hash
	m.RebuiltAt = time.Now().UTC().Truncate(time.Second)
	m.dirty = true
	return errs
}

// Dirty returns whether the manifest has unsaved changes. Exposed so
// hook callers can decide whether to acquire a lock before saving.
func (m *Manifest) Dirty() bool { return m.dirty }

// EnsureFresh loads the manifest and triggers Rebuild whenever the
// directory contents have drifted from the index — a hand-authored file
// the user dropped in, a memory deleted with rm, a body edited outside
// the binary, or a vocabulary shift on the engine side.
//
// Cheap in the steady state: the directory scan is one stat call per
// .md file, no body parsing unless a mismatch is found.
func EnsureFresh(dir string, vocab VocabSnapshot) (*Manifest, []error) {
	mf, err := Load(dir)
	if err != nil {
		mf = NewManifest()
	}
	needs, err := manifestNeedsRebuild(dir, mf, vocab)
	if err != nil {
		return mf, []error{err}
	}
	if !needs {
		return mf, nil
	}
	errs := mf.Rebuild(dir, vocab)
	if mf.Dirty() {
		if saveErr := mf.Save(dir); saveErr != nil {
			errs = append(errs, fmt.Errorf("save manifest after rebuild: %w", saveErr))
		}
	}
	return mf, errs
}

// manifestNeedsRebuild compares the manifest against the directory state
// and returns true if any of the following hold:
//   - vocabHash differs (engine vocabulary shifted since last write)
//   - a .md file on disk is not in the index
//   - an indexed file is missing from disk
//   - an indexed file's mtime is newer than the manifest's RebuiltAt
func manifestNeedsRebuild(dir string, mf *Manifest, vocab VocabSnapshot) (bool, error) {
	if mf.VocabHash != vocab.Hash {
		return true, nil
	}
	entries, err := os.ReadDir(dir)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			// Directory missing on disk but manifest claims entries.
			return len(mf.Entries) > 0, nil
		}
		return false, err
	}

	indexed := make(map[string]bool, len(mf.Entries))
	for _, e := range mf.Entries {
		indexed[e.Path] = true
	}

	onDisk := make(map[string]bool)
	for _, e := range entries {
		if e.IsDir() || !strings.HasSuffix(e.Name(), ".md") {
			continue
		}
		onDisk[e.Name()] = true
		if !indexed[e.Name()] {
			return true, nil
		}
		info, err := e.Info()
		if err != nil {
			continue
		}
		if info.ModTime().After(mf.RebuiltAt) {
			return true, nil
		}
	}

	for path := range indexed {
		if !onDisk[path] {
			return true, nil
		}
	}
	return false, nil
}
