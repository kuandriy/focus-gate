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
	"github.com/kuandriy/focus-gate/internal/text"
)

// DefaultSourceName is the synthesized name for the project's local
// memory directory when no explicit `memory.sources` config is provided.
// Multi-source attach (Phase 5) extends this; before then, every loaded
// manifest belongs to "personal".
const DefaultSourceName = "personal"

// IndexEntry is one row in the manifest. Carries exactly the data Surface
// needs for multi-tier matching — keeping the manifest small lets the
// hook load the whole thing once at startup and avoid per-prompt disk IO.
//
// Body content (chapter What/Why prose) is not stored here; it sits in
// the .md file and is read only when the user runs `fg: memory show`.
type IndexEntry struct {
	ID          string             `json:"id"`
	Source      string             `json:"source"`
	Title       string             `json:"title"`
	Path        string             `json:"path"` // relative to the source's memories dir
	Version     int                `json:"version"`
	Chapters    int                `json:"chapters"`
	TimeMarkers []string           `json:"timeMarkers,omitempty"`
	Interests   []WeightedEntry    `json:"interests,omitempty"`
	Topics      []WeightedEntry    `json:"topics,omitempty"`
	Assets      []string           `json:"assets,omitempty"`
	TopTerms    []string           `json:"topTerms,omitempty"`
	Fingerprint map[string]float64 `json:"fingerprint"`
	Created     time.Time          `json:"created"`
	Updated     time.Time          `json:"updated"`
	TouchedBy   int                `json:"touchedBy"`
	// LatestSnippet is a short single-line excerpt from the last
	// chapter's What. Surface renders it under each pointer row so the
	// LLM can judge "is this memory actually relevant?" at a glance,
	// without first calling Read on the file. Empty for pre-fix
	// manifests until the next Rebuild — forward-compatible.
	LatestSnippet string `json:"latestSnippet,omitempty"`
}

// Manifest is the on-disk index at <memoriesDir>/index.json. Cheap to
// load, cheap to scan, cheap to write — everything Surface needs sits
// in entries + inverted indexes.
//
// ByAsset, ByInterest, ByTopic map a normalized key (lowercase trim) to
// the IDs of memories that contain it. Populated by Rebuild and
// consulted by Surface for asset-first traversal without full scans.
type Manifest struct {
	SchemaVersion string              `json:"schemaVersion"`
	Source        string              `json:"source"`
	VocabHash     string              `json:"vocabHash"`
	RebuiltAt     time.Time           `json:"rebuiltAt"`
	Entries       []IndexEntry        `json:"entries"`
	ByAsset       map[string][]string `json:"byAsset,omitempty"`
	ByInterest    map[string][]string `json:"byInterest,omitempty"`
	ByTopic       map[string][]string `json:"byTopic,omitempty"`

	// dirty is in-memory only — set by touch increments so callers can
	// debounce manifest writes (at most once per prompt, not once per
	// surfaced entry).
	dirty bool `json:"-"`
}

// NewManifest returns an empty manifest stamped with the current schema
// and the default source name. Multi-source flow (Phase 5) sets
// `Source` explicitly; pre-Phase-5 callers get "personal".
func NewManifest() *Manifest {
	return &Manifest{
		SchemaVersion: SchemaVersion,
		Source:        DefaultSourceName,
		RebuiltAt:     time.Now().UTC().Truncate(time.Second),
	}
}

// IndexPath is the canonical filename for the manifest inside a memories
// directory.
func IndexPath(dir string) string {
	return filepath.Join(dir, "index.json")
}

// Load reads the manifest from disk. Always returns a non-nil Manifest:
// missing file → empty manifest with no error; read or parse failure →
// empty manifest plus a non-fatal error the caller can surface. This
// way callers don't need to nil-check the returned pointer before
// passing it to Rebuild/Save on the recovery path.
//
// On schema mismatch a fresh empty manifest is returned along with a
// (non-fatal) error so the caller can rebuild from disk.
func Load(dir string) (*Manifest, error) {
	path := IndexPath(dir)
	data, err := os.ReadFile(path)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return NewManifest(), nil
		}
		return NewManifest(), fmt.Errorf("read manifest: %w", err)
	}
	var m Manifest
	if err := json.Unmarshal(data, &m); err != nil {
		return NewManifest(), fmt.Errorf("parse manifest: %w", err)
	}
	if m.SchemaVersion != "" && m.SchemaVersion != SchemaVersion {
		return NewManifest(), fmt.Errorf("manifest schemaVersion %q, binary speaks %q — starting fresh",
			m.SchemaVersion, SchemaVersion)
	}
	if m.SchemaVersion == "" {
		m.SchemaVersion = SchemaVersion // legacy file with no version — accept and stamp
	}
	if m.Source == "" {
		m.Source = DefaultSourceName
	}
	// Stamp Source onto entries that lack one (forward-compat for older
	// v2 files written before Source was tracked per entry).
	for i := range m.Entries {
		if m.Entries[i].Source == "" {
			m.Entries[i].Source = m.Source
		}
	}
	return &m, nil
}

// Save persists the manifest to disk atomically. Clears the dirty flag.
func (m *Manifest) Save(dir string) error {
	if err := os.MkdirAll(dir, 0755); err != nil {
		return err
	}
	m.SchemaVersion = SchemaVersion
	if m.Source == "" {
		m.Source = DefaultSourceName
	}
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
// Inverted indexes are NOT incrementally maintained here — call
// RebuildInvertedIndexes after a batch of upserts, or rely on Rebuild
// for the full path. Matches the documented contract: inverted indexes
// are populated by Rebuild.
func (m *Manifest) Upsert(entry IndexEntry) {
	if entry.Source == "" {
		entry.Source = m.Source
	}
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
// hand-edits a file outside the binary, or via fg: memory reindex.
//
// Files that fail to parse are reported in errs but do not abort the
// rebuild — one broken file shouldn't invalidate the whole index.
func (m *Manifest) Rebuild(dir string, vocab VocabSnapshot) (errs []error) {
	if m.Source == "" {
		m.Source = DefaultSourceName
	}
	entries, err := os.ReadDir(dir)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			// First run — fresh manifest stays as-is.
			m.Entries = nil
			m.VocabHash = vocab.Hash
			m.RebuiltAt = time.Now().UTC().Truncate(time.Second)
			m.ByAsset, m.ByInterest, m.ByTopic = nil, nil, nil
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
			ID:            mem.ID,
			Source:        m.Source,
			Title:         mem.Title,
			Path:          e.Name(),
			Version:       mem.Version,
			Chapters:      mem.Chapters,
			TimeMarkers:   mem.TimeMarkers,
			Interests:     mem.Interests,
			Topics:        mem.Topics,
			Assets:        mem.Assets,
			TopTerms:      mem.TopTerms,
			Fingerprint:   mem.Fingerprint,
			Created:       mem.Created,
			Updated:       mem.Updated,
			TouchedBy:     touched,
			LatestSnippet: latestChapterSnippet(mem),
		})
	}
	sort.Slice(fresh, func(i, j int) bool {
		return fresh[i].Updated.After(fresh[j].Updated)
	})

	m.Entries = fresh
	m.VocabHash = vocab.Hash
	m.RebuiltAt = time.Now().UTC().Truncate(time.Second)
	m.RebuildInvertedIndexes()
	m.dirty = true
	return errs
}

// RebuildInvertedIndexes recomputes ByAsset / ByInterest / ByTopic from
// the current entry list. Keys are normalized via normalizeKey so case
// and whitespace differences collapse. Per-key ID lists are sorted for
// deterministic on-disk output.
//
// O(entries × (assets + interests + topics)) — small for realistic
// catalogs and only paid at rebuild time, not on the prompt path.
func (m *Manifest) RebuildInvertedIndexes() {
	byAsset := map[string][]string{}
	byInterest := map[string][]string{}
	byTopic := map[string][]string{}
	for _, e := range m.Entries {
		for _, a := range e.Assets {
			k := normalizeKey(a)
			if k == "" {
				continue
			}
			byAsset[k] = appendUnique(byAsset[k], e.ID)
		}
		for _, in := range e.Interests {
			k := normalizeKey(in.Name)
			if k == "" {
				continue
			}
			byInterest[k] = appendUnique(byInterest[k], e.ID)
		}
		for _, t := range e.Topics {
			k := normalizeKey(t.Name)
			if k == "" {
				continue
			}
			byTopic[k] = appendUnique(byTopic[k], e.ID)
		}
	}
	for k := range byAsset {
		sort.Strings(byAsset[k])
	}
	for k := range byInterest {
		sort.Strings(byInterest[k])
	}
	for k := range byTopic {
		sort.Strings(byTopic[k])
	}
	m.ByAsset = byAsset
	m.ByInterest = byInterest
	m.ByTopic = byTopic
}

// latestChapterSnippet returns a one-line excerpt from the latest
// chapter's What field, suitable for inline display in a Surface row.
// Truncates at the first sentence boundary or 120 runes, whichever
// comes first. Returns "" when no chapters or empty What.
func latestChapterSnippet(m *Memory) string {
	if m == nil || len(m.ChaptersList) == 0 {
		return ""
	}
	what := strings.TrimSpace(m.ChaptersList[len(m.ChaptersList)-1].What)
	if what == "" {
		return ""
	}
	// Collapse internal whitespace + line breaks so the snippet renders
	// on a single line regardless of authoring style.
	what = strings.Join(strings.Fields(what), " ")
	// First-sentence boundary: take everything up to the first ". ", "! ",
	// or "? ". Trailing terminator stays attached for natural reading.
	for _, term := range []string{". ", "! ", "? "} {
		if i := strings.Index(what, term); i > 0 {
			what = what[:i+1]
			break
		}
	}
	return text.TruncateRunesWithSuffix(what, 120, "…")
}

// normalizeKey lowercases and trims a string for use as an inverted-
// index key. Used by both index population and lookup so producers and
// consumers see the same key regardless of caller-side casing.
func normalizeKey(s string) string {
	return strings.ToLower(strings.TrimSpace(s))
}

func appendUnique(list []string, item string) []string {
	for _, existing := range list {
		if existing == item {
			return list
		}
	}
	return append(list, item)
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
