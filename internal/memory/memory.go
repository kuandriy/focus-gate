// Package memory implements Focus Gate's long-term memory layer.
//
// Memories are append-only stories stored as Markdown files with YAML-ish
// frontmatter. Each story contains one or more chapters; new chapters are
// appended over time. Old chapters never edit. Frontmatter list fields
// (timeMarkers, interests, topics, assets) only grow — they aggregate the
// per-chapter metadata into a per-memory index for fast lookup.
//
// This package owns the on-disk format, the manifest index that enables
// fast lookup, and the Surface routine that renders pointer blocks into
// the hook's injected context. It does NOT call any LLM — all prose is
// authored by the host (via fg: memory commit) or by the user's own
// editor. See docs/SHARED_MEMORY_PLAN.md for the full design.
package memory

import (
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/kuandriy/focus-gate/internal/persist"
	"github.com/kuandriy/focus-gate/internal/tfidf"
)

// SchemaVersion is the on-disk schema version for memory front-matter and
// for the manifest file. Bump when the layout changes incompatibly.
const SchemaVersion = "2"

// IDPrefix is the leading literal of a memory ID.
const IDPrefix = "mem_"

// WeightedEntry pairs a name with a weight ∈ [0,1]. Used for interests
// and topics in the per-memory index. Weights are computed at write time
// from chapter coverage (chapters_mentioning / total_chapters), saturated
// at 1.0 and floored at 0.1.
//
// JSON shape uses lowercase field names to stay consistent with every
// other JSON tag in the package. Older manifests written before the
// tags were added serialized the struct with Go's default casing
// ("Name", "Weight"), so UnmarshalJSON below accepts either form to
// avoid silently dropping data on read.
type WeightedEntry struct {
	Name   string  `json:"name"`
	Weight float64 `json:"weight"`
}

// UnmarshalJSON accepts either {"name","weight"} (current) or
// {"Name","Weight"} (legacy) so v2 manifests written by pre-fix
// binaries still load. Marshal always emits the lowercase form via the
// struct tags above, so on the next save the file converges.
func (w *WeightedEntry) UnmarshalJSON(data []byte) error {
	var both struct {
		Name         string  `json:"name"`
		Weight       float64 `json:"weight"`
		NameLegacy   string  `json:"Name"`
		WeightLegacy float64 `json:"Weight"`
	}
	if err := json.Unmarshal(data, &both); err != nil {
		return err
	}
	w.Name = both.Name
	if w.Name == "" {
		w.Name = both.NameLegacy
	}
	w.Weight = both.Weight
	if w.Weight == 0 {
		w.Weight = both.WeightLegacy
	}
	return nil
}

// Memory is one long-term knowledge record. The struct is fully populated
// on load and fully regenerated on save — binary-managed fields
// (Created/Updated/TopTerms/Fingerprint/VocabHash/TouchedBy) are
// overwritten every write.
//
// Append-only invariants (enforced by AppendChapter and Validate):
//  1. Chapters never disappear or edit. Corrections are new chapters.
//  2. Frontmatter list fields only grow.
//  3. ID is immutable.
type Memory struct {
	// Stable across rewrites.
	ID    string // "mem_YYYYMMDD_<6-hex>"; stamped once
	Title string // ≤80 chars

	// Versioning. Bump on every chapter append.
	Version  int
	Chapters int

	// Binary-managed timestamps. Created stamped on first write; Updated
	// rewritten every save (= latest chapter's date).
	Created time.Time
	Updated time.Time

	// Aggregate index fields. Computed from ChaptersList at write time;
	// derived from body chapters at parse time. Frontmatter holds these
	// for fast manifest reads, but the chapter list is the source of
	// truth.
	TimeMarkers []string
	Interests   []WeightedEntry
	Topics      []WeightedEntry
	Assets      []string

	// Binary-managed fingerprint & touch counter. Overwritten on every
	// save (except TouchedBy, which is preserved via manifest tracking).
	TopTerms    []string
	Fingerprint map[string]float64
	VocabHash   string
	TouchedBy   int

	// Free-form chapter body. The on-disk Markdown after the frontmatter
	// fence; contains one or more `## Chapter N — date — title` blocks,
	// each with `### What` and `### Why` subsections.
	Body string

	// ChaptersList is parsed from Body on load; populated by callers
	// (AppendChapter, migration) before save. Drives aggregate computation.
	ChaptersList []Chapter
}

// Path returns the on-disk filename for a memory inside a given directory.
func (m *Memory) Path(dir string) string {
	return filepath.Join(dir, m.ID+".md")
}

// Validate checks the v2 invariants: title set and bounded, at least one
// chapter, every chapter has non-empty `### What` and `### Why`. Binary-
// managed field contents are never validated because we overwrite them.
func (m *Memory) Validate() error {
	if m.Title == "" {
		return errors.New("title required")
	}
	if len([]rune(m.Title)) > 80 {
		return fmt.Errorf("title too long (%d runes, max 80)", len([]rune(m.Title)))
	}
	if len(m.ChaptersList) == 0 {
		return errors.New("memory must have at least one chapter")
	}
	for i, ch := range m.ChaptersList {
		if strings.TrimSpace(ch.What) == "" {
			return fmt.Errorf("chapter %d missing or empty `### What`", i+1)
		}
		if strings.TrimSpace(ch.Why) == "" {
			return fmt.Errorf("chapter %d missing or empty `### Why`", i+1)
		}
	}
	return nil
}

// ReadFile loads a Memory from disk. Returns an error if the file is
// missing, malformed, or the schema version is not "2". Use
// MigrateV1FileToV2 to convert legacy files first.
//
// Binary-managed fields are populated from what's on disk; callers that
// want a freshly-derived Fingerprint should call RefreshDerived after
// loading.
func ReadFile(path string) (*Memory, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	return parseFile(data)
}

// WriteFile persists a Memory to `dir`, computing Created/Updated, the
// aggregate index from ChaptersList, the body from ChaptersList, and the
// derived fields (TopTerms, Fingerprint, VocabHash). Uses persist.SaveAtomic
// so a crash mid-write can't leave the file half-populated.
//
// If m.ID is empty, a new ID is stamped from the current date and a
// random 6-hex suffix; the ID is set back onto the caller's struct.
// Created is set to now only on first write (i.e. when it is the zero
// time); Updated is always refreshed.
func WriteFile(dir string, m *Memory, vocab VocabSnapshot) error {
	if len(m.ChaptersList) == 0 {
		return errors.New("memory must have at least one chapter before write")
	}

	// Recompute the canonical body from chapters so the on-disk
	// representation always matches the structured chapter list.
	m.Body = renderChapters(m.ChaptersList)

	// Recompute aggregate index from chapters. Binary-managed; we
	// overwrite whatever the caller set.
	aggregateFromChapters(m)

	// Bump versioning to reflect the chapter count.
	m.Version = len(m.ChaptersList)
	m.Chapters = len(m.ChaptersList)

	if err := m.Validate(); err != nil {
		return err
	}
	if m.ID == "" {
		m.ID = NewID(time.Now())
	}
	now := time.Now().UTC().Truncate(time.Second)
	if m.Created.IsZero() {
		m.Created = now
	}
	// Updated mirrors the latest chapter date when present, else now.
	if last := m.ChaptersList[len(m.ChaptersList)-1]; !last.Date.IsZero() {
		m.Updated = last.Date.UTC().Truncate(time.Second)
	} else {
		m.Updated = now
	}

	RefreshDerived(m, vocab)

	if err := os.MkdirAll(dir, 0755); err != nil {
		return fmt.Errorf("ensure memory dir: %w", err)
	}
	return persist.SaveAtomicBytes(m.Path(dir), render(m))
}

// RefreshDerived recomputes TopTerms and Fingerprint from the body using
// the current engine's vocabulary snapshot. Also stamps VocabHash. The
// caller is responsible for persisting the Memory afterwards.
func RefreshDerived(m *Memory, vocab VocabSnapshot) {
	source := m.Body + " " + m.Title + " " + strings.Join(m.Assets, " ")
	for _, t := range m.Topics {
		source += " " + t.Name
	}
	for _, in := range m.Interests {
		source += " " + in.Name
	}
	weights := vocab.Vectorize(source)
	m.Fingerprint = weights
	m.VocabHash = vocab.Hash

	terms := make([]string, 0, len(weights))
	for term := range weights {
		terms = append(terms, term)
	}
	sort.Slice(terms, func(i, j int) bool {
		wi, wj := weights[terms[i]], weights[terms[j]]
		if wi != wj {
			return wi > wj
		}
		return terms[i] < terms[j]
	})
	const topN = 8
	if len(terms) > topN {
		terms = terms[:topN]
	}
	m.TopTerms = terms
}

// AsVector returns the memory's fingerprint as a sorted tfidf.Vector so
// the existing cosine-similarity routine can be used for lookup without
// any conversion at call sites.
func (m *Memory) AsVector() tfidf.Vector {
	if len(m.Fingerprint) == 0 {
		return nil
	}
	return tfidf.NewVector(m.Fingerprint)
}

// ---------------------------------------------------------------------------
// ID generation
// ---------------------------------------------------------------------------

// NewID returns a fresh memory ID using the given wall-clock time. The
// 6-hex (24-bit) random suffix is sourced from crypto/rand to keep
// collisions astronomically unlikely even when many memories are
// created on the same day across a shared corpus.
//
// Falls back to nanosecond fraction if crypto/rand is unavailable
// (extremely rare; the function never returns an empty string so
// callers don't have to defensively check).
func NewID(t time.Time) string {
	var b [3]byte
	if _, err := rand.Read(b[:]); err == nil {
		return fmt.Sprintf("%s%s_%s", IDPrefix, t.UTC().Format("20060102"), hex.EncodeToString(b[:]))
	}
	return fmt.Sprintf("%s%s_%06x", IDPrefix, t.UTC().Format("20060102"),
		uint32(t.UnixNano())&0xffffff)
}
