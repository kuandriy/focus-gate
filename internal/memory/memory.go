// Package memory implements Focus Gate's long-term memory layer.
//
// Memories are small Markdown files stored under the project's data
// directory that distill the valuable subset of prior Focus Gate activity
// — completed investigations, project conventions, "we did X because Y"
// decisions. Each file has a narrow YAML front-matter with binary-managed
// lookup metadata, followed by a free-form Markdown body.
//
// This package owns the on-disk format, the manifest index that enables
// fast cosine lookup, and the Surface routine that renders pointer blocks
// into the hook's injected context. It does NOT call any LLM — all prose
// is authored by the host (via fg: memory commit) or by the user's own
// editor. See docs/LONG_TERM_MEMORY_PLAN.md for the full design.
package memory

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"time"

	"github.com/kuandriy/focus-gate/internal/persist"
	"github.com/kuandriy/focus-gate/internal/tfidf"
)

// SchemaVersion is the on-disk schema version for memory front-matter and
// for the manifest file. Bump when the layout changes incompatibly.
const SchemaVersion = "1"

// IDPrefix is the leading literal of a memory ID. Makes a raw IDs
// recognizable at a glance and scoped so future features can introduce
// different stores without collision.
const IDPrefix = "mem_"

// Memory is one long-term knowledge record, parsed from a Markdown file
// with YAML-ish front-matter. The struct is fully populated on load and
// fully regenerated on save — binary-managed fields (Created/Updated/
// TopTerms/Fingerprint/VocabHash/TouchedBy) are overwritten every write,
// so whatever the caller sets in them is ignored.
type Memory struct {
	// Stable across rewrites.
	ID      string   // "mem_YYYYMMDD_<6-hex>"; stamped once
	Title   string   // ≤80 chars
	Sources []string // tree IDs that contributed; append-only on merge
	Refs    []string // project-relative file paths mentioned in the body

	// Binary-managed. Overwritten on every save.
	Created     time.Time
	Updated     time.Time
	TopTerms    []string           // top N terms by weight, human-readable
	Fingerprint map[string]float64 // term → weight for cosine lookup
	VocabHash   string             // snapshot of engine vocab at write time
	TouchedBy   int                // increments when surfaced in a prompt

	// Free-form markdown body after the second "---" fence. Two sections
	// are required (### 4.2 of the plan): "## What we did" and "## Why".
	// The rest is unvalidated.
	Body string
}

// Required sections that validation checks for. Missing either one causes
// Validate to return a non-nil error.
var requiredSections = []string{"## What we did", "## Why"}

// Path returns the on-disk filename for a memory inside a given directory.
// Uses the memory's ID as the basename so the relationship is trivially
// traceable (no filename parsing, no collisions, no rename chaos).
func (m *Memory) Path(dir string) string {
	return filepath.Join(dir, m.ID+".md")
}

// Validate checks the invariants the LLM can get wrong: required sections
// present and non-empty, title length sane, body not blank. Binary-managed
// field contents are never validated because we overwrite them.
func (m *Memory) Validate() error {
	if m.Title == "" {
		return errors.New("title required")
	}
	if len([]rune(m.Title)) > 80 {
		return fmt.Errorf("title too long (%d runes, max 80)", len([]rune(m.Title)))
	}
	if strings.TrimSpace(m.Body) == "" {
		return errors.New("body is empty")
	}
	for _, section := range requiredSections {
		if !hasSection(m.Body, section) {
			return fmt.Errorf("required section %q missing or empty", section)
		}
	}
	return nil
}

// hasSection returns true if the body contains a section heading exactly
// matching `heading` and that section has non-whitespace content before
// the next "## " heading (or EOF).
func hasSection(body, heading string) bool {
	lines := strings.Split(body, "\n")
	for i, line := range lines {
		if strings.TrimRight(line, " \t") != heading {
			continue
		}
		for j := i + 1; j < len(lines); j++ {
			next := lines[j]
			trimmed := strings.TrimSpace(next)
			if strings.HasPrefix(next, "## ") {
				break
			}
			if trimmed != "" {
				return true
			}
		}
		return false
	}
	return false
}

// ReadFile loads a Memory from disk. Returns an error if the file is
// missing, malformed, or fails required-section validation.
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

// WriteFile persists a Memory to `dir`, computing Created/Updated,
// refreshing TopTerms and Fingerprint from the body, and stamping the
// schemaVersion. Uses persist.SaveAtomic so a crash mid-write can't leave
// the file half-populated.
//
// If m.ID is empty, a new ID is stamped from the current date and a
// random 6-hex suffix; the ID is set back onto the caller's struct.
// Created is set to now only on first write (i.e. when it is the zero
// time); Updated is always refreshed.
func WriteFile(dir string, m *Memory, vocab VocabSnapshot) error {
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
	m.Updated = now

	RefreshDerived(m, vocab)

	return persist.SaveAtomicBytes(m.Path(dir), render(m))
}

// RefreshDerived recomputes TopTerms and Fingerprint from the body using
// the current engine's vocabulary snapshot. Also stamps VocabHash. The
// caller is responsible for persisting the Memory afterwards.
func RefreshDerived(m *Memory, vocab VocabSnapshot) {
	weights := vocab.Vectorize(m.Body + " " + m.Title + " " + strings.Join(m.Refs, " "))
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
// 6-hex suffix is derived from nanosecond fraction so multiple IDs minted
// in the same second remain distinct without requiring a source of
// randomness.
func NewID(t time.Time) string {
	return fmt.Sprintf("%s%s_%06x", IDPrefix, t.UTC().Format("20060102"),
		uint32(t.UnixNano())&0xffffff)
}

// ---------------------------------------------------------------------------
// Front-matter parser (narrow YAML-ish subset)
// ---------------------------------------------------------------------------

// parseFile splits a memory file into front-matter + body, decodes the
// front-matter into a Memory struct, and returns it. Validation is
// deferred — callers that need strict validation call m.Validate().
func parseFile(data []byte) (*Memory, error) {
	text := string(data)
	fm, body, err := splitFrontMatter(text)
	if err != nil {
		return nil, err
	}

	m := &Memory{Body: body}
	if err := decodeFrontMatter(fm, m); err != nil {
		return nil, err
	}
	if m.ID == "" {
		return nil, errors.New("frontmatter missing id")
	}
	return m, nil
}

var fenceRe = regexp.MustCompile(`(?m)^---\s*$`)

// splitFrontMatter expects `---\n<fm>\n---\n<body>`. Returns the
// front-matter block without the fences and the body unchanged.
func splitFrontMatter(text string) (fm, body string, err error) {
	locs := fenceRe.FindAllStringIndex(text, 2)
	if len(locs) < 2 {
		return "", "", errors.New("missing --- frontmatter fence")
	}
	open := locs[0]
	close := locs[1]
	if open[0] != 0 {
		return "", "", errors.New("--- fence must be at the start of file")
	}
	fm = strings.TrimSpace(text[open[1]:close[0]])
	body = strings.TrimLeft(text[close[1]:], "\n")
	return fm, body, nil
}

// decodeFrontMatter parses a narrow YAML-ish format:
//
//	key: value                         (scalar string)
//	key: [item1, item2]                (flow-style list)
//	key: 42                            (integer)
//	key: "quoted"                      (quoted string, escapes unsupported)
//
// Unknown keys are preserved as comments in the on-disk file but dropped
// on load (the rewrite path overwrites everything from the struct). This
// is intentional — future fields land cleanly without requiring old
// binaries to understand them.
func decodeFrontMatter(fm string, m *Memory) error {
	for _, line := range strings.Split(fm, "\n") {
		line = strings.TrimRight(line, " \t")
		if line == "" || strings.HasPrefix(strings.TrimSpace(line), "#") {
			continue
		}
		colon := strings.IndexByte(line, ':')
		if colon < 0 {
			return fmt.Errorf("malformed frontmatter line: %q", line)
		}
		key := strings.TrimSpace(line[:colon])
		value := strings.TrimSpace(line[colon+1:])
		if err := setField(m, key, value); err != nil {
			return err
		}
	}
	return nil
}

// setField populates one struct field by name. Silently ignores unknown
// keys so older binaries keep working against newer files.
func setField(m *Memory, key, rawValue string) error {
	switch key {
	case "schemaVersion":
		if v := unquote(rawValue); v != "" && v != SchemaVersion {
			return fmt.Errorf("unsupported schemaVersion %q (this binary speaks %q)", v, SchemaVersion)
		}
	case "id":
		m.ID = unquote(rawValue)
	case "title":
		m.Title = unquote(rawValue)
	case "sources":
		m.Sources = parseList(rawValue)
	case "refs":
		m.Refs = parseList(rawValue)
	case "topTerms":
		m.TopTerms = parseList(rawValue)
	case "fingerprint":
		m.Fingerprint = parseWeightMap(unquote(rawValue))
	case "vocabHash":
		m.VocabHash = unquote(rawValue)
	case "touchedBy":
		var n int
		_, err := fmt.Sscanf(rawValue, "%d", &n)
		if err == nil {
			m.TouchedBy = n
		}
	case "created":
		if t, err := time.Parse(time.RFC3339, unquote(rawValue)); err == nil {
			m.Created = t
		}
	case "updated":
		if t, err := time.Parse(time.RFC3339, unquote(rawValue)); err == nil {
			m.Updated = t
		}
	}
	return nil
}

// unquote strips surrounding double quotes if present. Escapes are not
// interpreted — quoted values here never contain embedded quotes in
// practice (title is ≤80 chars, IDs are ASCII).
func unquote(s string) string {
	if len(s) >= 2 && s[0] == '"' && s[len(s)-1] == '"' {
		return s[1 : len(s)-1]
	}
	return s
}

// parseList reads a flow-style YAML list:  [a, b, "c d"].
// Returns nil for an empty list, or if the value is not a list.
func parseList(raw string) []string {
	raw = strings.TrimSpace(raw)
	if !strings.HasPrefix(raw, "[") || !strings.HasSuffix(raw, "]") {
		return nil
	}
	inner := strings.TrimSpace(raw[1 : len(raw)-1])
	if inner == "" {
		return nil
	}
	parts := strings.Split(inner, ",")
	out := make([]string, 0, len(parts))
	for _, p := range parts {
		v := unquote(strings.TrimSpace(p))
		if v != "" {
			out = append(out, v)
		}
	}
	return out
}

// parseWeightMap parses "term1:0.48 term2:0.36 term3:0.12" into a map.
// This is the storage format for the memory's fingerprint — compact,
// YAML-safe (single quoted string), trivial to parse.
func parseWeightMap(s string) map[string]float64 {
	s = strings.TrimSpace(s)
	if s == "" {
		return nil
	}
	out := map[string]float64{}
	for _, pair := range strings.Fields(s) {
		colon := strings.LastIndexByte(pair, ':')
		if colon < 0 {
			continue
		}
		term := pair[:colon]
		var w float64
		if _, err := fmt.Sscanf(pair[colon+1:], "%f", &w); err != nil {
			continue
		}
		if term != "" {
			out[term] = w
		}
	}
	return out
}

// ---------------------------------------------------------------------------
// Front-matter renderer
// ---------------------------------------------------------------------------

// render emits the complete on-disk representation: fenced frontmatter +
// body. Field order is fixed for deterministic output — makes diffs
// readable and tests simple.
func render(m *Memory) []byte {
	var b strings.Builder
	b.WriteString("---\n")
	fmt.Fprintf(&b, "schemaVersion: %q\n", SchemaVersion)
	fmt.Fprintf(&b, "id: %q\n", m.ID)
	fmt.Fprintf(&b, "title: %q\n", m.Title)
	writeListField(&b, "sources", m.Sources)
	writeListField(&b, "refs", m.Refs)
	fmt.Fprintf(&b, "created: %q\n", m.Created.UTC().Format(time.RFC3339))
	fmt.Fprintf(&b, "updated: %q\n", m.Updated.UTC().Format(time.RFC3339))
	writeListField(&b, "topTerms", m.TopTerms)
	fmt.Fprintf(&b, "fingerprint: %q\n", formatWeightMap(m.Fingerprint))
	fmt.Fprintf(&b, "vocabHash: %q\n", m.VocabHash)
	fmt.Fprintf(&b, "touchedBy: %d\n", m.TouchedBy)
	b.WriteString("---\n\n")
	b.WriteString(strings.TrimLeft(m.Body, "\n"))
	if !strings.HasSuffix(m.Body, "\n") {
		b.WriteByte('\n')
	}
	return []byte(b.String())
}

func writeListField(b *strings.Builder, name string, items []string) {
	if len(items) == 0 {
		fmt.Fprintf(b, "%s: []\n", name)
		return
	}
	fmt.Fprintf(b, "%s: [", name)
	for i, it := range items {
		if i > 0 {
			b.WriteString(", ")
		}
		fmt.Fprintf(b, "%q", it)
	}
	b.WriteString("]\n")
}

// formatWeightMap emits "term1:0.4821 term2:0.3654 ..." with deterministic
// ordering (descending weight, then term asc). Precision is 4 decimal
// places — enough for cosine rank stability, tight enough to keep
// fingerprint lines readable.
func formatWeightMap(w map[string]float64) string {
	if len(w) == 0 {
		return ""
	}
	terms := make([]string, 0, len(w))
	for t := range w {
		terms = append(terms, t)
	}
	sort.Slice(terms, func(i, j int) bool {
		wi, wj := w[terms[i]], w[terms[j]]
		if wi != wj {
			return wi > wj
		}
		return terms[i] < terms[j]
	})
	parts := make([]string, len(terms))
	for i, t := range terms {
		parts[i] = fmt.Sprintf("%s:%.4f", t, w[t])
	}
	return strings.Join(parts, " ")
}
