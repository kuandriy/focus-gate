package memory

import (
	"fmt"
	"os"
	"regexp"
	"strings"
	"time"

	"github.com/kuandriy/focus-gate/internal/persist"
)

// V1SchemaVersion is the schema literal used by pre-v2 memory files.
// Migration recognises this verbatim and converts the file into a v2
// single-chapter story.
const V1SchemaVersion = "1"

// MigrateV1FileToV2 reads a v1 memory file at `path`, converts it to v2
// in place (preserving the original at `<path>.v1.bak`), and runs
// RefreshDerived against the supplied vocab. Returns the loaded v2
// Memory on success.
//
// If the file is already v2 (schemaVersion: "2"), no work is done and
// the existing memory is returned unchanged.
//
// On any parse error the original file is left intact and the .v1.bak
// is not created — callers can safely retry after the user fixes
// whatever made the file unparseable.
func MigrateV1FileToV2(path string, vocab VocabSnapshot) (*Memory, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}

	// Already v2? Read normally.
	if isV2File(data) {
		m, err := parseFile(data)
		if err != nil {
			return nil, err
		}
		return m, nil
	}

	v1, body, err := parseV1(data)
	if err != nil {
		return nil, fmt.Errorf("parse v1: %w", err)
	}

	// Build the v2 chapter from the v1 body. The v1 body has `## What we
	// did` and `## Why` sections (validated by v1 binaries on write).
	what := extractV1Section(body, "## What we did")
	why := extractV1Section(body, "## Why")
	if strings.TrimSpace(what) == "" {
		// Some v1 files used `## What`; tolerate.
		what = extractV1Section(body, "## What")
	}
	if strings.TrimSpace(what) == "" || strings.TrimSpace(why) == "" {
		return nil, fmt.Errorf("v1 file missing required sections (## What we did / ## Why)")
	}

	// Derive a chapter time marker from created..updated.
	tf := ""
	switch {
	case !v1.Created.IsZero() && !v1.Updated.IsZero() && !sameDay(v1.Created, v1.Updated):
		tf = fmt.Sprintf("%s..%s",
			v1.Created.UTC().Format("2006-01-02"),
			v1.Updated.UTC().Format("2006-01-02"))
	case !v1.Created.IsZero():
		tf = v1.Created.UTC().Format("2006-01-02")
	}

	// Best-effort topics from the title: a single topic at weight 0.8 so
	// future learning iterations can append more granular ones.
	var topics []string
	if v1.Title != "" {
		topics = []string{v1.Title}
	}

	chapter := Chapter{
		Index:      1,
		Date:       pickV1Date(v1),
		Title:      "Initial",
		TimeMarker: tf,
		Assets:     append([]string{}, v1.Refs...),
		Topics:     topics,
		What:       strings.TrimSpace(what),
		Why:        strings.TrimSpace(why),
	}

	m := &Memory{
		ID:           v1.ID,
		Title:        v1.Title,
		Created:      v1.Created,
		Updated:      v1.Updated,
		TouchedBy:    v1.TouchedBy,
		ChaptersList: []Chapter{chapter},
	}

	// Preserve the original file alongside the rewrite so a botched
	// migration can be reverted manually.
	backup := path + ".v1.bak"
	if err := os.WriteFile(backup, data, 0644); err != nil {
		return nil, fmt.Errorf("write backup: %w", err)
	}

	// Overwrite the file in place with v2 representation.
	m.Body = renderChapters(m.ChaptersList)
	aggregateFromChapters(m)
	m.Version = len(m.ChaptersList)
	m.Chapters = len(m.ChaptersList)
	if m.Updated.IsZero() {
		m.Updated = chapter.Date
	}
	if m.Created.IsZero() {
		m.Created = chapter.Date
	}
	RefreshDerived(m, vocab)

	if err := persist.SaveAtomicBytes(path, render(m)); err != nil {
		return nil, fmt.Errorf("write v2: %w", err)
	}
	return m, nil
}

func sameDay(a, b time.Time) bool {
	return a.UTC().Format("2006-01-02") == b.UTC().Format("2006-01-02")
}

func pickV1Date(v1 v1Memory) time.Time {
	if !v1.Updated.IsZero() {
		return v1.Updated
	}
	if !v1.Created.IsZero() {
		return v1.Created
	}
	return time.Now().UTC().Truncate(time.Second)
}

// ---------------------------------------------------------------------------
// v1 frontmatter sniff + parser (private — only migration uses these).
// ---------------------------------------------------------------------------

var v1SchemaSniffRe = regexp.MustCompile(`(?m)^schemaVersion:\s*"?(\d+)"?\s*$`)

// IsV2File returns true if the file's frontmatter contains
// schemaVersion: "2". Anything else (including "1", missing, or
// malformed) returns false. Exported so the slash command can skip
// already-migrated files without re-parsing.
func IsV2File(data []byte) bool {
	return isV2File(data)
}

// isV2File is the internal entry point used by migration.
func isV2File(data []byte) bool {
	// Only inspect the frontmatter region.
	end := frontmatterEnd(data)
	if end < 0 {
		return false
	}
	m := v1SchemaSniffRe.FindStringSubmatch(string(data[:end]))
	if len(m) < 2 {
		return false
	}
	return m[1] == "2"
}

func frontmatterEnd(data []byte) int {
	// Find the second "---" fence.
	matches := fenceRe.FindAllIndex(data, 2)
	if len(matches) < 2 {
		return -1
	}
	return matches[1][1]
}

// v1Memory mirrors the v1 Memory struct shape so we can decode without
// pulling in the v1 binary.
type v1Memory struct {
	ID        string
	Title     string
	Refs      []string
	Created   time.Time
	Updated   time.Time
	TouchedBy int
}

// parseV1 splits a v1 file and decodes its frontmatter without going
// through setField (which now rejects schemaVersion=1).
func parseV1(data []byte) (v1Memory, string, error) {
	var v1 v1Memory
	text := string(data)
	fm, body, err := splitFrontMatter(text)
	if err != nil {
		return v1, "", err
	}
	for _, line := range strings.Split(fm, "\n") {
		line = strings.TrimRight(line, " \t")
		if line == "" || strings.HasPrefix(strings.TrimSpace(line), "#") {
			continue
		}
		colon := strings.IndexByte(line, ':')
		if colon < 0 {
			continue
		}
		key := strings.TrimSpace(line[:colon])
		raw := strings.TrimSpace(line[colon+1:])
		switch key {
		case "id":
			v1.ID = unquote(raw)
		case "title":
			v1.Title = unquote(raw)
		case "refs":
			v1.Refs = parseList(raw)
		case "created":
			if t, err := time.Parse(time.RFC3339, unquote(raw)); err == nil {
				v1.Created = t
			}
		case "updated":
			if t, err := time.Parse(time.RFC3339, unquote(raw)); err == nil {
				v1.Updated = t
			}
		case "touchedBy":
			var n int
			_, _ = fmt.Sscanf(raw, "%d", &n)
			v1.TouchedBy = n
		}
	}
	if v1.ID == "" {
		return v1, "", fmt.Errorf("v1 file missing id")
	}
	return v1, body, nil
}

// extractV1Section returns the body of a `## Heading` section up to the
// next `## ` heading or EOF. Trims whitespace.
func extractV1Section(body, heading string) string {
	lines := strings.Split(body, "\n")
	var out strings.Builder
	in := false
	for _, line := range lines {
		trimmed := strings.TrimRight(line, " \t")
		if !in {
			if trimmed == heading {
				in = true
			}
			continue
		}
		if strings.HasPrefix(trimmed, "## ") {
			break
		}
		out.WriteString(line)
		out.WriteString("\n")
	}
	return strings.TrimSpace(out.String())
}
