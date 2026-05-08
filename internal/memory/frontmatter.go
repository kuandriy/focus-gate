package memory

import (
	"errors"
	"fmt"
	"regexp"
	"sort"
	"strings"
	"time"
)

// ---------------------------------------------------------------------------
// Frontmatter parser (narrow YAML-ish subset, v2)
// ---------------------------------------------------------------------------

// parseFile splits a memory file into front-matter + body, decodes the
// front-matter into a Memory struct, parses the body chapters, and
// returns it. Validation is deferred — callers that need strict
// validation call m.Validate().
//
// Errors when schemaVersion is not "2". Use migration to convert v1 files.
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

	// Body chapters are the source of truth; aggregate fields in
	// frontmatter are a cache. Parse chapters and re-derive aggregates
	// to detect drift (e.g. a hand edit added a chapter without
	// updating frontmatter).
	if strings.TrimSpace(body) != "" {
		chapters, err := parseChapters(body)
		if err != nil {
			return nil, fmt.Errorf("parse chapters: %w", err)
		}
		m.ChaptersList = chapters
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

// decodeFrontMatter parses the v2 frontmatter format. Unknown keys are
// ignored so older binaries can be forgiving against newer files.
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
		v := unquote(rawValue)
		if v != "" && v != SchemaVersion {
			return fmt.Errorf("unsupported schemaVersion %q (this binary speaks %q)", v, SchemaVersion)
		}
	case "id":
		m.ID = unquote(rawValue)
	case "title":
		m.Title = unquote(rawValue)
	case "version":
		var n int
		_, _ = fmt.Sscanf(rawValue, "%d", &n)
		m.Version = n
	case "chapters":
		var n int
		_, _ = fmt.Sscanf(rawValue, "%d", &n)
		m.Chapters = n
	case "timeMarkers":
		m.TimeMarkers = parseList(rawValue)
	case "interests":
		m.Interests = parseWeightedList(rawValue)
	case "topics":
		m.Topics = parseWeightedList(rawValue)
	case "assets":
		m.Assets = parseList(rawValue)
	case "topTerms":
		m.TopTerms = parseList(rawValue)
	case "fingerprint":
		m.Fingerprint = parseWeightMap(unquote(rawValue))
	case "vocabHash":
		m.VocabHash = unquote(rawValue)
	case "touchedBy":
		var n int
		_, _ = fmt.Sscanf(rawValue, "%d", &n)
		m.TouchedBy = n
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
	parts := splitFlowList(inner)
	out := make([]string, 0, len(parts))
	for _, p := range parts {
		v := unquote(strings.TrimSpace(p))
		if v != "" {
			out = append(out, v)
		}
	}
	return out
}

// splitFlowList splits on commas but respects double-quoted strings so
// quoted entries containing commas survive intact. Used by parseList and
// parseWeightedList.
func splitFlowList(inner string) []string {
	var out []string
	var cur strings.Builder
	inQuotes := false
	for i := 0; i < len(inner); i++ {
		c := inner[i]
		if c == '"' {
			inQuotes = !inQuotes
			cur.WriteByte(c)
			continue
		}
		if c == ',' && !inQuotes {
			out = append(out, cur.String())
			cur.Reset()
			continue
		}
		cur.WriteByte(c)
	}
	if cur.Len() > 0 {
		out = append(out, cur.String())
	}
	return out
}

// parseWeightedList reads a flow-style list of weighted entries:
//
//	[name@0.90, "another name@0.50"]
//
// The "@<float>" suffix is optional; missing or unparseable weights
// default to 1.0. Used by interests and topics.
func parseWeightedList(raw string) []WeightedEntry {
	items := parseList(raw)
	if len(items) == 0 {
		return nil
	}
	out := make([]WeightedEntry, 0, len(items))
	for _, item := range items {
		name, weight := splitWeighted(item)
		if name == "" {
			continue
		}
		out = append(out, WeightedEntry{Name: name, Weight: weight})
	}
	return out
}

func splitWeighted(item string) (string, float64) {
	at := strings.LastIndex(item, "@")
	if at < 0 {
		return strings.TrimSpace(item), 1.0
	}
	name := strings.TrimSpace(item[:at])
	wraw := strings.TrimSpace(item[at+1:])
	var w float64
	if _, err := fmt.Sscanf(wraw, "%f", &w); err != nil || w <= 0 {
		return name, 1.0
	}
	if w > 1.0 {
		w = 1.0
	}
	return name, w
}

// parseWeightMap parses "term1:0.48 term2:0.36 term3:0.12" into a map.
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
// Frontmatter renderer
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
	fmt.Fprintf(&b, "version: %d\n", m.Version)
	fmt.Fprintf(&b, "chapters: %d\n", m.Chapters)
	fmt.Fprintf(&b, "created: %q\n", m.Created.UTC().Format(time.RFC3339))
	fmt.Fprintf(&b, "updated: %q\n", m.Updated.UTC().Format(time.RFC3339))
	writeListField(&b, "timeMarkers", m.TimeMarkers)
	writeWeightedListField(&b, "interests", m.Interests)
	writeWeightedListField(&b, "topics", m.Topics)
	writeListField(&b, "assets", m.Assets)
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

func writeWeightedListField(b *strings.Builder, name string, items []WeightedEntry) {
	if len(items) == 0 {
		fmt.Fprintf(b, "%s: []\n", name)
		return
	}
	fmt.Fprintf(b, "%s: [", name)
	for i, it := range items {
		if i > 0 {
			b.WriteString(", ")
		}
		fmt.Fprintf(b, "%q", fmt.Sprintf("%s@%.2f", it.Name, it.Weight))
	}
	b.WriteString("]\n")
}

// formatWeightMap emits "term1:0.4821 term2:0.3654 ..." with deterministic
// ordering (descending weight, then term asc). Precision is 4 decimal
// places — enough for cosine rank stability.
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
