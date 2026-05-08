package memory

import (
	"errors"
	"fmt"
	"regexp"
	"sort"
	"strings"
	"time"
)

// Chapter is one entry in a memory's append-only story. Each chapter
// has its own time marker, the assets/interests/topics it introduces,
// and the What/Why prose. Aggregating across all chapters produces the
// per-memory frontmatter index.
type Chapter struct {
	// Sequence number, 1-based. Set by AppendChapter; surface text
	// renders this in the chapter heading.
	Index int

	// Date is the chapter's anchor date — when the knowledge captured
	// here is dated, not when the file was written. Used to compute
	// Memory.Updated and to render the chapter heading.
	Date time.Time

	// Title is the human-readable chapter label after the date in the
	// heading: "## Chapter 2 — 2026-04-12 — Refresh token rotation"
	// → "Refresh token rotation".
	Title string

	// TimeMarker is a free-form short temporal label — an ISO date
	// ("2026-04-12"), an ISO range ("2026-04-10..2026-04-12"), or a
	// developer-shorthand marker ("sprint-42", "v1.2-release",
	// "Q4-2026"). Captured verbatim from the chapter metadata block;
	// aggregated into Memory.TimeMarkers on save.
	TimeMarker string

	// Assets, Interests, Topics list the items this chapter introduces
	// or touches. Names only — weights are derived at the memory level
	// from chapter coverage.
	Assets    []string
	Interests []string
	Topics    []string

	// What and Why are the chapter's body sections, captured verbatim
	// (TrimSpaced). Both must be non-empty for validation to pass.
	What string
	Why  string
}

// AppendChapter adds a new chapter to a memory. It enforces the append-
// only invariants:
//   - Chapters never disappear (this function only grows the list).
//   - The new chapter's Index is set to len(prior) + 1 — callers do not
//     supply it.
//   - Frontmatter list fields only grow (handled by aggregateFromChapters
//     on save: union of all chapters' interests/topics/assets).
//
// Returns an error if the chapter is malformed (empty What or Why). The
// caller follows up with WriteFile to persist.
func AppendChapter(m *Memory, ch Chapter) error {
	if m == nil {
		return errors.New("nil memory")
	}
	if strings.TrimSpace(ch.What) == "" {
		return errors.New("chapter requires non-empty `### What`")
	}
	if strings.TrimSpace(ch.Why) == "" {
		return errors.New("chapter requires non-empty `### Why`")
	}
	ch.Index = len(m.ChaptersList) + 1
	if ch.Date.IsZero() {
		ch.Date = time.Now().UTC().Truncate(time.Second)
	}
	m.ChaptersList = append(m.ChaptersList, ch)
	// Version/Chapters reflect the in-memory list immediately; WriteFile
	// will assert this on persist.
	m.Version = len(m.ChaptersList)
	m.Chapters = len(m.ChaptersList)
	return nil
}

// ---------------------------------------------------------------------------
// Aggregation: chapters → memory frontmatter index
// ---------------------------------------------------------------------------

// aggregateFromChapters rebuilds Memory.TimeMarkers, Memory.Assets,
// Memory.Interests, and Memory.Topics from the chapter list. Weights for
// interests and topics are derived from chapter coverage:
//
//	weight = chapters_mentioning / total_chapters
//	saturated at 1.0, floored at 0.1
//
// Names are de-duplicated case-insensitively but rendered in the casing
// of their first occurrence. Order across the slices is deterministic
// for stable on-disk output: TimeMarkers preserve append order; Assets/
// Interests/Topics sort by descending weight, then ascending name.
func aggregateFromChapters(m *Memory) {
	if len(m.ChaptersList) == 0 {
		m.TimeMarkers = nil
		m.Assets = nil
		m.Interests = nil
		m.Topics = nil
		return
	}

	totalChapters := len(m.ChaptersList)

	// Time markers: append-order, dedup.
	tmSeen := map[string]bool{}
	tms := make([]string, 0, totalChapters)
	for _, ch := range m.ChaptersList {
		tm := strings.TrimSpace(ch.TimeMarker)
		if tm == "" || tmSeen[tm] {
			continue
		}
		tmSeen[tm] = true
		tms = append(tms, tm)
	}
	m.TimeMarkers = tms

	// Assets: dedup, union across chapters. Sorted alphabetically.
	assetSeen := map[string]string{}
	for _, ch := range m.ChaptersList {
		for _, a := range ch.Assets {
			a = strings.TrimSpace(a)
			if a == "" {
				continue
			}
			key := strings.ToLower(a)
			if _, ok := assetSeen[key]; !ok {
				assetSeen[key] = a
			}
		}
	}
	assets := make([]string, 0, len(assetSeen))
	for _, v := range assetSeen {
		assets = append(assets, v)
	}
	sort.Strings(assets)
	m.Assets = assets

	m.Interests = aggregateWeighted(m.ChaptersList, totalChapters, func(ch Chapter) []string { return ch.Interests })
	m.Topics = aggregateWeighted(m.ChaptersList, totalChapters, func(ch Chapter) []string { return ch.Topics })
}

// aggregateWeighted produces a WeightedEntry slice for either interests
// or topics. The selector returns the relevant slice from a chapter.
// Entries whose first occurrence appears in chapter K are weighted by
// (chapters_mentioning / total_chapters), saturated at 1.0, floored at
// 0.1, rounded to 2 decimal places to keep on-disk output stable.
func aggregateWeighted(chapters []Chapter, totalChapters int, sel func(Chapter) []string) []WeightedEntry {
	if totalChapters == 0 {
		return nil
	}
	mentions := map[string]int{}
	display := map[string]string{}
	for _, ch := range chapters {
		seenInChapter := map[string]bool{}
		for _, name := range sel(ch) {
			name = strings.TrimSpace(name)
			if name == "" {
				continue
			}
			key := strings.ToLower(name)
			if seenInChapter[key] {
				continue
			}
			seenInChapter[key] = true
			mentions[key]++
			if _, ok := display[key]; !ok {
				display[key] = name
			}
		}
	}
	if len(mentions) == 0 {
		return nil
	}
	out := make([]WeightedEntry, 0, len(mentions))
	for key, count := range mentions {
		w := float64(count) / float64(totalChapters)
		if w > 1.0 {
			w = 1.0
		}
		if w < 0.1 {
			w = 0.1
		}
		w = roundTo2(w)
		out = append(out, WeightedEntry{Name: display[key], Weight: w})
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].Weight != out[j].Weight {
			return out[i].Weight > out[j].Weight
		}
		return strings.ToLower(out[i].Name) < strings.ToLower(out[j].Name)
	})
	return out
}

// roundTo2 rounds a float to two decimal places. Keeps the on-disk
// representation deterministic (avoid 0.6666666 vs 0.667 drift between
// runs) without introducing dependency on big.Float.
func roundTo2(f float64) float64 {
	return float64(int(f*100+0.5)) / 100.0
}

// ---------------------------------------------------------------------------
// Chapter rendering
// ---------------------------------------------------------------------------

// renderChapters serialises a chapter list as the canonical body string
// — fenceless Markdown with `## Chapter N — date — title` headings and
// per-chapter metadata + What/Why subsections.
//
// Empty chapter metadata fields are omitted to keep the file readable;
// the aggregate index in frontmatter is the source of truth for
// machine-driven lookup.
func renderChapters(chapters []Chapter) string {
	if len(chapters) == 0 {
		return ""
	}
	var b strings.Builder
	for i, ch := range chapters {
		if i > 0 {
			b.WriteString("\n")
		}
		date := ""
		if !ch.Date.IsZero() {
			date = ch.Date.UTC().Format("2006-01-02")
		}
		fmt.Fprintf(&b, "## Chapter %d", ch.Index)
		if date != "" {
			fmt.Fprintf(&b, " — %s", date)
		}
		if strings.TrimSpace(ch.Title) != "" {
			fmt.Fprintf(&b, " — %s", strings.TrimSpace(ch.Title))
		}
		b.WriteString("\n")

		if strings.TrimSpace(ch.TimeMarker) != "" {
			fmt.Fprintf(&b, "**Time marker:** %s\n", strings.TrimSpace(ch.TimeMarker))
		}
		if len(ch.Assets) > 0 {
			fmt.Fprintf(&b, "**Assets introduced:** %s\n", strings.Join(trimEach(ch.Assets), ", "))
		}
		if len(ch.Interests) > 0 {
			fmt.Fprintf(&b, "**Interests:** %s\n", strings.Join(trimEach(ch.Interests), ", "))
		}
		if len(ch.Topics) > 0 {
			fmt.Fprintf(&b, "**Topics:** %s\n", strings.Join(trimEach(ch.Topics), ", "))
		}

		b.WriteString("\n### What\n")
		b.WriteString(strings.TrimSpace(ch.What))
		b.WriteString("\n\n### Why\n")
		b.WriteString(strings.TrimSpace(ch.Why))
		b.WriteString("\n")
	}
	return b.String()
}

func trimEach(items []string) []string {
	out := make([]string, 0, len(items))
	for _, s := range items {
		s = strings.TrimSpace(s)
		if s != "" {
			out = append(out, s)
		}
	}
	return out
}

// ---------------------------------------------------------------------------
// Chapter parsing (body → []Chapter)
// ---------------------------------------------------------------------------

var chapterHeadingRe = regexp.MustCompile(`^##\s+Chapter\s+(\d+)(?:\s*—\s*(.*))?$`)

// parseChapters scans a body string and returns the structured chapter
// list. Tolerant of whitespace and missing optional metadata, but strict
// about chapter heading format and presence of `### What` / `### Why`.
//
// Returns the slice plus an error if any chapter is missing a required
// section. The slice is returned even on error so callers can show a
// best-effort view in diagnostic UIs.
func parseChapters(body string) ([]Chapter, error) {
	lines := strings.Split(body, "\n")

	// First, identify chapter heading line indices.
	headings := []int{}
	for i, line := range lines {
		if chapterHeadingRe.MatchString(strings.TrimRight(line, " \t")) {
			headings = append(headings, i)
		}
	}
	if len(headings) == 0 {
		return nil, errors.New("no chapter headings found")
	}

	chapters := make([]Chapter, 0, len(headings))
	for k, start := range headings {
		end := len(lines)
		if k+1 < len(headings) {
			end = headings[k+1]
		}
		ch, err := parseOneChapter(lines[start:end])
		if err != nil {
			return chapters, fmt.Errorf("chapter %d: %w", k+1, err)
		}
		chapters = append(chapters, ch)
	}
	return chapters, nil
}

func parseOneChapter(lines []string) (Chapter, error) {
	var ch Chapter
	if len(lines) == 0 {
		return ch, errors.New("empty chapter")
	}
	heading := strings.TrimRight(lines[0], " \t")
	matches := chapterHeadingRe.FindStringSubmatch(heading)
	if matches == nil {
		return ch, fmt.Errorf("malformed chapter heading: %q", heading)
	}
	if _, err := fmt.Sscanf(matches[1], "%d", &ch.Index); err != nil {
		return ch, fmt.Errorf("invalid chapter index in heading: %q", heading)
	}
	rest := matches[2]
	// rest is "<date> — <title>" or just "<date>" or just "<title>" or empty.
	parts := splitEmDash(rest, 2)
	for _, part := range parts {
		part = strings.TrimSpace(part)
		if part == "" {
			continue
		}
		if t, err := time.Parse("2006-01-02", part); err == nil && ch.Date.IsZero() {
			ch.Date = t
			continue
		}
		if ch.Title == "" {
			ch.Title = part
		} else {
			ch.Title = ch.Title + " — " + part
		}
	}

	// Parse remaining body: metadata lines, then ### What / ### Why.
	mode := "preamble" // preamble | what | why
	var what, why strings.Builder
	for _, line := range lines[1:] {
		trimmed := strings.TrimSpace(line)
		switch {
		case trimmed == "### What":
			mode = "what"
			continue
		case trimmed == "### Why":
			mode = "why"
			continue
		}
		if mode == "preamble" {
			if strings.HasPrefix(trimmed, "**") {
				parseChapterMetaLine(trimmed, &ch)
			}
			continue
		}
		if mode == "what" {
			what.WriteString(line)
			what.WriteString("\n")
		} else if mode == "why" {
			why.WriteString(line)
			why.WriteString("\n")
		}
	}
	ch.What = strings.TrimSpace(what.String())
	ch.Why = strings.TrimSpace(why.String())
	if ch.What == "" {
		return ch, errors.New("missing or empty `### What`")
	}
	if ch.Why == "" {
		return ch, errors.New("missing or empty `### Why`")
	}
	return ch, nil
}

// splitEmDash splits on " — " (em-dash with surrounding spaces) up to n
// pieces. Falls back to plain " - " (hyphen with spaces) when the em-dash
// is absent — the chapter heading format uses em-dash but copy-paste from
// terminals can mangle it.
func splitEmDash(s string, n int) []string {
	if strings.Contains(s, " — ") {
		return strings.SplitN(s, " — ", n)
	}
	if strings.Contains(s, " - ") {
		return strings.SplitN(s, " - ", n)
	}
	return []string{s}
}

// parseChapterMetaLine parses lines like "**Time marker:** 2026-04-12"
// into chapter fields. Unknown labels are ignored.
func parseChapterMetaLine(line string, ch *Chapter) {
	if !strings.HasPrefix(line, "**") {
		return
	}
	closeIdx := strings.Index(line[2:], ":**")
	if closeIdx < 0 {
		return
	}
	label := strings.TrimSpace(line[2 : 2+closeIdx])
	value := strings.TrimSpace(line[2+closeIdx+3:])
	if value == "" {
		return
	}
	switch strings.ToLower(label) {
	case "time marker":
		ch.TimeMarker = value
	case "assets introduced", "assets":
		ch.Assets = splitCSV(value)
	case "interests":
		ch.Interests = splitCSV(value)
	case "topics":
		ch.Topics = splitCSV(value)
	}
}

func splitCSV(s string) []string {
	parts := strings.Split(s, ",")
	out := make([]string, 0, len(parts))
	for _, p := range parts {
		p = strings.TrimSpace(p)
		if p != "" {
			out = append(out, p)
		}
	}
	return out
}
