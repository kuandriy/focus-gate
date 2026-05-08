package memory

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// fakeVocab returns a VocabSnapshot whose Vectorize counts each space-
// separated token with weight 1.0. Sufficient for tests that need to
// verify the storage round-trip without coupling to the real TF-IDF
// engine.
func fakeVocab(hash string) VocabSnapshot {
	return VocabSnapshot{
		Hash: hash,
		Vectorize: func(text string) map[string]float64 {
			out := map[string]float64{}
			for _, t := range strings.Fields(text) {
				if len(t) >= 3 {
					out[strings.ToLower(t)]++
				}
			}
			return out
		},
	}
}

// chapterFixture returns a minimal valid chapter so tests don't repeat
// the boilerplate. Date defaults to 2026-04-01 — fixed so reproducible.
func chapterFixture(title string) Chapter {
	return Chapter{
		Date:       time.Date(2026, 4, 1, 0, 0, 0, 0, time.UTC),
		Title:      title,
		TimeMarker: "2026-04-01",
		What:       "Did the thing.",
		Why:        "Because we needed to.",
	}
}

func TestValidate_RequiresAtLeastOneChapter(t *testing.T) {
	m := &Memory{Title: "x"}
	if err := m.Validate(); err == nil {
		t.Error("expected validation to fail with zero chapters")
	}
}

func TestValidate_ChapterMissingWhat(t *testing.T) {
	ch := chapterFixture("c1")
	ch.What = ""
	m := &Memory{Title: "x", ChaptersList: []Chapter{ch}}
	if err := m.Validate(); err == nil {
		t.Error("expected validation to fail when chapter has empty What")
	}
}

func TestValidate_ChapterMissingWhy(t *testing.T) {
	ch := chapterFixture("c1")
	ch.Why = ""
	m := &Memory{Title: "x", ChaptersList: []Chapter{ch}}
	if err := m.Validate(); err == nil {
		t.Error("expected validation to fail when chapter has empty Why")
	}
}

func TestValidate_TitleConstraints(t *testing.T) {
	m := &Memory{ChaptersList: []Chapter{chapterFixture("c1")}}
	if err := m.Validate(); err == nil {
		t.Error("expected validation to fail on empty title")
	}
	m.Title = strings.Repeat("a", 81)
	if err := m.Validate(); err == nil {
		t.Error("expected validation to fail on >80-char title")
	}
}

func TestAppendChapter_BumpsVersion(t *testing.T) {
	m := &Memory{Title: "x"}
	if err := AppendChapter(m, chapterFixture("c1")); err != nil {
		t.Fatal(err)
	}
	if m.Version != 1 || m.Chapters != 1 {
		t.Errorf("after first append: version=%d chapters=%d, want 1/1", m.Version, m.Chapters)
	}
	if m.ChaptersList[0].Index != 1 {
		t.Errorf("first chapter index = %d, want 1", m.ChaptersList[0].Index)
	}

	if err := AppendChapter(m, chapterFixture("c2")); err != nil {
		t.Fatal(err)
	}
	if m.Version != 2 || m.Chapters != 2 {
		t.Errorf("after second append: version=%d chapters=%d, want 2/2", m.Version, m.Chapters)
	}
	if m.ChaptersList[1].Index != 2 {
		t.Errorf("second chapter index = %d, want 2", m.ChaptersList[1].Index)
	}
}

func TestAppendChapter_RejectsEmptyWhatOrWhy(t *testing.T) {
	m := &Memory{Title: "x"}
	bad := chapterFixture("c1")
	bad.What = "  "
	if err := AppendChapter(m, bad); err == nil {
		t.Error("expected error for empty What")
	}
	bad = chapterFixture("c1")
	bad.Why = "  "
	if err := AppendChapter(m, bad); err == nil {
		t.Error("expected error for empty Why")
	}
}

func TestAggregate_WeightsByChapterCoverage(t *testing.T) {
	c1 := chapterFixture("c1")
	c1.Interests = []string{"shared", "only-one"}
	c1.Topics = []string{"shared topic"}
	c1.Assets = []string{"a.go", "b.go"}

	c2 := chapterFixture("c2")
	c2.Interests = []string{"shared"}
	c2.Topics = []string{"shared topic", "only-two"}
	c2.Assets = []string{"b.go", "c.go"}

	m := &Memory{Title: "x", ChaptersList: []Chapter{c1, c2}}
	aggregateFromChapters(m)

	// "shared" appears in 2/2 chapters → weight 1.0.
	// "only-one" appears in 1/2 → 0.5.
	// "shared topic" → 1.0; "only-two" → 0.5.
	wantInterest := map[string]float64{"shared": 1.0, "only-one": 0.5}
	for _, in := range m.Interests {
		want, ok := wantInterest[strings.ToLower(in.Name)]
		if !ok {
			t.Errorf("unexpected interest %q", in.Name)
			continue
		}
		if in.Weight != want {
			t.Errorf("interest %q weight = %.2f, want %.2f", in.Name, in.Weight, want)
		}
	}
	if len(m.Interests) != 2 {
		t.Errorf("got %d interests, want 2", len(m.Interests))
	}
	if len(m.Topics) != 2 {
		t.Errorf("got %d topics, want 2", len(m.Topics))
	}

	// Assets are union, sorted alphabetically.
	wantAssets := []string{"a.go", "b.go", "c.go"}
	if len(m.Assets) != len(wantAssets) {
		t.Fatalf("got %d assets, want %d", len(m.Assets), len(wantAssets))
	}
	for i, a := range wantAssets {
		if m.Assets[i] != a {
			t.Errorf("asset[%d] = %q, want %q", i, m.Assets[i], a)
		}
	}
}

func TestWriteRead_RoundTrip(t *testing.T) {
	dir := t.TempDir()
	vocab := fakeVocab("vh1")
	original := &Memory{Title: "Auth & session model"}
	ch := Chapter{
		Date:       time.Date(2026, 3, 22, 0, 0, 0, 0, time.UTC),
		Title:      "Initial design",
		TimeMarker: "2026-03-15..2026-03-22",
		Assets:     []string{"cmd/api/auth.go", "internal/session/store.go"},
		Interests:  []string{"session lifecycle", "RS256"},
		Topics:     []string{"JWT authentication", "session model"},
		What:       "Used JWT with RS256, 15-minute access tokens.",
		Why:        "Multi-service rotation needs asymmetric keys.",
	}
	if err := AppendChapter(original, ch); err != nil {
		t.Fatal(err)
	}
	if err := WriteFile(dir, original, vocab); err != nil {
		t.Fatalf("WriteFile failed: %v", err)
	}
	if original.ID == "" {
		t.Fatal("WriteFile should have stamped an ID")
	}
	if original.Created.IsZero() || original.Updated.IsZero() {
		t.Fatal("WriteFile should have stamped Created and Updated")
	}

	loaded, err := ReadFile(original.Path(dir))
	if err != nil {
		t.Fatalf("ReadFile failed: %v", err)
	}
	if loaded.ID != original.ID {
		t.Errorf("ID mismatch: got %q, want %q", loaded.ID, original.ID)
	}
	if loaded.Title != original.Title {
		t.Errorf("title mismatch: got %q, want %q", loaded.Title, original.Title)
	}
	if loaded.VocabHash != "vh1" {
		t.Errorf("vocab hash not persisted: got %q", loaded.VocabHash)
	}
	if len(loaded.Fingerprint) == 0 {
		t.Error("fingerprint should be populated after WriteFile")
	}
	if len(loaded.ChaptersList) != 1 {
		t.Fatalf("expected 1 chapter, got %d", len(loaded.ChaptersList))
	}
	got := loaded.ChaptersList[0]
	if got.Title != "Initial design" {
		t.Errorf("chapter title = %q, want %q", got.Title, "Initial design")
	}
	if got.TimeMarker != "2026-03-15..2026-03-22" {
		t.Errorf("chapter time marker = %q", got.TimeMarker)
	}
	if !strings.Contains(got.What, "JWT with RS256") {
		t.Errorf("chapter What = %q", got.What)
	}
	if !strings.Contains(got.Why, "asymmetric") {
		t.Errorf("chapter Why = %q", got.Why)
	}
	if len(got.Assets) != 2 {
		t.Errorf("chapter assets = %v", got.Assets)
	}
	if loaded.Version != 1 || loaded.Chapters != 1 {
		t.Errorf("version=%d chapters=%d, want 1/1", loaded.Version, loaded.Chapters)
	}
	if len(loaded.Interests) != 2 || len(loaded.Topics) != 2 || len(loaded.Assets) != 2 {
		t.Errorf("aggregate index lengths off: interests=%d topics=%d assets=%d",
			len(loaded.Interests), len(loaded.Topics), len(loaded.Assets))
	}
}

func TestWriteRead_TwoChapterStoryAggregates(t *testing.T) {
	dir := t.TempDir()
	vocab := fakeVocab("vh1")
	m := &Memory{Title: "Auth & session model"}
	if err := AppendChapter(m, Chapter{
		Date:       time.Date(2026, 3, 22, 0, 0, 0, 0, time.UTC),
		Title:      "Initial design",
		TimeMarker: "2026-03-22",
		Assets:     []string{"cmd/api/auth.go"},
		Interests:  []string{"shared", "first-only"},
		Topics:     []string{"shared topic"},
		What:       "first-what",
		Why:        "first-why",
	}); err != nil {
		t.Fatal(err)
	}
	if err := AppendChapter(m, Chapter{
		Date:       time.Date(2026, 4, 12, 0, 0, 0, 0, time.UTC),
		Title:      "Refresh token rotation",
		TimeMarker: "2026-04-12",
		Assets:     []string{"middleware/refresh.go"},
		Interests:  []string{"shared"},
		Topics:     []string{"shared topic", "second-only"},
		What:       "second-what",
		Why:        "second-why",
	}); err != nil {
		t.Fatal(err)
	}
	if err := WriteFile(dir, m, vocab); err != nil {
		t.Fatal(err)
	}
	loaded, err := ReadFile(m.Path(dir))
	if err != nil {
		t.Fatal(err)
	}
	if len(loaded.ChaptersList) != 2 {
		t.Fatalf("got %d chapters, want 2", len(loaded.ChaptersList))
	}
	if len(loaded.TimeMarkers) != 2 {
		t.Errorf("got %d time markers, want 2", len(loaded.TimeMarkers))
	}
	// Updated should mirror the latest chapter date (2026-04-12).
	if loaded.Updated.Format("2006-01-02") != "2026-04-12" {
		t.Errorf("updated date = %s, want 2026-04-12", loaded.Updated.Format("2006-01-02"))
	}
}

func TestNewID_Format(t *testing.T) {
	now := time.Date(2026, 4, 22, 14, 1, 2, 0, time.UTC)
	id := NewID(now)
	if !strings.HasPrefix(id, "mem_20260422_") {
		t.Errorf("ID prefix wrong: %q", id)
	}
	if len(id) != len("mem_20260422_")+6 {
		t.Errorf("ID hex suffix wrong length: %q", id)
	}
}

// Manifests written by pre-fix binaries serialized WeightedEntry with
// Go's default capitalization ({"Name":..., "Weight":...}). New
// binaries must still parse those files (e.g. seed-memories/index.json
// shipped with this repo) — regression coverage for the
// UnmarshalJSON compat shim.
func TestWeightedEntry_AcceptsBothLowercaseAndLegacyJSON(t *testing.T) {
	cases := []struct {
		name string
		raw  string
		want WeightedEntry
	}{
		{
			name: "lowercase (current)",
			raw:  `{"name":"jwt","weight":0.5}`,
			want: WeightedEntry{Name: "jwt", Weight: 0.5},
		},
		{
			name: "legacy capitalized",
			raw:  `{"Name":"jwt","Weight":0.5}`,
			want: WeightedEntry{Name: "jwt", Weight: 0.5},
		},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			var got WeightedEntry
			if err := json.Unmarshal([]byte(c.raw), &got); err != nil {
				t.Fatalf("unmarshal: %v", err)
			}
			if got != c.want {
				t.Errorf("got %+v, want %+v", got, c.want)
			}
		})
	}
}

func TestParseList_FlowStyle(t *testing.T) {
	got := parseList(`["a", "b c", "d"]`)
	want := []string{"a", "b c", "d"}
	if len(got) != len(want) {
		t.Fatalf("got %v, want %v", got, want)
	}
	for i := range got {
		if got[i] != want[i] {
			t.Errorf("[%d] got %q want %q", i, got[i], want[i])
		}
	}
	if parseList("not a list") != nil {
		t.Error("non-list value should return nil")
	}
}

func TestParseWeightedList(t *testing.T) {
	got := parseWeightedList(`["foo@0.50", "bar@1.00", "baz"]`)
	if len(got) != 3 {
		t.Fatalf("got %d entries, want 3: %v", len(got), got)
	}
	want := []WeightedEntry{
		{Name: "foo", Weight: 0.5},
		{Name: "bar", Weight: 1.0},
		{Name: "baz", Weight: 1.0}, // missing weight → defaults to 1.0
	}
	for i, w := range want {
		if got[i] != w {
			t.Errorf("[%d] got %+v, want %+v", i, got[i], w)
		}
	}
}

func TestParseWeightMap(t *testing.T) {
	got := parseWeightMap("jwt:0.4821 session:0.3654")
	if len(got) != 2 || got["jwt"] != 0.4821 || got["session"] != 0.3654 {
		t.Errorf("weight map round-trip failed: %v", got)
	}
}

func TestManifest_LoadEmpty(t *testing.T) {
	dir := t.TempDir()
	mf, err := Load(dir)
	if err != nil {
		t.Fatalf("Load on missing manifest should not error: %v", err)
	}
	if mf == nil || mf.Entries == nil && len(mf.Entries) != 0 {
		t.Error("Load on missing manifest should return empty manifest")
	}
	if mf.SchemaVersion != SchemaVersion {
		t.Errorf("schema version mismatch: got %q, want %q", mf.SchemaVersion, SchemaVersion)
	}
}

// TestManifest_LoadCorruptReturnsNonNil guards the contract that callers
// (slash_memory.go's commit/forget/migrate-v1 post-writes) rely on:
// even when the file on disk is unreadable, Load returns a usable empty
// manifest plus the error, never nil. Without this, a corrupted index
// would crash the slash path during the post-write rebuild.
func TestManifest_LoadCorruptReturnsNonNil(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(IndexPath(dir), []byte("{not json"), 0644); err != nil {
		t.Fatal(err)
	}
	mf, err := Load(dir)
	if err == nil {
		t.Error("expected non-nil error for corrupt manifest")
	}
	if mf == nil {
		t.Fatal("Load must return non-nil manifest even on parse error")
	}
	if mf.SchemaVersion != SchemaVersion {
		t.Errorf("expected fresh empty manifest, got schema=%q", mf.SchemaVersion)
	}
}

func TestManifest_UpsertAndTouch(t *testing.T) {
	dir := t.TempDir()
	mf := NewManifest()
	mf.Upsert(IndexEntry{ID: "mem_x", Title: "Test", Path: "mem_x.md"})
	if !mf.Dirty() {
		t.Error("upsert should mark manifest dirty")
	}
	if err := mf.Save(dir); err != nil {
		t.Fatal(err)
	}
	if mf.Dirty() {
		t.Error("save should clear dirty flag")
	}

	mf.Touch("mem_x")
	mf.Touch("mem_x")
	got, _ := mf.Get("mem_x")
	if got.TouchedBy != 2 {
		t.Errorf("touch counter = %d, want 2", got.TouchedBy)
	}
	if !mf.Dirty() {
		t.Error("touch should mark manifest dirty")
	}
}

func TestManifest_Rebuild(t *testing.T) {
	dir := t.TempDir()
	vocab := fakeVocab("vh1")

	for _, title := range []string{"Auth model", "Test conventions"} {
		m := &Memory{Title: title}
		if err := AppendChapter(m, chapterFixture("c1")); err != nil {
			t.Fatal(err)
		}
		if err := WriteFile(dir, m, vocab); err != nil {
			t.Fatal(err)
		}
	}

	mf := NewManifest()
	errs := mf.Rebuild(dir, vocab)
	if len(errs) > 0 {
		t.Errorf("rebuild errors: %v", errs)
	}
	if len(mf.Entries) != 2 {
		t.Errorf("expected 2 entries after rebuild, got %d", len(mf.Entries))
	}

	// Drop a malformed file and confirm rebuild reports it but continues.
	if err := os.WriteFile(filepath.Join(dir, "bad.md"), []byte("not valid frontmatter"), 0644); err != nil {
		t.Fatal(err)
	}
	mf2 := NewManifest()
	errs = mf2.Rebuild(dir, vocab)
	if len(errs) == 0 {
		t.Error("expected rebuild to report the broken file")
	}
	if len(mf2.Entries) != 2 {
		t.Errorf("rebuild should ignore bad files, got %d entries", len(mf2.Entries))
	}
}

// Rebuild must populate IndexEntry.LatestSnippet from the latest
// chapter's What so Surface can render it inline. Truncated at the
// first sentence boundary; multi-line What is collapsed to one line.
func TestManifest_RebuildPopulatesLatestSnippet(t *testing.T) {
	dir := t.TempDir()
	vocab := fakeVocab("vh1")

	m := &Memory{Title: "Auth"}
	if err := AppendChapter(m, Chapter{
		Date:       time.Date(2026, 4, 1, 0, 0, 0, 0, time.UTC),
		TimeMarker: "2026-04-01",
		What:       "Used JWT with RS256.\nAdded 15-minute access tokens.",
		Why:        "Multi-service rotation.",
	}); err != nil {
		t.Fatal(err)
	}
	if err := AppendChapter(m, Chapter{
		Date:       time.Date(2026, 4, 12, 0, 0, 0, 0, time.UTC),
		TimeMarker: "2026-04-12",
		What:       "Rotated refresh tokens every 24h. Single-use enforced.",
		Why:        "Limit blast radius.",
	}); err != nil {
		t.Fatal(err)
	}
	if err := WriteFile(dir, m, vocab); err != nil {
		t.Fatal(err)
	}

	mf := NewManifest()
	if errs := mf.Rebuild(dir, vocab); len(errs) > 0 {
		t.Fatalf("rebuild: %v", errs)
	}
	got, ok := mf.Get(m.ID)
	if !ok {
		t.Fatal("manifest missing the freshly-written memory")
	}
	if got.LatestSnippet == "" {
		t.Error("expected LatestSnippet populated from latest chapter's What")
	}
	if !strings.Contains(got.LatestSnippet, "Rotated refresh tokens every 24h") {
		t.Errorf("snippet should be from the LATEST chapter, got %q", got.LatestSnippet)
	}
	if strings.Contains(got.LatestSnippet, "Single-use enforced") {
		t.Errorf("snippet should stop at the first sentence boundary, got %q", got.LatestSnippet)
	}
}

func TestMigrateV1ToV2_RoundTrip(t *testing.T) {
	dir := t.TempDir()
	v1Body := `---
schemaVersion: "1"
id: "mem_20260101_abc123"
title: "Auth & session model"
sources: []
refs: ["cmd/api/auth.go", "internal/session/store.go"]
created: "2026-01-01T00:00:00Z"
updated: "2026-01-15T00:00:00Z"
topTerms: []
fingerprint: ""
vocabHash: ""
touchedBy: 5
---

## What we did
Used JWT with RS256, 15-minute access tokens.

## Why
Multi-service rotation needs asymmetric keys.
`
	path := filepath.Join(dir, "mem_20260101_abc123.md")
	if err := os.WriteFile(path, []byte(v1Body), 0644); err != nil {
		t.Fatal(err)
	}

	vocab := fakeVocab("vh1")
	m, err := MigrateV1FileToV2(path, vocab)
	if err != nil {
		t.Fatalf("migration failed: %v", err)
	}
	if m.ID != "mem_20260101_abc123" {
		t.Errorf("ID changed: got %q", m.ID)
	}
	if m.TouchedBy != 5 {
		t.Errorf("TouchedBy lost: got %d, want 5", m.TouchedBy)
	}
	if len(m.ChaptersList) != 1 {
		t.Fatalf("expected 1 chapter, got %d", len(m.ChaptersList))
	}
	ch := m.ChaptersList[0]
	if !strings.Contains(ch.What, "JWT with RS256") {
		t.Errorf("What body lost: %q", ch.What)
	}
	if !strings.Contains(ch.Why, "asymmetric") {
		t.Errorf("Why body lost: %q", ch.Why)
	}
	if len(ch.Assets) != 2 {
		t.Errorf("assets lost: %v", ch.Assets)
	}

	// Backup file present.
	if _, err := os.Stat(path + ".v1.bak"); err != nil {
		t.Errorf("expected backup at %s.v1.bak", path)
	}

	// File on disk now parseable as v2.
	loaded, err := ReadFile(path)
	if err != nil {
		t.Fatalf("v2 file unreadable: %v", err)
	}
	if len(loaded.ChaptersList) != 1 {
		t.Errorf("v2-parsed chapters = %d, want 1", len(loaded.ChaptersList))
	}
}

func TestMigrateV1ToV2_NoOpOnV2(t *testing.T) {
	dir := t.TempDir()
	vocab := fakeVocab("vh1")
	m := &Memory{Title: "Already v2"}
	if err := AppendChapter(m, chapterFixture("c1")); err != nil {
		t.Fatal(err)
	}
	if err := WriteFile(dir, m, vocab); err != nil {
		t.Fatal(err)
	}
	if _, err := MigrateV1FileToV2(m.Path(dir), vocab); err != nil {
		t.Errorf("migrate on v2 file should be a no-op, got %v", err)
	}
	// No backup should be written for v2 files.
	if _, err := os.Stat(m.Path(dir) + ".v1.bak"); err == nil {
		t.Error("backup created for v2 file (should have been a no-op)")
	}
}
