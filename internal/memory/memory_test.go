package memory

import (
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

func TestValidate_RequiredSections(t *testing.T) {
	m := &Memory{
		Title: "x",
		Body:  "## What we did\nimplemented x\n## Why\nbecause y\n",
	}
	if err := m.Validate(); err != nil {
		t.Errorf("expected validation to pass, got %v", err)
	}

	m.Body = "## What we did\nimplemented x\n"
	if err := m.Validate(); err == nil {
		t.Error("expected validation to fail when ## Why missing")
	}

	m.Body = "## What we did\n\n## Why\nbecause y\n"
	if err := m.Validate(); err == nil {
		t.Error("expected validation to fail when ## What we did is empty")
	}
}

func TestValidate_TitleConstraints(t *testing.T) {
	m := &Memory{
		Body: "## What we did\nx\n## Why\ny\n",
	}
	if err := m.Validate(); err == nil {
		t.Error("expected validation to fail on empty title")
	}
	m.Title = strings.Repeat("a", 81)
	if err := m.Validate(); err == nil {
		t.Error("expected validation to fail on >80-char title")
	}
}

func TestWriteRead_RoundTrip(t *testing.T) {
	dir := t.TempDir()
	vocab := fakeVocab("vh1")
	original := &Memory{
		Title:   "Auth & session model",
		Sources: []string{"tree_b21f"},
		Refs:    []string{"cmd/api/auth.go"},
		Body: "## What we did\n" +
			"Used JWT with RS256, 15m access tokens.\n" +
			"## Why\nMulti-service rotation needs asymmetric keys.\n",
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
	if !strings.Contains(loaded.Body, "## What we did") {
		t.Errorf("body missing required section: %q", loaded.Body)
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

	// Write two memories.
	for _, title := range []string{"Auth model", "Test conventions"} {
		m := &Memory{
			Title: title,
			Body:  "## What we did\n" + title + "\n## Why\nbecause\n",
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
