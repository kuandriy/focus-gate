package memory

import (
	"os"
	"path/filepath"
	"testing"
)

// sourcesFixture sets up two on-disk memory directories (one for
// personal, one for a hypothetical "team" shared source) so attach/
// detach paths can be exercised end-to-end.
func sourcesFixture(t *testing.T) (string, string, string) {
	t.Helper()
	dataDir := t.TempDir()
	personalDir := filepath.Join(dataDir, "memories")
	teamDir := t.TempDir()
	if err := os.MkdirAll(personalDir, 0755); err != nil {
		t.Fatal(err)
	}
	return dataDir, personalDir, teamDir
}

func TestLoadSources_SynthesizesPersonalWhenFileMissing(t *testing.T) {
	dataDir, personalDir, _ := sourcesFixture(t)
	r, err := LoadSources(dataDir, personalDir)
	if err != nil {
		t.Fatal(err)
	}
	if r.Default != DefaultSourceName {
		t.Errorf("default = %q, want %q", r.Default, DefaultSourceName)
	}
	personal, ok := r.Get(DefaultSourceName)
	if !ok {
		t.Fatal("personal source missing")
	}
	if personal.Path != personalDir {
		t.Errorf("personal.path = %q, want %q", personal.Path, personalDir)
	}
	if !personal.Enabled || !personal.Writable {
		t.Errorf("personal flags off: enabled=%v writable=%v", personal.Enabled, personal.Writable)
	}
}

func TestSources_AttachAndDetach(t *testing.T) {
	dataDir, personalDir, teamDir := sourcesFixture(t)
	r, _ := LoadSources(dataDir, personalDir)
	if err := r.Attach("team", teamDir, false); err != nil {
		t.Fatalf("attach: %v", err)
	}
	if got, ok := r.Get("team"); !ok || got.Writable {
		t.Errorf("attached team source not as expected: ok=%v got=%+v", ok, got)
	}

	// Detach removes it; detaching personal is rejected.
	if err := r.Detach("team"); err != nil {
		t.Fatalf("detach: %v", err)
	}
	if _, ok := r.Get("team"); ok {
		t.Error("team still attached after detach")
	}
	if err := r.Detach(DefaultSourceName); err == nil {
		t.Error("detaching personal should be refused")
	}
}

func TestSources_AttachRejectsDuplicateAndMissingPath(t *testing.T) {
	dataDir, personalDir, teamDir := sourcesFixture(t)
	r, _ := LoadSources(dataDir, personalDir)

	if err := r.Attach("team", teamDir, true); err != nil {
		t.Fatal(err)
	}
	if err := r.Attach("team", teamDir, true); err == nil {
		t.Error("expected duplicate-name error")
	}
	if err := r.Attach("missing", "/path/does/not/exist", true); err == nil {
		t.Error("expected missing-path error")
	}
}

func TestSources_EnableDisable(t *testing.T) {
	dataDir, personalDir, teamDir := sourcesFixture(t)
	r, _ := LoadSources(dataDir, personalDir)
	if err := r.Attach("team", teamDir, true); err != nil {
		t.Fatal(err)
	}
	if err := r.Disable("team"); err != nil {
		t.Fatal(err)
	}
	got, _ := r.Get("team")
	if got.Enabled {
		t.Error("expected disabled")
	}
	if err := r.Enable("team"); err != nil {
		t.Fatal(err)
	}
	got, _ = r.Get("team")
	if !got.Enabled {
		t.Error("expected enabled")
	}
}

func TestSources_SetDefaultRejectsNonWritable(t *testing.T) {
	dataDir, personalDir, teamDir := sourcesFixture(t)
	r, _ := LoadSources(dataDir, personalDir)
	if err := r.Attach("team", teamDir, false); err != nil {
		t.Fatal(err)
	}
	if err := r.SetDefault("team"); err == nil {
		t.Error("expected error setting read-only source as default")
	}
	if err := r.SetDefault("nope"); err == nil {
		t.Error("expected error for unknown source")
	}
	if err := r.SetDefault(DefaultSourceName); err != nil {
		t.Errorf("default → personal should succeed, got %v", err)
	}
}

func TestSources_SaveRoundTrip(t *testing.T) {
	dataDir, personalDir, teamDir := sourcesFixture(t)
	r, _ := LoadSources(dataDir, personalDir)
	if err := r.Attach("team", teamDir, true); err != nil {
		t.Fatal(err)
	}
	if err := r.Save(dataDir); err != nil {
		t.Fatal(err)
	}

	r2, err := LoadSources(dataDir, personalDir)
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := r2.Get("team"); !ok {
		t.Error("round-trip lost team source")
	}
	if _, ok := r2.Get(DefaultSourceName); !ok {
		t.Error("round-trip lost personal source")
	}
}

func TestSources_LoadEnabledManifestsTagsEntries(t *testing.T) {
	dataDir, personalDir, teamDir := sourcesFixture(t)
	vocab := fakeVocab("vh1")

	// Write one memory in personal, one in team.
	personal := &Memory{Title: "Personal note"}
	_ = AppendChapter(personal, chapterFixture("c1"))
	_ = WriteFile(personalDir, personal, vocab)

	team := &Memory{Title: "Team note"}
	_ = AppendChapter(team, chapterFixture("c1"))
	_ = WriteFile(teamDir, team, vocab)

	r, _ := LoadSources(dataDir, personalDir)
	if err := r.Attach("team", teamDir, false); err != nil {
		t.Fatal(err)
	}
	manifests, errs := r.LoadEnabledManifests(vocab)
	if len(errs) > 0 {
		t.Fatalf("load errors: %v", errs)
	}
	if len(manifests) != 2 {
		t.Fatalf("got %d manifests, want 2", len(manifests))
	}
	for _, m := range manifests {
		if m.Source == "" {
			t.Errorf("manifest missing source name")
		}
		for _, e := range m.Entries {
			if e.Source != m.Source {
				t.Errorf("entry source = %q, want %q", e.Source, m.Source)
			}
		}
	}
}

func TestSources_DisabledSourceSkippedInLoad(t *testing.T) {
	dataDir, personalDir, teamDir := sourcesFixture(t)
	vocab := fakeVocab("vh1")

	team := &Memory{Title: "Team note"}
	_ = AppendChapter(team, chapterFixture("c1"))
	_ = WriteFile(teamDir, team, vocab)

	r, _ := LoadSources(dataDir, personalDir)
	if err := r.Attach("team", teamDir, false); err != nil {
		t.Fatal(err)
	}
	if err := r.Disable("team"); err != nil {
		t.Fatal(err)
	}
	manifests, _ := r.LoadEnabledManifests(vocab)
	if len(manifests) != 1 || manifests[0].Source != DefaultSourceName {
		t.Errorf("disabled source should be skipped; got %d manifests, first source=%q",
			len(manifests),
			func() string {
				if len(manifests) == 0 {
					return "(none)"
				}
				return manifests[0].Source
			}())
	}
}

func TestSources_SourceDirsFiltersReadOnly(t *testing.T) {
	dataDir, personalDir, teamDir := sourcesFixture(t)
	r, _ := LoadSources(dataDir, personalDir)
	if err := r.Attach("team", teamDir, false); err != nil {
		t.Fatal(err)
	}
	dirs := r.SourceDirs()
	if _, ok := dirs[DefaultSourceName]; !ok {
		t.Error("personal missing from writable dirs")
	}
	if _, ok := dirs["team"]; ok {
		t.Error("read-only team should not appear in writable dirs")
	}
}
