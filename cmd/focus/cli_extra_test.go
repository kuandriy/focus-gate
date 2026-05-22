package main

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/kuandriy/focus-gate/internal/memory"
	"github.com/kuandriy/focus-gate/internal/persist"
	"github.com/kuandriy/focus-gate/internal/tfidf"
)

// setupTestPaths returns a paths struct rooted at a fresh tempdir plus
// a config that mirrors defaults, so each test gets its own isolated
// state directory without touching ~/.focus-gate/.
func setupTestPaths(t *testing.T) (paths, config) {
	t.Helper()
	dataDir := t.TempDir()
	memDir := filepath.Join(dataDir, "memories")
	if err := os.MkdirAll(memDir, 0755); err != nil {
		t.Fatal(err)
	}
	p := paths{
		dataDir:    dataDir,
		intentFile: filepath.Join(dataDir, "intent.json"),
		engineFile: filepath.Join(dataDir, "engine.json"),
		guideFile:  filepath.Join(dataDir, "guide.json"),
		lockFile:   filepath.Join(dataDir, ".lock"),
		memoryDir:  memDir,
	}
	cfg := defaultConfig()
	return p, cfg
}

// handleReset must wipe intent/engine/guide files so the next prompt
// starts from a clean slate. Idempotent — running on an already-empty
// dir is a no-op, not an error.
func TestHandleReset_RemovesStateFiles(t *testing.T) {
	p, _ := setupTestPaths(t)
	for _, fp := range []string{p.intentFile, p.engineFile, p.guideFile} {
		if err := os.WriteFile(fp, []byte(`{}`), 0644); err != nil {
			t.Fatal(err)
		}
	}
	if err := handleReset(p); err != nil {
		t.Fatalf("handleReset: %v", err)
	}
	for _, fp := range []string{p.intentFile, p.engineFile, p.guideFile} {
		if _, err := os.Stat(fp); !os.IsNotExist(err) {
			t.Errorf("expected %s removed, err=%v", fp, err)
		}
	}
	// Idempotent: running again on already-empty state should not error.
	if err := handleReset(p); err != nil {
		t.Errorf("handleReset on empty state should be a no-op, got %v", err)
	}
}

// dirSize / humanSize are leaf helpers used by --list-projects. Easy
// to test in isolation since they take no global state.
func TestDirSize_SumsRegularFiles(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "a.txt"), []byte("hello"), 0644); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Join(dir, "sub"), 0755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "sub", "b.txt"), []byte("world!"), 0644); err != nil {
		t.Fatal(err)
	}
	got := dirSize(dir)
	if got < int64(len("hello")+len("world!")) {
		t.Errorf("dirSize too low: %d", got)
	}
}

func TestHumanSize_FormatsByMagnitude(t *testing.T) {
	cases := []struct {
		in   int64
		want string
	}{
		{0, "0 B"},
		{512, "512 B"},
		{2048, "2.0 KB"},
		{int64(2.5 * 1024 * 1024), "2.5 MB"},
	}
	for _, c := range cases {
		if got := humanSize(c.in); got != c.want {
			t.Errorf("humanSize(%d) = %q, want %q", c.in, got, c.want)
		}
	}
}

// flagValue must reject the next-token-as-value when that token looks
// like another flag. Was a footgun (B-5) before the guard landed.
func TestFlagValue_RefusesNextFlagAsValue(t *testing.T) {
	got := flagValue([]string{"focus-gate", "--data-dir", "--reset"}, "--data-dir")
	if got != "" {
		t.Errorf("expected empty, got %q (next-flag-as-value bug)", got)
	}

	got = flagValue([]string{"focus-gate", "--data-dir", "/tmp/x"}, "--data-dir")
	if got != "/tmp/x" {
		t.Errorf("expected /tmp/x, got %q", got)
	}

	got = flagValue([]string{"focus-gate", "--data-dir=/tmp/y"}, "--data-dir")
	if got != "/tmp/y" {
		t.Errorf("expected /tmp/y from --data-dir=, got %q", got)
	}
}

// printCLIHelp writes its full menu to the supplied writer. Capture
// it and verify the user-facing surface mentions every documented
// command, so a future delete of e.g. --inspect from the dispatch
// without updating help is caught.
func TestPrintCLIHelp_MentionsKeyFlags(t *testing.T) {
	var buf bytes.Buffer
	if err := printCLIHelp(&buf); err != nil {
		t.Fatal(err)
	}
	out := buf.String()
	for _, want := range []string{
		"--status", "--inspect", "--dry-run", "--reset",
		"--list-projects", "--cmd", "--help", "--version",
		"FOCUS_GATE_DATA_DIR", "FOCUS_GATE_CONFIG",
		"/focus",
	} {
		if !strings.Contains(out, want) {
			t.Errorf("help output missing %q", want)
		}
	}
}

// slashMemorySource (attach/detach/enable/disable/default) must
// persist registry changes via SourceRegistry.Save and surface
// confirmation lines. Round-trip through LoadSources catches any
// silent state drop.
func TestSlashMemorySource_AttachDisableDetachRoundTrip(t *testing.T) {
	p, _ := setupTestPaths(t)
	otherDir := t.TempDir()

	var buf bytes.Buffer
	// Attach a writable source.
	if err := slashMemorySource(&buf, p, "attach team "+otherDir); err != nil {
		t.Fatalf("attach: %v", err)
	}
	if !strings.Contains(buf.String(), "Attached source \"team\"") {
		t.Errorf("expected attach confirmation, got %q", buf.String())
	}

	// Verify on-disk registry has the new source.
	r, err := memory.LoadSources(p.dataDir, p.memoryDir)
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := r.Get("team"); !ok {
		t.Fatal("team source missing from registry after attach")
	}

	// Disable, then verify EnabledSources excludes it.
	buf.Reset()
	if err := slashMemorySource(&buf, p, "disable team"); err != nil {
		t.Fatalf("disable: %v", err)
	}
	r, _ = memory.LoadSources(p.dataDir, p.memoryDir)
	for _, s := range r.EnabledSources() {
		if s.Name == "team" {
			t.Error("team should be disabled, but appears in EnabledSources")
		}
	}

	// Re-enable, then detach.
	buf.Reset()
	if err := slashMemorySource(&buf, p, "enable team"); err != nil {
		t.Fatalf("enable: %v", err)
	}
	buf.Reset()
	if err := slashMemorySource(&buf, p, "detach team"); err != nil {
		t.Fatalf("detach: %v", err)
	}
	r, _ = memory.LoadSources(p.dataDir, p.memoryDir)
	if _, ok := r.Get("team"); ok {
		t.Error("team should be gone after detach")
	}

	// Personal cannot be detached — must surface the rejection.
	buf.Reset()
	if err := slashMemorySource(&buf, p, "detach personal"); err != nil {
		t.Fatalf("detach personal: %v", err)
	}
	if !strings.Contains(strings.ToLower(buf.String()), "cannot detach") {
		t.Errorf("expected refusal to detach personal, got %q", buf.String())
	}
}

// slashMemoryReindex --source <name> must rebuild only the named
// source and surface the count. The pre-fix behaviour silently
// ignored the flag (B-7 / D-5).
func TestSlashMemoryReindex_TargetsNamedSource(t *testing.T) {
	p, _ := setupTestPaths(t)
	otherDir := t.TempDir()

	// Write one memory in the named source.
	engine := tfidf.NewEngine()
	vocab := memory.NewVocabSnapshot(engine)
	m := &memory.Memory{Title: "Auth"}
	if err := memory.AppendChapter(m, memory.Chapter{
		Date: time.Date(2026, 4, 1, 0, 0, 0, 0, time.UTC),
		What: "did", Why: "because",
	}); err != nil {
		t.Fatal(err)
	}
	if err := memory.WriteFile(otherDir, m, vocab); err != nil {
		t.Fatal(err)
	}

	// Attach + reindex.
	r, _ := memory.LoadSources(p.dataDir, p.memoryDir)
	if err := r.Attach("team", otherDir, true); err != nil {
		t.Fatal(err)
	}
	if err := r.Save(p.dataDir); err != nil {
		t.Fatal(err)
	}

	var buf bytes.Buffer
	if err := slashMemoryReindex(&buf, p, engine, "--source team"); err != nil {
		t.Fatalf("reindex: %v", err)
	}
	out := buf.String()
	if !strings.Contains(out, "Reindexed 1 memorie(s)") {
		t.Errorf("expected reindex confirmation, got %q", out)
	}
	if !strings.Contains(out, "[team]") {
		t.Errorf("expected source tag in output, got %q", out)
	}
}

// slashMemoryWhy with a prompt that has no overlap surfaces the
// "no memory had any tier hit" branch. Confirms the new debug command
// exits cleanly when the catalog has nothing to say.
func TestSlashMemoryWhy_EmptyCatalogBranch(t *testing.T) {
	p, cfg := setupTestPaths(t)
	engine := tfidf.NewEngine()
	var buf bytes.Buffer
	if err := slashMemoryWhy(&buf, p, cfg, engine, "anything"); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(buf.String(), "No memories in any enabled source") {
		t.Errorf("expected empty-catalog message, got %q", buf.String())
	}
}

// /focus memory add '<json>' must run the same ValidateCommit +
// ApplyCommit pipeline as /focus memory commit but without requiring a
// pending tempId. This is the entry point for hand-curated domain
// memory authoring (skill catalogs, architecture docs).
func TestSlashMemoryAdd_CreatesMemoryViaCanonicalPipeline(t *testing.T) {
	p, _ := setupTestPaths(t)
	engine := tfidf.NewEngine()

	payload := `{
		"action": "create",
		"newMemory": {
			"title": "Core: claims",
			"chapter": {
				"title": "Initial brief",
				"timeMarker": "2026-05-08",
				"assets": ["services/src/el/components/claims"],
				"topics": [{"name": "claim adjudication", "weight": 1.0}],
				"what": "Claims domain owns submission, history, cancellation, and adjudication integration.",
				"why": "Core differentiator; clients pay for claims handling rigor."
			}
		}
	}`

	var buf bytes.Buffer
	if err := slashMemoryAdd(&buf, p, engine, payload); err != nil {
		t.Fatalf("add: %v", err)
	}
	out := buf.String()
	if !strings.Contains(out, "Created memory") {
		t.Errorf("expected creation confirmation, got %q", out)
	}

	// Manifest must be rebuilt — verify by reading back.
	mf, _ := memory.Load(p.memoryDir)
	if len(mf.Entries) != 1 {
		t.Fatalf("expected 1 memory in manifest after add, got %d", len(mf.Entries))
	}
	if mf.Entries[0].Title != "Core: claims" {
		t.Errorf("title mismatch: got %q", mf.Entries[0].Title)
	}
}

// add must surface ValidateCommit's structured errors verbatim — same
// {field, reason, hint} shape the commit path produces — so a
// pedantic-mode author can correct a malformed payload in one
// round-trip.
func TestSlashMemoryAdd_SurfacesValidationErrors(t *testing.T) {
	p, _ := setupTestPaths(t)
	engine := tfidf.NewEngine()

	// Bad: action=create but missing newMemory.
	badPayload := `{"action": "create"}`
	var buf bytes.Buffer
	if err := slashMemoryAdd(&buf, p, engine, badPayload); err != nil {
		t.Fatal(err)
	}
	out := buf.String()
	if !strings.Contains(out, "newMemory") {
		t.Errorf("expected newMemory validation error, got %q", out)
	}
	// No memory should have been written.
	mf, _ := memory.Load(p.memoryDir)
	if len(mf.Entries) != 0 {
		t.Errorf("expected zero memories after failed validation, got %d", len(mf.Entries))
	}
}

// discard via add is a conceptual mismatch (nothing to discard from
// when there's no pending queue lookup). Must be rejected with a clear
// message rather than silently succeeding as a no-op.
func TestSlashMemoryAdd_RejectsDiscardAction(t *testing.T) {
	p, _ := setupTestPaths(t)
	engine := tfidf.NewEngine()
	var buf bytes.Buffer
	if err := slashMemoryAdd(&buf, p, engine, `{"action":"discard"}`); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(buf.String(), "discard") {
		t.Errorf("expected discard-mismatch message, got %q", buf.String())
	}
}

// /focus memory diff <id> renders chapter-by-chapter delta, showing
// only newly-introduced assets/topics/interests per chapter (not the
// running union). Closes the U-2 gap.
func TestSlashMemoryDiff_ShowsPerChapterDelta(t *testing.T) {
	p, _ := setupTestPaths(t)
	engine := tfidf.NewEngine()
	vocab := memory.NewVocabSnapshot(engine)

	m := &memory.Memory{Title: "Auth"}
	if err := memory.AppendChapter(m, memory.Chapter{
		Date:       time.Date(2026, 4, 1, 0, 0, 0, 0, time.UTC),
		Title:      "Initial",
		TimeMarker: "2026-04-01",
		Assets:     []string{"cmd/api/auth.go"},
		Topics:     []string{"JWT"},
		What:       "Used JWT with RS256.",
		Why:        "Multi-service rotation.",
	}); err != nil {
		t.Fatal(err)
	}
	if err := memory.AppendChapter(m, memory.Chapter{
		Date:       time.Date(2026, 4, 12, 0, 0, 0, 0, time.UTC),
		Title:      "Refresh rotation",
		TimeMarker: "2026-04-12",
		Assets:     []string{"middleware/refresh.go", "cmd/api/auth.go"}, // auth.go duplicated
		Topics:     []string{"refresh tokens"},                           // new
		What:       "Added 24h refresh window.",
		Why:        "Limit blast radius.",
	}); err != nil {
		t.Fatal(err)
	}
	if err := memory.WriteFile(p.memoryDir, m, vocab); err != nil {
		t.Fatal(err)
	}

	var buf bytes.Buffer
	if err := slashMemoryDiff(&buf, p, engine, m.ID); err != nil {
		t.Fatal(err)
	}
	out := buf.String()
	for _, want := range []string{
		"Chapter 1", "Chapter 2",
		"cmd/api/auth.go",       // introduced in ch1
		"middleware/refresh.go", // introduced in ch2
		"refresh tokens",        // new topic in ch2
	} {
		if !strings.Contains(out, want) {
			t.Errorf("diff output missing %q; full output:\n%s", want, out)
		}
	}
	// auth.go must NOT appear in chapter 2's section — it was already
	// introduced in chapter 1, so the diff should suppress duplicates.
	ch2Start := strings.Index(out, "Chapter 2")
	if ch2Start < 0 {
		t.Fatal("Chapter 2 section missing")
	}
	if strings.Contains(out[ch2Start:], "cmd/api/auth.go") {
		t.Errorf("diff should not repeat assets introduced in earlier chapters; output:\n%s", out)
	}
}

// Smoke test: persist.SaveAtomic + handleReset cycle on real files.
// Catches regressions where Reset stops removing a state file the
// hook actually creates.
func TestHandleReset_AfterRealSave(t *testing.T) {
	p, _ := setupTestPaths(t)

	// Use the real save path so the test exercises whatever filename
	// handlePrompt would have written.
	if err := persist.SaveAtomic(p.intentFile, map[string]any{"x": 1}); err != nil {
		t.Fatal(err)
	}
	if err := handleReset(p); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(p.intentFile); !os.IsNotExist(err) {
		t.Errorf("intentFile should be gone, got err=%v", err)
	}
}
