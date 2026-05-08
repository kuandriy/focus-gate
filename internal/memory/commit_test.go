package memory

import (
	"strings"
	"testing"
	"time"
)

// commitFixtureMemory writes a v2 memory to a fresh dir and returns
// (dir, memory, vocab). Used by append-action commit tests so they
// have a real target to extend.
func commitFixtureMemory(t *testing.T) (string, *Memory, VocabSnapshot) {
	t.Helper()
	dir := t.TempDir()
	vocab := fakeVocab("vh1")
	m := &Memory{Title: "Auth"}
	if err := AppendChapter(m, Chapter{
		Date:       time.Date(2026, 4, 1, 0, 0, 0, 0, time.UTC),
		Title:      "Initial",
		TimeMarker: "2026-04-01",
		Assets:     []string{"cmd/api/auth.go"},
		Topics:     []string{"JWT authentication"},
		What:       "did the thing",
		Why:        "because",
	}); err != nil {
		t.Fatal(err)
	}
	if err := WriteFile(dir, m, vocab); err != nil {
		t.Fatal(err)
	}
	return dir, m, vocab
}

func TestParseCommitJSON_ValidPayload(t *testing.T) {
	raw := `{
		"action": "create",
		"newMemory": {
			"title": "Auth & session model",
			"chapter": {
				"timeMarker": "2026-04-01",
				"what": "...",
				"why": "..."
			}
		}
	}`
	p, errs := ParseCommitJSON([]byte(raw))
	if len(errs) != 0 {
		t.Fatalf("parse errors: %v", errs)
	}
	if p.Action != "create" {
		t.Errorf("action = %q, want create", p.Action)
	}
	if p.NewMemory == nil || p.NewMemory.Title != "Auth & session model" {
		t.Errorf("title not parsed: %+v", p.NewMemory)
	}
}

func TestParseCommitJSON_MalformedReportsError(t *testing.T) {
	_, errs := ParseCommitJSON([]byte(`{not json`))
	if len(errs) == 0 {
		t.Fatal("expected parse errors")
	}
	if errs[0].Field != "(root)" {
		t.Errorf("field = %q, want (root)", errs[0].Field)
	}
}

func TestParseCommitJSON_TolerantToUnknownFields(t *testing.T) {
	// LLM may include extra fields like confidence; we accept and ignore.
	raw := `{"action": "discard", "confidence": 0.9, "notes": "hello"}`
	p, errs := ParseCommitJSON([]byte(raw))
	if len(errs) != 0 {
		t.Fatalf("unknown fields should be tolerated: %v", errs)
	}
	if p.Action != "discard" {
		t.Errorf("action = %q, want discard", p.Action)
	}
}

func TestValidateCommit_AppendRequiresTargetAndChapter(t *testing.T) {
	msi := NewMultiSourceIndex(NewManifest())
	errs := ValidateCommit(&CommitPayload{Action: "append"}, msi)
	got := errFields(errs)
	for _, want := range []string{"targetId", "chapter"} {
		if !got[want] {
			t.Errorf("expected error for %q, got %v", want, errFieldList(errs))
		}
	}
}

func TestValidateCommit_AppendUnknownTarget(t *testing.T) {
	msi := NewMultiSourceIndex(NewManifest())
	p := &CommitPayload{
		Action:   "append",
		TargetID: "mem_does_not_exist",
		Chapter:  &CommitChapter{What: "x", Why: "y"},
	}
	errs := ValidateCommit(p, msi)
	if !errFields(errs)["targetId"] {
		t.Errorf("expected targetId error for missing memory, got %v", errFieldList(errs))
	}
}

func TestValidateCommit_AppendKnownTargetSucceeds(t *testing.T) {
	mf := NewManifest()
	mf.Entries = []IndexEntry{{ID: "mem_xyz", Source: "personal"}}
	msi := NewMultiSourceIndex(mf)

	p := &CommitPayload{
		Action:   "append",
		TargetID: "mem_xyz",
		Chapter:  &CommitChapter{What: "what", Why: "why", TimeMarker: "2026-04-01"},
	}
	errs := ValidateCommit(p, msi)
	if len(errs) != 0 {
		t.Fatalf("unexpected validation errors: %v", errs)
	}
}

func TestValidateCommit_CreateRequiresNewMemory(t *testing.T) {
	errs := ValidateCommit(&CommitPayload{Action: "create"}, nil)
	if !errFields(errs)["newMemory"] {
		t.Errorf("expected newMemory error, got %v", errFieldList(errs))
	}
}

func TestValidateCommit_CreateChecksTitleAndChapter(t *testing.T) {
	p := &CommitPayload{
		Action: "create",
		NewMemory: &CommitNewMemory{
			Title: "", // missing
			// Chapter missing too
		},
	}
	errs := ValidateCommit(p, nil)
	got := errFields(errs)
	for _, want := range []string{"newMemory.title", "newMemory.chapter"} {
		if !got[want] {
			t.Errorf("expected error for %q, got %v", want, errFieldList(errs))
		}
	}
}

func TestValidateCommit_DiscardRejectsExtraneousFields(t *testing.T) {
	p := &CommitPayload{
		Action:   "discard",
		TargetID: "mem_xyz", // shouldn't be present
	}
	errs := ValidateCommit(p, nil)
	if len(errs) == 0 {
		t.Fatal("expected validation error for discard with extra fields")
	}
}

func TestValidateCommit_UnknownActionRejected(t *testing.T) {
	errs := ValidateCommit(&CommitPayload{Action: "merge"}, nil)
	if !errFields(errs)["action"] {
		t.Errorf("expected action error, got %v", errFieldList(errs))
	}
}

// Time markers are free-form by design — ISO dates, ranges, and
// developer-shorthand labels are all welcome. Only obvious garbage
// (over the length cap, or containing control characters) is
// rejected.
func TestValidateCommit_TimeMarkerInvalidIsRejected(t *testing.T) {
	long := strings.Repeat("x", timeMarkerMaxLen+1)
	for _, bad := range []string{long, "line one\nline two", "has\ttab", "with\rcontrol"} {
		p := &CommitPayload{
			Action: "create",
			NewMemory: &CommitNewMemory{
				Title: "x",
				Chapter: &CommitChapter{
					TimeMarker: bad,
					What:       "x",
					Why:        "y",
				},
			},
		}
		errs := ValidateCommit(p, nil)
		if !errFields(errs)["newMemory.chapter.timeMarker"] {
			t.Errorf("time marker %q should be rejected, got %v", bad, errFieldList(errs))
		}
	}
}

func TestValidateCommit_TimeMarkerValidIsAccepted(t *testing.T) {
	for _, ok := range []string{
		"2026-04-01",
		"2026-04-01..2026-04-15",
		"sprint-42",
		"v1.2-release",
		"Q4 2026",
		"yesterday",
		"",
	} {
		p := &CommitPayload{
			Action: "create",
			NewMemory: &CommitNewMemory{
				Title: "x",
				Chapter: &CommitChapter{
					TimeMarker: ok,
					What:       "x",
					Why:        "y",
				},
			},
		}
		errs := ValidateCommit(p, nil)
		if errFields(errs)["newMemory.chapter.timeMarker"] {
			t.Errorf("time marker %q should be accepted, got %v", ok, errFieldList(errs))
		}
	}
}

func TestValidateCommit_WeightOutOfRange(t *testing.T) {
	p := &CommitPayload{
		Action: "create",
		NewMemory: &CommitNewMemory{
			Title: "x",
			Chapter: &CommitChapter{
				What: "x",
				Why:  "y",
				Topics: []CommitWeightedEntry{
					{Name: "ok", Weight: 1.5},
				},
			},
		},
	}
	errs := ValidateCommit(p, nil)
	if !errFields(errs)["newMemory.chapter.topics[0].weight"] {
		t.Errorf("expected weight bounds error, got %v", errFieldList(errs))
	}
}

func TestApplyCommit_CreateWritesNewMemory(t *testing.T) {
	dir := t.TempDir()
	vocab := fakeVocab("vh1")
	ctx := CommitContext{
		SourceDirs:    map[string]string{"personal": dir},
		DefaultSource: "personal",
		Vocab:         vocab,
	}
	p := &CommitPayload{
		Action: "create",
		NewMemory: &CommitNewMemory{
			Title: "Newly minted",
			Chapter: &CommitChapter{
				TimeMarker: "2026-04-01",
				Assets:     []string{"a.go"},
				Topics:     []CommitWeightedEntry{{Name: "x", Weight: 1.0}},
				What:       "what",
				Why:        "why",
			},
		},
	}
	res, err := ApplyCommit(p, ctx)
	if err != nil {
		t.Fatal(err)
	}
	if res.Memory == nil || res.Memory.Title != "Newly minted" {
		t.Errorf("memory not populated: %+v", res.Memory)
	}
	if res.FilePath == "" || !strings.HasSuffix(res.FilePath, ".md") {
		t.Errorf("expected .md path, got %q", res.FilePath)
	}
	loaded, err := ReadFile(res.FilePath)
	if err != nil {
		t.Fatal(err)
	}
	if len(loaded.ChaptersList) != 1 {
		t.Errorf("expected 1 chapter on disk, got %d", len(loaded.ChaptersList))
	}
	if len(loaded.Topics) != 1 || loaded.Topics[0].Name != "x" {
		t.Errorf("topics not aggregated: %+v", loaded.Topics)
	}
}

func TestApplyCommit_AppendBumpsChapterCount(t *testing.T) {
	dir, fixture, vocab := commitFixtureMemory(t)
	ctx := CommitContext{
		SourceDirs:    map[string]string{"personal": dir},
		DefaultSource: "personal",
		Vocab:         vocab,
	}
	p := &CommitPayload{
		Action:   "append",
		TargetID: fixture.ID,
		Chapter: &CommitChapter{
			TimeMarker: "2026-04-12",
			Assets:     []string{"middleware/refresh.go"},
			What:       "rotated tokens",
			Why:        "to limit blast radius",
		},
	}
	res, err := ApplyCommit(p, ctx)
	if err != nil {
		t.Fatal(err)
	}
	if res.Memory.Chapters != 2 {
		t.Errorf("chapters = %d, want 2", res.Memory.Chapters)
	}
	loaded, err := ReadFile(res.FilePath)
	if err != nil {
		t.Fatal(err)
	}
	if len(loaded.ChaptersList) != 2 {
		t.Errorf("on-disk chapters = %d, want 2", len(loaded.ChaptersList))
	}
	// Aggregate should include both chapters' assets.
	if len(loaded.Assets) != 2 {
		t.Errorf("aggregated assets = %v, want 2", loaded.Assets)
	}
}

func TestApplyCommit_DiscardIsNoop(t *testing.T) {
	res, err := ApplyCommit(&CommitPayload{Action: "discard"}, CommitContext{})
	if err != nil {
		t.Fatal(err)
	}
	if res.Action != "discard" {
		t.Errorf("action = %q, want discard", res.Action)
	}
	if res.Memory != nil {
		t.Error("discard should not populate Memory")
	}
}

// When the LLM emits an append payload without targetSource and the
// target lives in a non-default source, ApplyCommit must still find
// the file via the multi-source index instead of trying DefaultSource
// blindly. Without this resolution the user sees a "read target: file
// not found" error even though Validate just accepted the payload.
func TestApplyCommit_AppendImplicitSourceResolvesViaIndex(t *testing.T) {
	personalDir := t.TempDir()
	teamDir := t.TempDir()
	vocab := fakeVocab("vh1")

	// Target memory exists only in the team source.
	teamMem := &Memory{Title: "Shared"}
	if err := AppendChapter(teamMem, chapterFixture("c1")); err != nil {
		t.Fatal(err)
	}
	if err := WriteFile(teamDir, teamMem, vocab); err != nil {
		t.Fatal(err)
	}

	teamMf := NewManifest()
	teamMf.Source = "team"
	if errs := teamMf.Rebuild(teamDir, vocab); len(errs) > 0 {
		t.Fatalf("rebuild: %v", errs)
	}

	ctx := CommitContext{
		SourceDirs: map[string]string{
			"personal": personalDir,
			"team":   teamDir,
		},
		DefaultSource: "personal",
		Vocab:         vocab,
		Index:         NewMultiSourceIndex(teamMf),
	}
	p := &CommitPayload{
		Action:   "append",
		TargetID: teamMem.ID,
		// TargetSource omitted on purpose
		Chapter: &CommitChapter{
			TimeMarker: "2026-04-01",
			What:       "appended",
			Why:        "to test implicit source resolution",
		},
	}
	res, err := ApplyCommit(p, ctx)
	if err != nil {
		t.Fatalf("expected ApplyCommit to resolve source via index, got %v", err)
	}
	if res.Source != "team" {
		t.Errorf("resolved source = %q, want %q", res.Source, "team")
	}
}

func TestApplyCommit_AppendUnknownSource(t *testing.T) {
	ctx := CommitContext{
		SourceDirs:    map[string]string{"personal": t.TempDir()},
		DefaultSource: "personal",
		Vocab:         fakeVocab("vh1"),
	}
	p := &CommitPayload{
		Action:       "append",
		TargetID:     "mem_x",
		TargetSource: "team",
		Chapter:      &CommitChapter{What: "x", Why: "y"},
	}
	if _, err := ApplyCommit(p, ctx); err == nil {
		t.Error("expected error for unknown source")
	}
}

func TestBuildStageBPrompt_ContainsKeySections(t *testing.T) {
	mf := NewManifest()
	mf.Entries = []IndexEntry{{
		ID:          "mem_existing",
		Source:      "personal",
		Title:       "Auth",
		TimeMarkers: []string{"2026-04-01"},
		Topics:      []WeightedEntry{{Name: "JWT authentication", Weight: 1.0}},
		Assets:      []string{"cmd/api/auth.go"},
	}}
	msi := NewMultiSourceIndex(mf)

	c := &Candidate{
		TempID:                "cand_xyz",
		Reason:                "prune",
		SourceTreeID:          "t1",
		RootAbstraction:       "auth jwt session",
		PromptCount:           5,
		AgeHours:              4.2,
		Refs:                  []string{"cmd/api/auth.go"},
		TopTerms:              []string{"jwt", "session"},
		NodeContents:          []string{"refresh tokens", "rate limit"},
		SuggestedAction:       "append",
		SuggestedTargetID:     "mem_existing",
		SuggestedTargetSource: "personal",
	}
	got := BuildStageBPrompt(c, msi, PromoteOptions{})
	for _, want := range []string{
		"cand_xyz",
		"CANDIDATE SNAPSHOT",
		"EXISTING MEMORIES",
		"mem_existing",
		"JWT authentication@1.00",
		"DECIDE one of",
		"fg: memory commit cand_xyz",
		"\"action\":",
	} {
		if !strings.Contains(got, want) {
			t.Errorf("prompt missing %q in:\n%s", want, got)
		}
	}
}

// The Stage B prompt should ship literal, copy-paste-ready append /
// create / discard payloads ahead of any field reference. LLMs follow
// concrete examples more reliably than schema prose; if these
// disappear from the prompt, the LLM's commit-payload accuracy
// regresses and the validation-retry loop has more work to do.
func TestBuildStageBPrompt_IncludesFewShotExamples(t *testing.T) {
	c := &Candidate{
		TempID:          "cand_examples",
		Reason:          "prune",
		SourceTreeID:    "t1",
		RootAbstraction: "auth jwt session",
	}
	got := BuildStageBPrompt(c, NewMultiSourceIndex(), PromoteOptions{})
	for _, want := range []string{
		"EXAMPLES",
		"Example A — append",
		"Example B — create",
		"Example C — discard",
		`"action": "append"`,
		`"action": "create"`,
		`"action": "discard"`,
		"FIELD REFERENCE",
	} {
		if !strings.Contains(got, want) {
			t.Errorf("prompt missing %q; full prompt:\n%s", want, got)
		}
	}
}

// The original draft told the LLM "you are the student" three times in
// different phrasings. Tightening the framing to one preamble line is
// the explicit goal — guard against regressions where someone restores
// the redundant wording.
func TestBuildStageBPrompt_NoRedundantStudentFraming(t *testing.T) {
	c := &Candidate{TempID: "cand_x", RootAbstraction: "x"}
	got := BuildStageBPrompt(c, NewMultiSourceIndex(), PromoteOptions{})
	if strings.Count(got, "the student") > 1 {
		t.Errorf("prompt repeats 'the student' framing more than once:\n%s", got)
	}
}

// errFields collapses []CommitError into a set keyed by field for easy
// "expected this error" assertions.
func errFields(errs []CommitError) map[string]bool {
	m := map[string]bool{}
	for _, e := range errs {
		m[e.Field] = true
	}
	return m
}

func errFieldList(errs []CommitError) []string {
	out := make([]string, len(errs))
	for i, e := range errs {
		out[i] = e.Field + "/" + e.Reason
	}
	return out
}
