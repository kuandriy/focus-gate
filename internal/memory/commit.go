package memory

import (
	"encoding/json"
	"fmt"
	"path/filepath"
	"strings"
)

// timeMarkerMaxLen caps the length of a chapter time marker. Time
// markers are free-form short temporal labels — ISO dates ("2026-04-12"),
// ranges ("2026-04-10..2026-04-12"), or developer-shorthand markers
// ("sprint-42", "v1.2-release", "Q4-2026"). The constraints are
// minimal on purpose: anything humans actually use to anchor a
// memory in time should pass; only obvious garbage (multi-line
// pastes, prose paragraphs, control characters) is rejected.
const timeMarkerMaxLen = 60

// invalidMarkerRune returns true for runes a time marker must never
// contain: ASCII control chars (including newline/tab) and the Unicode
// line/paragraph separators. Used by validateTimeMarker so the rejection
// criterion is one line of code instead of a regexp.
func invalidMarkerRune(r rune) bool {
	if r < 0x20 || r == 0x7f {
		return true
	}
	if r == 0x2028 || r == 0x2029 { // LINE SEPARATOR, PARAGRAPH SEPARATOR
		return true
	}
	return false
}

// CommitAction is the decision string the LLM emits at the end of a
// Stage B review. Three actions only — anything else is a validation
// error.
const (
	CommitActionAppend  = "append"
	CommitActionCreate  = "create"
	CommitActionDiscard = "discard"
)

// CommitPayload is the parsed `fg: memory commit` JSON. Mirrors the
// schema in SHARED_MEMORY_PLAN §6 verbatim.
//
// TargetID and TargetSource are only meaningful for action=append; for
// action=create the LLM populates NewMemory (which itself contains a
// Chapter — that becomes Chapter 1).
//
// Discard carries no body — just the action string.
type CommitPayload struct {
	Action       string           `json:"action"`
	TargetID     string           `json:"targetId,omitempty"`
	TargetSource string           `json:"targetSource,omitempty"`
	Chapter      *CommitChapter   `json:"chapter,omitempty"`
	NewMemory    *CommitNewMemory `json:"newMemory,omitempty"`
}

// CommitChapter is one chapter as proposed by the LLM. Title is
// human-readable; the persisted Chapter heading is composed by
// renderChapters from index/date/title at WriteFile time.
type CommitChapter struct {
	Title      string                `json:"title,omitempty"`
	TimeMarker string                `json:"timeMarker,omitempty"`
	Assets     []string              `json:"assets,omitempty"`
	Interests  []CommitWeightedEntry `json:"interests,omitempty"`
	Topics     []CommitWeightedEntry `json:"topics,omitempty"`
	What       string                `json:"what,omitempty"`
	Why        string                `json:"why,omitempty"`
}

// CommitWeightedEntry mirrors WeightedEntry but in the JSON shape the
// LLM sees. Validation converts to WeightedEntry before applying.
type CommitWeightedEntry struct {
	Name   string  `json:"name"`
	Weight float64 `json:"weight"`
}

// CommitNewMemory carries the registration metadata + Chapter 1 for a
// brand-new story. Top-level Title is the only field strictly required
// — Interests/Topics/Assets/TimeMarkers are derived from the chapter on
// save, but the LLM is invited to seed them so future chapters extend a
// known scope.
type CommitNewMemory struct {
	Title       string                `json:"title"`
	TimeMarkers []string              `json:"timeMarkers,omitempty"`
	Assets      []string              `json:"assets,omitempty"`
	Interests   []CommitWeightedEntry `json:"interests,omitempty"`
	Topics      []CommitWeightedEntry `json:"topics,omitempty"`
	Chapter     *CommitChapter        `json:"chapter"`
}

// CommitError is one structured validation failure. The LLM receives a
// list of these on a failed commit so it can correct in one round-trip
// rather than guessing.
type CommitError struct {
	Field  string `json:"field"`
	Reason string `json:"reason"`
	Hint   string `json:"hint,omitempty"`
}

func (e CommitError) Error() string {
	if e.Hint != "" {
		return fmt.Sprintf("%s: %s (%s)", e.Field, e.Reason, e.Hint)
	}
	return fmt.Sprintf("%s: %s", e.Field, e.Reason)
}

// ParseCommitJSON decodes raw JSON into a CommitPayload. Returns a
// single-element CommitError slice on a JSON-level failure so callers
// don't need to special-case parse vs validate errors.
//
// Unknown top-level fields are tolerated — the LLM may include extra
// metadata (notes, confidence, etc.) and we'd rather process the parts
// we recognize than reject the whole payload.
func ParseCommitJSON(raw []byte) (*CommitPayload, []CommitError) {
	var p CommitPayload
	if err := json.Unmarshal(raw, &p); err != nil {
		return nil, []CommitError{{
			Field:  "(root)",
			Reason: "invalid JSON",
			Hint:   err.Error(),
		}}
	}
	return &p, nil
}

// ValidateCommit checks every invariant the LLM can violate: action
// in the allowed set, action-appropriate fields populated, target
// memory exists for append, no empty What/Why on chapters. Returns a
// list of structured errors so the LLM can fix in one shot.
//
// The MultiSourceIndex is consulted for:
//   - confirming the targetId exists for append
//   - confirming targetSource is attached and writable (writability
//     is enforced by Apply, not here — Validate just confirms the
//     source is known)
func ValidateCommit(p *CommitPayload, msi *MultiSourceIndex) []CommitError {
	var errs []CommitError
	if p == nil {
		return []CommitError{{Field: "(root)", Reason: "missing payload"}}
	}
	switch p.Action {
	case CommitActionAppend:
		if p.TargetID == "" {
			errs = append(errs, CommitError{
				Field:  "targetId",
				Reason: "required for action=append",
				Hint:   "set targetId to the id of the existing memory you're appending to",
			})
		}
		if p.Chapter == nil {
			errs = append(errs, CommitError{
				Field:  "chapter",
				Reason: "required for action=append",
				Hint:   "include a chapter object with at minimum what/why/timeMarker",
			})
		} else {
			errs = append(errs, validateChapter("chapter", p.Chapter)...)
		}
		if msi != nil && p.TargetID != "" {
			if !targetExists(msi, p.TargetID, p.TargetSource) {
				errs = append(errs, CommitError{
					Field:  "targetId",
					Reason: "no memory with that id in the requested source",
					Hint:   "double-check targetId; targetSource defaults to the configured default if omitted",
				})
			}
		}
		if p.NewMemory != nil {
			errs = append(errs, CommitError{
				Field:  "newMemory",
				Reason: "must be omitted for action=append",
			})
		}
	case CommitActionCreate:
		if p.NewMemory == nil {
			errs = append(errs, CommitError{
				Field:  "newMemory",
				Reason: "required for action=create",
				Hint:   "include newMemory object with title and chapter",
			})
		} else {
			if strings.TrimSpace(p.NewMemory.Title) == "" {
				errs = append(errs, CommitError{
					Field:  "newMemory.title",
					Reason: "required",
				})
			} else if len([]rune(p.NewMemory.Title)) > 80 {
				errs = append(errs, CommitError{
					Field:  "newMemory.title",
					Reason: "too long (max 80 runes)",
				})
			}
			if p.NewMemory.Chapter == nil {
				errs = append(errs, CommitError{
					Field:  "newMemory.chapter",
					Reason: "required for action=create — this becomes Chapter 1",
				})
			} else {
				errs = append(errs, validateChapter("newMemory.chapter", p.NewMemory.Chapter)...)
			}
		}
		if p.TargetID != "" {
			errs = append(errs, CommitError{
				Field:  "targetId",
				Reason: "must be omitted for action=create",
			})
		}
		if p.Chapter != nil {
			errs = append(errs, CommitError{
				Field:  "chapter",
				Reason: "must be omitted for action=create — put the chapter inside newMemory.chapter",
			})
		}
	case CommitActionDiscard:
		if p.NewMemory != nil || p.Chapter != nil || p.TargetID != "" || p.TargetSource != "" {
			errs = append(errs, CommitError{
				Field:  "(root)",
				Reason: "discard takes no body — only action: discard is permitted",
			})
		}
	default:
		errs = append(errs, CommitError{
			Field:  "action",
			Reason: fmt.Sprintf("unknown action %q", p.Action),
			Hint:   "one of: append, create, discard",
		})
	}
	return errs
}

func validateChapter(prefix string, ch *CommitChapter) []CommitError {
	var errs []CommitError
	if strings.TrimSpace(ch.What) == "" {
		errs = append(errs, CommitError{
			Field:  prefix + ".what",
			Reason: "required, non-empty",
			Hint:   "what was decided/learned, in past tense",
		})
	}
	if strings.TrimSpace(ch.Why) == "" {
		errs = append(errs, CommitError{
			Field:  prefix + ".why",
			Reason: "required, non-empty",
			Hint:   "the rationale or constraint that drove the decision",
		})
	}
	// Time marker is optional but if present must be a short single-line
	// label. Free-form is allowed by design — "sprint-42", "v1.2-release",
	// and "2026-04-12" are all valid. We reject only obvious garbage:
	// multi-line pastes, prose paragraphs, control characters.
	if tm := strings.TrimSpace(ch.TimeMarker); tm != "" {
		if n := len([]rune(tm)); n > timeMarkerMaxLen {
			errs = append(errs, CommitError{
				Field:  prefix + ".timeMarker",
				Reason: fmt.Sprintf("too long (%d runes, max %d)", n, timeMarkerMaxLen),
				Hint:   "keep it short — a date, range, or short label like \"sprint-42\"",
			})
		}
		for _, r := range tm {
			if invalidMarkerRune(r) {
				errs = append(errs, CommitError{
					Field:  prefix + ".timeMarker",
					Reason: "contains a control character or line separator",
					Hint:   "single line only; no tabs/newlines",
				})
				break
			}
		}
	}
	for i, w := range ch.Interests {
		if strings.TrimSpace(w.Name) == "" {
			errs = append(errs, CommitError{
				Field:  fmt.Sprintf("%s.interests[%d].name", prefix, i),
				Reason: "required",
			})
		}
		if w.Weight < 0 || w.Weight > 1 {
			errs = append(errs, CommitError{
				Field:  fmt.Sprintf("%s.interests[%d].weight", prefix, i),
				Reason: "must be in [0,1]",
			})
		}
	}
	for i, w := range ch.Topics {
		if strings.TrimSpace(w.Name) == "" {
			errs = append(errs, CommitError{
				Field:  fmt.Sprintf("%s.topics[%d].name", prefix, i),
				Reason: "required",
			})
		}
		if w.Weight < 0 || w.Weight > 1 {
			errs = append(errs, CommitError{
				Field:  fmt.Sprintf("%s.topics[%d].weight", prefix, i),
				Reason: "must be in [0,1]",
			})
		}
	}
	return errs
}

func targetExists(msi *MultiSourceIndex, id, source string) bool {
	for _, m := range msi.Manifests {
		if source != "" && m.Source != source {
			continue
		}
		if _, ok := m.Get(id); ok {
			return true
		}
	}
	return false
}

// CommitContext bundles what Apply needs to actually persist a
// validated commit: the per-source memory directories (so we can find
// the target file for append, or write a new one for create), the
// vocab snapshot for fingerprint/topTerms recompute, and a default
// source name for when targetSource is null/empty.
type CommitContext struct {
	// SourceDirs maps source name → on-disk memories directory. The
	// commit handler looks up the relevant entry to read/write files.
	// Phase 5 wires this from `memory.sources`; before then, the
	// caller passes a single-entry map keyed by DefaultSourceName.
	SourceDirs map[string]string

	// DefaultSource is used when the LLM omits TargetSource.
	DefaultSource string

	// Vocab is the engine's current vocabulary snapshot. Passed to
	// WriteFile so fingerprints stay consistent across the catalog.
	Vocab VocabSnapshot

	// Index resolves a TargetID to its owning source when the LLM
	// emits an append payload without TargetSource. Validate accepts
	// "any source contains the id" so Apply must do the same — using
	// DefaultSource blindly would 404 on cross-source appends.
	// Optional: callers that maintain a single-source setup can leave
	// it nil and rely on DefaultSource.
	Index *MultiSourceIndex
}

// ApplyResult names the memory that was created or updated. Surface so
// the slash command can print a confirmation pointing at the affected
// file.
type ApplyResult struct {
	Action   string
	Memory   *Memory // populated for append/create
	FilePath string  // absolute path of the .md file
	Source   string
}

// ApplyCommit persists a validated commit. Caller is expected to have
// invoked ValidateCommit first; ApplyCommit re-checks the action shape
// only enough to dispatch — it trusts validation for field-level
// invariants.
//
// Discard is a no-op at the file level; the caller is responsible for
// removing the matching candidate from the pending queue.
func ApplyCommit(p *CommitPayload, ctx CommitContext) (*ApplyResult, error) {
	if p == nil {
		return nil, fmt.Errorf("nil payload")
	}
	switch p.Action {
	case CommitActionDiscard:
		return &ApplyResult{Action: p.Action}, nil
	case CommitActionAppend:
		return applyAppend(p, ctx)
	case CommitActionCreate:
		return applyCreate(p, ctx)
	default:
		return nil, fmt.Errorf("unsupported action %q", p.Action)
	}
}

func applyAppend(p *CommitPayload, ctx CommitContext) (*ApplyResult, error) {
	src := p.TargetSource
	if src == "" && ctx.Index != nil {
		// LLM omitted TargetSource — find which attached source actually
		// owns this id. Avoids a 404 when the target lives outside the
		// configured DefaultSource.
		for _, m := range ctx.Index.Manifests {
			if _, ok := m.Get(p.TargetID); ok {
				src = m.Source
				break
			}
		}
	}
	if src == "" {
		src = ctx.DefaultSource
	}
	if src == "" {
		src = DefaultSourceName
	}
	dir, ok := ctx.SourceDirs[src]
	if !ok {
		return nil, fmt.Errorf("unknown source %q for append", src)
	}
	// Locate the target file. The IndexEntry.Path is relative to the
	// source directory; we don't have the manifest here, so reconstruct
	// the path from the ID convention via filepath.Join (handles
	// trailing-separator and Windows-separator edge cases correctly).
	path := filepath.Join(dir, p.TargetID+".md")
	mem, err := ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read target: %w", err)
	}
	if err := AppendChapter(mem, chapterFromPayload(p.Chapter)); err != nil {
		return nil, fmt.Errorf("append chapter: %w", err)
	}
	if err := WriteFile(dir, mem, ctx.Vocab); err != nil {
		return nil, fmt.Errorf("write memory: %w", err)
	}
	return &ApplyResult{
		Action:   p.Action,
		Memory:   mem,
		FilePath: mem.Path(dir),
		Source:   src,
	}, nil
}

func applyCreate(p *CommitPayload, ctx CommitContext) (*ApplyResult, error) {
	src := p.TargetSource
	if src == "" {
		src = ctx.DefaultSource
	}
	if src == "" {
		src = DefaultSourceName
	}
	dir, ok := ctx.SourceDirs[src]
	if !ok {
		return nil, fmt.Errorf("unknown source %q for create", src)
	}
	mem := &Memory{Title: p.NewMemory.Title}
	if err := AppendChapter(mem, chapterFromPayload(p.NewMemory.Chapter)); err != nil {
		return nil, fmt.Errorf("append chapter: %w", err)
	}
	if err := WriteFile(dir, mem, ctx.Vocab); err != nil {
		return nil, fmt.Errorf("write memory: %w", err)
	}
	return &ApplyResult{
		Action:   p.Action,
		Memory:   mem,
		FilePath: mem.Path(dir),
		Source:   src,
	}, nil
}

// chapterFromPayload converts the JSON-shape CommitChapter into the
// in-memory Chapter type that AppendChapter accepts. Date is the
// zero value — AppendChapter stamps now() in that case.
func chapterFromPayload(c *CommitChapter) Chapter {
	if c == nil {
		return Chapter{}
	}
	out := Chapter{
		Title:      c.Title,
		TimeMarker: c.TimeMarker,
		Assets:     append([]string{}, c.Assets...),
		What:       c.What,
		Why:        c.Why,
	}
	for _, w := range c.Interests {
		out.Interests = append(out.Interests, w.Name)
	}
	for _, w := range c.Topics {
		out.Topics = append(out.Topics, w.Name)
	}
	return out
}
