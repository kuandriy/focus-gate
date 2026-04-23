package main

import (
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/kuandriy/focus-gate/internal/memory"
	"github.com/kuandriy/focus-gate/internal/tfidf"
)

// slashMemory dispatches the /focus memory ... family. Splits the
// remaining argument into a subcommand + its own argument tail. Reads
// state directly from disk on each call so the inspector stays decoupled
// from the hook's mutating path.
//
// Session A surface: list / show / health (read-only). Pending and
// promote/commit ship in Sessions B & C of LONG_TERM_MEMORY_PLAN.md.
func slashMemory(w io.Writer, p paths, cfg config, e *tfidf.Engine, arg string) error {
	if !cfg.Memory.Enabled {
		fmt.Fprintln(w, "[Focus] Memory subsystem is disabled (set memory.enabled=true to use it).")
		return nil
	}

	sub, rest := splitArg(arg)
	switch sub {
	case "", "list":
		return slashMemoryList(w, p, e)
	case "show":
		return slashMemoryShow(w, p, e, rest)
	case "pending":
		return slashMemoryPending(w, p, cfg)
	case "discard":
		return slashMemoryDiscard(w, p, cfg, rest)
	case "health":
		return slashMemoryHealth(w, p, e)
	default:
		fmt.Fprintf(w, "[Focus] Unknown memory subcommand: %q\n", sub)
		fmt.Fprintln(w, "Available: fg: memory [list | show <id> | pending | discard <id|all> | health]")
		return nil
	}
}

// splitArg peels the first space-delimited token off arg and returns
// (head, remainder). Used so "list", "show abc123", "health" all parse
// uniformly without a second-level slashCommand struct.
func splitArg(arg string) (string, string) {
	arg = strings.TrimSpace(arg)
	if arg == "" {
		return "", ""
	}
	parts := strings.SplitN(arg, " ", 2)
	if len(parts) == 1 {
		return strings.ToLower(parts[0]), ""
	}
	return strings.ToLower(parts[0]), strings.TrimSpace(parts[1])
}

// slashMemoryList prints the manifest as a compact table. Columns chosen
// for at-a-glance triage: ID + title + last-updated + how often surfaced.
// Refs and source-tree IDs are useful but too noisy for the default view;
// users can `fg: memory show` for the full picture.
func slashMemoryList(w io.Writer, p paths, e *tfidf.Engine) error {
	mf := loadMemoryManifest(p.memoryDir, e)
	if len(mf.Entries) == 0 {
		fmt.Fprintln(w, "[Focus] No memories yet.")
		fmt.Fprintf(w, "  Memory directory: %s\n", p.memoryDir)
		fmt.Fprintln(w, "  Hand-author a .md file there or wait for the promote pipeline (Session C).")
		return nil
	}
	fmt.Fprintln(w, "=== Memories ===")
	fmt.Fprintf(w, "  Directory: %s\n", p.memoryDir)
	fmt.Fprintf(w, "  Total: %d, manifest rebuilt %s\n", len(mf.Entries), formatRelTime(mf.RebuiltAt))
	fmt.Fprintln(w)
	fmt.Fprintf(w, "  %-26s  %-6s  %-12s  %s\n", "ID", "TOUCH", "UPDATED", "TITLE")
	for _, e := range mf.Entries {
		title := e.Title
		if len(title) > 60 {
			title = title[:57] + "..."
		}
		fmt.Fprintf(w, "  %-26s  %-6d  %-12s  %s\n",
			e.ID, e.TouchedBy, formatRelTime(e.Updated), title)
	}
	return nil
}

// slashMemoryShow renders a single memory's frontmatter + body. ID
// matching is exact OR by prefix (so users can type the first 6 hex
// chars). Useful when the AI surfaces a pointer and the user wants to
// see the full body without leaving the chat.
func slashMemoryShow(w io.Writer, p paths, e *tfidf.Engine, idOrPrefix string) error {
	idOrPrefix = strings.TrimSpace(idOrPrefix)
	if idOrPrefix == "" {
		fmt.Fprintln(w, "[Focus] Usage: fg: memory show <id-or-prefix>")
		return nil
	}
	mf := loadMemoryManifest(p.memoryDir, e)

	var match *memory.IndexEntry
	for i := range mf.Entries {
		if mf.Entries[i].ID == idOrPrefix || strings.HasPrefix(mf.Entries[i].ID, idOrPrefix) {
			if match != nil {
				fmt.Fprintf(w, "[Focus] Ambiguous prefix %q — matches multiple memories. Use the full ID.\n", idOrPrefix)
				return nil
			}
			match = &mf.Entries[i]
		}
	}
	if match == nil {
		fmt.Fprintf(w, "[Focus] No memory matches %q.\n", idOrPrefix)
		return nil
	}

	mem, err := memory.ReadFile(filepath.Join(p.memoryDir, match.Path))
	if err != nil {
		fmt.Fprintf(w, "[Focus] Failed to read %s: %v\n", match.Path, err)
		return nil
	}
	fmt.Fprintf(w, "=== %s ===\n", mem.ID)
	fmt.Fprintf(w, "  Title:     %s\n", mem.Title)
	fmt.Fprintf(w, "  Path:      %s\n", filepath.Join(p.memoryDir, match.Path))
	fmt.Fprintf(w, "  Created:   %s\n", mem.Created.Format(time.RFC3339))
	fmt.Fprintf(w, "  Updated:   %s\n", mem.Updated.Format(time.RFC3339))
	fmt.Fprintf(w, "  TouchedBy: %d\n", mem.TouchedBy)
	if len(mem.Sources) > 0 {
		fmt.Fprintf(w, "  Sources:   %s\n", strings.Join(mem.Sources, ", "))
	}
	if len(mem.Refs) > 0 {
		fmt.Fprintf(w, "  Refs:      %s\n", strings.Join(mem.Refs, ", "))
	}
	if len(mem.TopTerms) > 0 {
		fmt.Fprintf(w, "  TopTerms:  %s\n", strings.Join(mem.TopTerms, ", "))
	}
	fmt.Fprintln(w)
	fmt.Fprintln(w, mem.Body)
	return nil
}

// slashMemoryPending prints the pending candidate queue. Each row shows
// the tempId, reason (prune vs merge), suggested action, and the first
// line of the source tree's abstraction so the user can eyeball the
// queue before running `fg: memory promote`.
func slashMemoryPending(w io.Writer, p paths, cfg config) error {
	maxAge := parseDuration(cfg.Memory.PendingMaxAge, 168*time.Hour)
	pq, err := memory.LoadPending(p.dataDir, maxAge)
	if err != nil {
		fmt.Fprintf(w, "[Focus] Failed to load pending queue: %v\n", err)
		return nil
	}
	if pq == nil || len(pq.Candidates) == 0 {
		fmt.Fprintln(w, "[Focus] No pending memory candidates.")
		fmt.Fprintln(w, "  Candidates are emitted automatically when the forest prunes")
		fmt.Fprintln(w, "  substantive trees or cluster-merges them away.")
		return nil
	}
	fmt.Fprintln(w, "=== Pending Memory Candidates ===")
	fmt.Fprintf(w, "  Queue: %d candidate(s), last updated %s\n\n", len(pq.Candidates), formatRelTime(pq.UpdatedAt))
	fmt.Fprintf(w, "  %-34s  %-6s  %-8s  %s\n", "TEMPID", "REASON", "ACTION", "ABSTRACTION")
	for _, c := range pq.Sorted() {
		abs := c.RootAbstraction
		if len(abs) > 60 {
			abs = abs[:57] + "..."
		}
		target := c.SuggestedAction
		if c.SuggestedAction == "merge" && c.SuggestedTargetID != "" {
			target = "merge→" + shortID(c.SuggestedTargetID)
		}
		fmt.Fprintf(w, "  %-34s  %-6s  %-8s  %s\n", c.TempID, c.Reason, target, abs)
	}
	fmt.Fprintln(w)
	fmt.Fprintln(w, "Run `fg: memory promote` to generate an LLM-ready bundle (Session C).")
	fmt.Fprintln(w, "Run `fg: memory discard <tempId|all>` to clear entries manually.")
	return nil
}

// slashMemoryDiscard removes entries from the pending queue without
// creating memories. Supports a single tempId (exact match) or the
// literal "all" to clear the queue. Useful when the AI keeps disagreeing
// with a candidate or the user decides a topic isn't memory-worthy.
func slashMemoryDiscard(w io.Writer, p paths, cfg config, arg string) error {
	arg = strings.TrimSpace(arg)
	if arg == "" {
		fmt.Fprintln(w, "[Focus] Usage: fg: memory discard <tempId|all>")
		return nil
	}
	maxAge := parseDuration(cfg.Memory.PendingMaxAge, 168*time.Hour)
	pq, err := memory.LoadPending(p.dataDir, maxAge)
	if err != nil || pq == nil {
		fmt.Fprintf(w, "[Focus] Failed to load pending queue: %v\n", err)
		return nil
	}
	if arg == "all" {
		n := pq.Clear()
		if err := pq.Save(p.dataDir); err != nil {
			fmt.Fprintf(w, "[Focus] Discarded %d in memory but save failed: %v\n", n, err)
			return nil
		}
		fmt.Fprintf(w, "[Focus] Discarded %d pending candidate(s).\n", n)
		return nil
	}
	if !pq.Remove(arg) {
		fmt.Fprintf(w, "[Focus] No pending candidate with tempId %q.\n", arg)
		return nil
	}
	if err := pq.Save(p.dataDir); err != nil {
		fmt.Fprintf(w, "[Focus] Removed in memory but save failed: %v\n", err)
		return nil
	}
	fmt.Fprintf(w, "[Focus] Discarded candidate %s.\n", arg)
	return nil
}

// shortID returns a truncated memory ID for display tables. Keeps tables
// from wrapping on narrow terminals without losing identifiability.
func shortID(id string) string {
	if len(id) <= 16 {
		return id
	}
	return id[:16] + "..."
}

// slashMemoryHealth reports manifest state, soft warnings (stale refs,
// large files), and counts. Read-only diagnostic — never modifies files.
func slashMemoryHealth(w io.Writer, p paths, e *tfidf.Engine) error {
	mf := loadMemoryManifest(p.memoryDir, e)
	fmt.Fprintln(w, "=== Memory Health ===")
	fmt.Fprintf(w, "  Directory:   %s\n", p.memoryDir)
	fmt.Fprintf(w, "  Manifest:    %d entries, schema v%s, rebuilt %s\n",
		len(mf.Entries), mf.SchemaVersion, formatRelTime(mf.RebuiltAt))
	fmt.Fprintf(w, "  Vocab hash:  %s\n", shortHash(mf.VocabHash))

	if len(mf.Entries) == 0 {
		fmt.Fprintln(w)
		fmt.Fprintln(w, "  No memories yet — health checks skipped.")
		return nil
	}

	cwd, _ := os.Getwd()
	totalRefs, missingRefs := 0, 0
	totalTouches := 0
	for _, e := range mf.Entries {
		totalTouches += e.TouchedBy
		for _, r := range e.Refs {
			totalRefs++
			abs := r
			if !filepath.IsAbs(r) && cwd != "" {
				abs = filepath.Join(cwd, r)
			}
			if _, err := os.Stat(abs); err != nil {
				missingRefs++
			}
		}
	}
	fmt.Fprintf(w, "  Touches:     %d (sum across all memories)\n", totalTouches)
	fmt.Fprintf(w, "  Refs:        %d total, %d missing on disk\n", totalRefs, missingRefs)
	if missingRefs > 0 {
		fmt.Fprintln(w, "  ⚠ Some refs no longer exist; affected memories will render them with (missing).")
	}
	fmt.Fprintln(w)
	fmt.Fprintln(w, "  (Pending candidate queue and promotion arrive in Session B.)")
	return nil
}

// formatRelTime returns a short human-readable age like "5m ago", "2.3h
// ago", "1.2d ago". Falls back to absolute timestamp for very old entries.
func formatRelTime(t time.Time) string {
	if t.IsZero() {
		return "(never)"
	}
	d := time.Since(t)
	switch {
	case d < time.Minute:
		return fmt.Sprintf("%ds ago", int(d.Seconds()))
	case d < time.Hour:
		return fmt.Sprintf("%dm ago", int(d.Minutes()))
	case d < 48*time.Hour:
		return fmt.Sprintf("%.1fh ago", d.Hours())
	case d < 30*24*time.Hour:
		return fmt.Sprintf("%.1fd ago", d.Hours()/24)
	default:
		return t.Format("2006-01-02")
	}
}

func shortHash(h string) string {
	if len(h) > 12 {
		return h[:12] + "..."
	}
	if h == "" {
		return "(unset)"
	}
	return h
}
