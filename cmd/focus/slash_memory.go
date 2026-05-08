package main

import (
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/kuandriy/focus-gate/internal/memory"
	"github.com/kuandriy/focus-gate/internal/persist"
	"github.com/kuandriy/focus-gate/internal/text"
	"github.com/kuandriy/focus-gate/internal/tfidf"
)

// memoryMutators lists the /focus memory subcommands that write to
// disk (pending queue, manifest, memory files, sources.json). Adding
// a new write-path subcommand without listing it here means it can
// race a concurrent hook — keep this list and the dispatch in sync.
var memoryMutators = map[string]bool{
	"discard":    true,
	"commit":     true,
	"reindex":    true,
	"migrate-v1": true,
	"forget":     true,
}

// sourceMutators are the /focus memory source subcommands that mutate
// sources.json. Tracked separately so the dispatcher can detect them
// from the second token, not the first.
var sourceMutators = map[string]bool{
	"attach":  true,
	"detach":  true,
	"enable":  true,
	"disable": true,
	"default": true,
}

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

	// Hold the same project-level lock the hook uses for any subcommand
	// known to mutate disk state. Read-only paths (list, show, pending,
	// promote, health, why, source list) skip the lock to keep the
	// inspector cheap. Without this, a /focus memory commit running in
	// one terminal can race a hook firing in another and either drop a
	// touch increment or write a half-rebuilt manifest.
	if memoryMutators[sub] || (sub == "source" && isSourceMutation(rest)) {
		lock, err := persist.Acquire(p.lockFile)
		if err != nil {
			logErr("acquire lock", err)
		}
		defer func() {
			if lock != nil {
				_ = lock.Release()
			}
		}()
	}

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
	case "reindex":
		return slashMemoryReindex(w, p, e, rest)
	case "migrate-v1":
		return slashMemoryMigrateV1(w, p, e)
	case "promote":
		return slashMemoryPromote(w, p, cfg, e, rest)
	case "commit":
		return slashMemoryCommit(w, p, cfg, e, rest)
	case "forget":
		return slashMemoryForget(w, p, e, rest)
	case "why":
		return slashMemoryWhy(w, p, cfg, e, rest)
	case "diff":
		return slashMemoryDiff(w, p, e, rest)
	case "source":
		return slashMemorySource(w, p, rest)
	default:
		fmt.Fprintf(w, "[Focus] Unknown memory subcommand: %q\n", sub)
		fmt.Fprintln(w, "Available: fg: memory [list | show <id> | diff <id> | pending | promote [tempId] | commit <tempId> <json> | discard <id|all> | forget <id> [--yes] | why \"prompt\" | health | reindex [--source <name>] | migrate-v1 | source ...]")
		return nil
	}
}

// isSourceMutation returns true when the `source` subcommand argument
// resolves to one of the registered mutators (attach/detach/enable/
// disable/default). Used by the slashMemory dispatcher to decide
// whether to acquire the hook's project lock before running.
func isSourceMutation(arg string) bool {
	sub, _ := splitArg(arg)
	return sourceMutators[sub]
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
		title := text.TruncateRunesWithSuffix(e.Title, 60, "...")
		fmt.Fprintf(w, "  %-26s  %-6d  %-12s  %s\n",
			e.ID, e.TouchedBy, formatRelTime(e.Updated), title)
	}
	return nil
}

// slashMemoryShow renders a single memory's frontmatter + body. ID
// matching is exact OR by prefix (so users can type the first 6 hex
// chars). Searches every attached source so a memory in a shared repo
// is just as reachable as a personal one.
func slashMemoryShow(w io.Writer, p paths, e *tfidf.Engine, idOrPrefix string) error {
	idOrPrefix = strings.TrimSpace(idOrPrefix)
	if idOrPrefix == "" {
		fmt.Fprintln(w, "[Focus] Usage: fg: memory show <id-or-prefix>")
		return nil
	}

	registry, _ := memory.LoadSources(p.dataDir, p.memoryDir)
	vocab := memory.NewVocabSnapshot(e)

	type hit struct {
		source string
		path   string
		entry  memory.IndexEntry
	}
	var matches []hit
	for _, s := range registry.Sources {
		mf, _ := memory.EnsureFresh(s.Path, vocab)
		for _, ent := range mf.Entries {
			if ent.ID == idOrPrefix || strings.HasPrefix(ent.ID, idOrPrefix) {
				matches = append(matches, hit{
					source: s.Name,
					path:   filepath.Join(s.Path, ent.Path),
					entry:  ent,
				})
			}
		}
	}
	if len(matches) == 0 {
		fmt.Fprintf(w, "[Focus] No memory matches %q across attached sources.\n", idOrPrefix)
		return nil
	}
	if len(matches) > 1 && matches[0].entry.ID != idOrPrefix {
		fmt.Fprintf(w, "[Focus] Ambiguous prefix %q — matches %d memories. Use the full ID:\n", idOrPrefix, len(matches))
		for _, h := range matches {
			fmt.Fprintf(w, "  [%s] %s — %s\n", h.source, h.entry.ID, h.entry.Title)
		}
		return nil
	}
	match := matches[0]

	mem, err := memory.ReadFile(match.path)
	if err != nil {
		fmt.Fprintf(w, "[Focus] Failed to read %s: %v\n", match.path, err)
		return nil
	}
	fmt.Fprintf(w, "=== %s ===\n", mem.ID)
	fmt.Fprintf(w, "  Source:    %s\n", match.source)
	fmt.Fprintf(w, "  Title:     %s\n", mem.Title)
	fmt.Fprintf(w, "  Path:      %s\n", match.path)
	fmt.Fprintf(w, "  Created:   %s\n", mem.Created.Format(time.RFC3339))
	fmt.Fprintf(w, "  Updated:   %s\n", mem.Updated.Format(time.RFC3339))
	fmt.Fprintf(w, "  TouchedBy: %d\n", mem.TouchedBy)
	if len(mem.Assets) > 0 {
		fmt.Fprintf(w, "  Assets:    %s\n", strings.Join(mem.Assets, ", "))
	}
	if len(mem.TimeMarkers) > 0 {
		fmt.Fprintf(w, "  TimeMarkers: %s\n", strings.Join(mem.TimeMarkers, ", "))
	}
	if len(mem.Topics) > 0 {
		topicStrs := make([]string, len(mem.Topics))
		for i, t := range mem.Topics {
			topicStrs[i] = fmt.Sprintf("%s@%.2f", t.Name, t.Weight)
		}
		fmt.Fprintf(w, "  Topics:    %s\n", strings.Join(topicStrs, ", "))
	}
	if len(mem.Interests) > 0 {
		interestStrs := make([]string, len(mem.Interests))
		for i, in := range mem.Interests {
			interestStrs[i] = fmt.Sprintf("%s@%.2f", in.Name, in.Weight)
		}
		fmt.Fprintf(w, "  Interests: %s\n", strings.Join(interestStrs, ", "))
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
		abs := text.TruncateRunesWithSuffix(c.RootAbstraction, 60, "...")
		target := c.SuggestedAction
		if c.SuggestedAction == "append" && c.SuggestedTargetID != "" {
			target = "append→" + shortID(c.SuggestedTargetID)
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

// looksLikeFilePath returns true when an asset string looks like a file
// path (contains a slash and a dotted extension). Used by Health to
// only stat-check things that could plausibly exist on disk; endpoint
// strings ("POST /api/...") and env vars ("JWT_PRIVATE_KEY") are
// asset-shaped but not file-shaped.
func looksLikeFilePath(s string) bool {
	if !strings.ContainsRune(s, '/') {
		return false
	}
	if strings.HasPrefix(s, "POST ") || strings.HasPrefix(s, "GET ") ||
		strings.HasPrefix(s, "PUT ") || strings.HasPrefix(s, "DELETE ") ||
		strings.HasPrefix(s, "PATCH ") {
		return false
	}
	dot := strings.LastIndexByte(s, '.')
	slash := strings.LastIndexByte(s, '/')
	return dot > slash
}

// shortID returns a truncated memory ID for display tables. Keeps tables
// from wrapping on narrow terminals without losing identifiability.
func shortID(id string) string {
	if len(id) <= 16 {
		return id
	}
	return text.TruncateRunesWithSuffix(id, 19, "...")
}

// slashMemoryPromote emits the Stage B prompt for a single pending
// candidate (or for every pending candidate if no tempId is supplied).
// The output is text the LLM reads and replies to with `fg: memory
// commit <tempId> <json>`.
func slashMemoryPromote(w io.Writer, p paths, cfg config, e *tfidf.Engine, arg string) error {
	tempID := strings.TrimSpace(arg)
	maxAge := parseDuration(cfg.Memory.PendingMaxAge, 168*time.Hour)
	pq, err := memory.LoadPending(p.dataDir, maxAge)
	if err != nil || pq == nil {
		fmt.Fprintf(w, "[Focus] Failed to load pending queue: %v\n", err)
		return nil
	}
	if len(pq.Candidates) == 0 {
		fmt.Fprintln(w, "[Focus] No pending candidates to promote.")
		return nil
	}
	// Compose the existing-memories view from every enabled source so the
	// LLM's append/create decision can land on a memory that lives in a
	// shared/attached source — not just the local "personal" one.
	registry, _ := memory.LoadSources(p.dataDir, p.memoryDir)
	vocab := memory.NewVocabSnapshot(e)
	manifests, _ := registry.LoadEnabledManifests(vocab)
	msi := memory.NewMultiSourceIndex(manifests...)

	emitted := 0
	for _, c := range pq.Sorted() {
		if tempID != "" && c.TempID != tempID {
			continue
		}
		fmt.Fprintln(w, memory.BuildStageBPrompt(c, msi, memory.PromoteOptions{
			DefaultSource: registry.Default,
		}))
		emitted++
		if tempID != "" {
			break
		}
	}
	if emitted == 0 {
		fmt.Fprintf(w, "[Focus] No pending candidate matches tempId %q.\n", tempID)
	}
	return nil
}

// slashMemoryCommit consumes a `fg: memory commit <tempId> <json>`
// payload, validates it against the multi-source index, and persists
// the resulting append/create. Discard simply removes the candidate.
//
// On validation failure the structured errors are printed and the
// candidate's CommitRetries counter is incremented; if the budget is
// exhausted, the candidate is left in the pending queue with a note.
func slashMemoryCommit(w io.Writer, p paths, cfg config, e *tfidf.Engine, arg string) error {
	tempID, jsonText := splitArg(arg)
	tempID = strings.TrimSpace(tempID)
	jsonText = strings.TrimSpace(jsonText)
	// JSON often comes wrapped in single quotes from the slash-command
	// transport — strip them.
	jsonText = strings.Trim(jsonText, "'")
	if tempID == "" || jsonText == "" {
		fmt.Fprintln(w, "[Focus] Usage: fg: memory commit <tempId> '<json>'")
		return nil
	}

	maxAge := parseDuration(cfg.Memory.PendingMaxAge, 168*time.Hour)
	pq, err := memory.LoadPending(p.dataDir, maxAge)
	if err != nil || pq == nil {
		fmt.Fprintf(w, "[Focus] Failed to load pending queue: %v\n", err)
		return nil
	}

	var cand *memory.Candidate
	for _, c := range pq.Candidates {
		if c.TempID == tempID {
			cand = c
			break
		}
	}
	if cand == nil {
		fmt.Fprintf(w, "[Focus] No pending candidate with tempId %q.\n", tempID)
		return nil
	}

	payload, parseErrs := memory.ParseCommitJSON([]byte(jsonText))
	if len(parseErrs) > 0 {
		bumpRetry(pq, cand, cfg, w)
		emitCommitErrors(w, parseErrs)
		_ = pq.Save(p.dataDir)
		return nil
	}

	registry, _ := memory.LoadSources(p.dataDir, p.memoryDir)
	vocab := memory.NewVocabSnapshot(e)
	manifests, _ := registry.LoadEnabledManifests(vocab)
	msi := memory.NewMultiSourceIndex(manifests...)
	if errs := memory.ValidateCommit(payload, msi); len(errs) > 0 {
		bumpRetry(pq, cand, cfg, w)
		emitCommitErrors(w, errs)
		_ = pq.Save(p.dataDir)
		return nil
	}

	ctx := memory.CommitContext{
		SourceDirs:    registry.SourceDirs(),
		DefaultSource: registry.Default,
		Vocab:         vocab,
		Index:         msi,
	}
	res, err := memory.ApplyCommit(payload, ctx)
	if err != nil {
		fmt.Fprintf(w, "[Focus] Apply failed: %v\n", err)
		// Apply errors are bugs (e.g. file IO failed mid-write) —
		// bumping retry count here would just make a real bug look
		// like a transient validation issue. Leave the queue alone.
		return nil
	}

	pq.Remove(tempID)
	if err := pq.Save(p.dataDir); err != nil {
		fmt.Fprintf(w, "[Focus] Saved memory but failed to update pending queue: %v\n", err)
	}

	switch res.Action {
	case memory.CommitActionDiscard:
		fmt.Fprintf(w, "[Focus] Discarded candidate %s.\n", tempID)
	case memory.CommitActionAppend:
		fmt.Fprintf(w, "[Focus] Appended chapter to %s [%s] (now %d chapters) at %s\n",
			res.Memory.ID, res.Source, res.Memory.Chapters, res.FilePath)
	case memory.CommitActionCreate:
		fmt.Fprintf(w, "[Focus] Created memory %s [%s] at %s\n", res.Memory.ID, res.Source, res.FilePath)
	}

	// Force a manifest rebuild on the affected source so the new or
	// updated memory becomes surfaceable on the very next prompt.
	if res.Action != memory.CommitActionDiscard {
		if dir, ok := registry.SourceDirs()[res.Source]; ok {
			fresh, _ := memory.Load(dir)
			_ = fresh.Rebuild(dir, vocab)
			_ = fresh.Save(dir)
		}
	}
	return nil
}

// slashMemorySource dispatches the source registry subcommands. Each
// subcommand mutates `<dataDir>/sources.json` directly; callers don't
// touch config.json.
func slashMemorySource(w io.Writer, p paths, arg string) error {
	sub, rest := splitArg(arg)
	switch sub {
	case "", "list":
		return slashMemorySourceList(w, p)
	case "attach":
		return slashMemorySourceAttach(w, p, rest)
	case "detach":
		return slashMemorySourceDetach(w, p, rest)
	case "enable":
		return slashMemorySourceSetEnabled(w, p, rest, true)
	case "disable":
		return slashMemorySourceSetEnabled(w, p, rest, false)
	case "default":
		return slashMemorySourceDefault(w, p, rest)
	default:
		fmt.Fprintf(w, "[Focus] Unknown source subcommand: %q\n", sub)
		fmt.Fprintln(w, "Available: fg: memory source [list | attach <name> <path> [--read-only] | detach <name> | enable <name> | disable <name> | default <name>]")
		return nil
	}
}

func slashMemorySourceList(w io.Writer, p paths) error {
	r, err := memory.LoadSources(p.dataDir, p.memoryDir)
	if err != nil {
		fmt.Fprintf(w, "[Focus] Load sources: %v\n", err)
	}
	fmt.Fprintln(w, "=== Memory Sources ===")
	fmt.Fprintf(w, "  Default: %s\n\n", r.Default)
	fmt.Fprintf(w, "  %-12s  %-7s  %-8s  %-5s  %s\n", "NAME", "ENABLED", "WRITABLE", "COUNT", "PATH")
	for _, s := range r.Sources {
		count := sourceMemoryCount(s.Path)
		fmt.Fprintf(w, "  %-12s  %-7v  %-8v  %-5d  %s\n",
			s.Name, s.Enabled, s.Writable, count, s.Path)
	}
	return nil
}

// sourceMemoryCount returns the number of `.md` files in dir, excluding
// `.v1.bak` migration backups. Read-only; never errors out (a missing
// directory is reported as 0).
func sourceMemoryCount(dir string) int {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return 0
	}
	n := 0
	for _, e := range entries {
		if e.IsDir() {
			continue
		}
		name := e.Name()
		if !strings.HasSuffix(name, ".md") {
			continue
		}
		if strings.HasSuffix(name, ".v1.bak") {
			continue
		}
		n++
	}
	return n
}

func slashMemorySourceAttach(w io.Writer, p paths, arg string) error {
	parts := strings.Fields(arg)
	if len(parts) < 2 {
		fmt.Fprintln(w, "[Focus] Usage: fg: memory source attach <name> <path> [--read-only]")
		return nil
	}
	name := parts[0]
	path := parts[1]
	writable := true
	for _, flag := range parts[2:] {
		if flag == "--read-only" {
			writable = false
		}
	}
	r, _ := memory.LoadSources(p.dataDir, p.memoryDir)
	if err := r.Attach(name, path, writable); err != nil {
		fmt.Fprintf(w, "[Focus] Attach failed: %v\n", err)
		return nil
	}
	if err := r.Save(p.dataDir); err != nil {
		fmt.Fprintf(w, "[Focus] Saved in memory but persist failed: %v\n", err)
		return nil
	}
	fmt.Fprintf(w, "[Focus] Attached source %q at %s (writable=%v)\n", name, path, writable)
	return nil
}

func slashMemorySourceDetach(w io.Writer, p paths, name string) error {
	name = strings.TrimSpace(name)
	if name == "" {
		fmt.Fprintln(w, "[Focus] Usage: fg: memory source detach <name>")
		return nil
	}
	r, _ := memory.LoadSources(p.dataDir, p.memoryDir)
	if err := r.Detach(name); err != nil {
		fmt.Fprintf(w, "[Focus] Detach failed: %v\n", err)
		return nil
	}
	if err := r.Save(p.dataDir); err != nil {
		fmt.Fprintf(w, "[Focus] Detach in memory but persist failed: %v\n", err)
		return nil
	}
	fmt.Fprintf(w, "[Focus] Detached source %q.\n", name)
	return nil
}

func slashMemorySourceSetEnabled(w io.Writer, p paths, name string, enable bool) error {
	name = strings.TrimSpace(name)
	if name == "" {
		verb := "enable"
		if !enable {
			verb = "disable"
		}
		fmt.Fprintf(w, "[Focus] Usage: fg: memory source %s <name>\n", verb)
		return nil
	}
	r, _ := memory.LoadSources(p.dataDir, p.memoryDir)
	var err error
	if enable {
		err = r.Enable(name)
	} else {
		err = r.Disable(name)
	}
	if err != nil {
		fmt.Fprintf(w, "[Focus] %v\n", err)
		return nil
	}
	if err := r.Save(p.dataDir); err != nil {
		fmt.Fprintf(w, "[Focus] State changed in memory but persist failed: %v\n", err)
		return nil
	}
	state := "enabled"
	if !enable {
		state = "disabled"
	}
	fmt.Fprintf(w, "[Focus] Source %q %s.\n", name, state)
	return nil
}

func slashMemorySourceDefault(w io.Writer, p paths, name string) error {
	name = strings.TrimSpace(name)
	if name == "" {
		fmt.Fprintln(w, "[Focus] Usage: fg: memory source default <name>")
		return nil
	}
	r, _ := memory.LoadSources(p.dataDir, p.memoryDir)
	if err := r.SetDefault(name); err != nil {
		fmt.Fprintf(w, "[Focus] %v\n", err)
		return nil
	}
	if err := r.Save(p.dataDir); err != nil {
		fmt.Fprintf(w, "[Focus] Default changed in memory but persist failed: %v\n", err)
		return nil
	}
	fmt.Fprintf(w, "[Focus] Default source set to %q.\n", name)
	return nil
}

// bumpRetry increments the candidate's CommitRetries counter and tells
// the user when the budget is reached. Save happens at the call site.
func bumpRetry(pq *memory.PendingQueue, cand *memory.Candidate, cfg config, w io.Writer) {
	cand.CommitRetries++
	budget := cfg.Memory.CommitRetries
	if budget <= 0 {
		budget = 2
	}
	if cand.CommitRetries >= budget {
		fmt.Fprintf(w, "[Focus] Candidate %s has reached %d/%d retry attempts. Inspect with `fg: memory pending` or remove with `fg: memory discard %s`.\n",
			cand.TempID, cand.CommitRetries, budget, cand.TempID)
	} else {
		fmt.Fprintf(w, "[Focus] Validation failed (%d/%d attempts). Fix the JSON and retry.\n",
			cand.CommitRetries, budget)
	}
}

// emitCommitErrors prints structured commit errors in the field/reason/
// hint format the LLM expects.
func emitCommitErrors(w io.Writer, errs []memory.CommitError) {
	for _, e := range errs {
		if e.Hint != "" {
			fmt.Fprintf(w, "  ! %s — %s (hint: %s)\n", e.Field, e.Reason, e.Hint)
		} else {
			fmt.Fprintf(w, "  ! %s — %s\n", e.Field, e.Reason)
		}
	}
}

// slashMemoryForget deletes a memory file plus its manifest entry from
// every writable source that contains the given ID. Idempotent: a
// missing ID is reported, not an error. The "I really mean it"
// confirmation gate lives in the slash command surface — typing
// `--yes` after the id confirms; without it, only a dry-run preview
// is emitted.
//
// Usage:
//
//	fg: memory forget <id-or-prefix>          # dry run, shows what would happen
//	fg: memory forget <id-or-prefix> --yes    # actually delete
func slashMemoryForget(w io.Writer, p paths, e *tfidf.Engine, arg string) error {
	fields := strings.Fields(arg)
	if len(fields) == 0 {
		fmt.Fprintln(w, "[Focus] Usage: fg: memory forget <id-or-prefix> [--yes]")
		return nil
	}
	target := fields[0]
	confirm := false
	for _, f := range fields[1:] {
		if f == "--yes" {
			confirm = true
		}
	}

	registry, _ := memory.LoadSources(p.dataDir, p.memoryDir)
	vocab := memory.NewVocabSnapshot(e)

	// Find every (source, file path, manifest entry) tuple that matches.
	type hit struct {
		source string
		path   string
		entry  memory.IndexEntry
	}
	var hits []hit
	for _, s := range registry.Sources {
		mf, _ := memory.EnsureFresh(s.Path, vocab)
		for _, ent := range mf.Entries {
			if ent.ID == target || strings.HasPrefix(ent.ID, target) {
				hits = append(hits, hit{
					source: s.Name,
					path:   filepath.Join(s.Path, ent.Path),
					entry:  ent,
				})
			}
		}
	}
	if len(hits) == 0 {
		fmt.Fprintf(w, "[Focus] No memory matches %q across attached sources.\n", target)
		return nil
	}
	if len(hits) > 1 && target != hits[0].entry.ID {
		fmt.Fprintf(w, "[Focus] Prefix %q matches %d memories. Use a longer prefix or full ID:\n", target, len(hits))
		for _, h := range hits {
			fmt.Fprintf(w, "  [%s] %s — %s\n", h.source, h.entry.ID, h.entry.Title)
		}
		return nil
	}

	if !confirm {
		fmt.Fprintln(w, "[Focus] Dry run — pass --yes to actually delete:")
		for _, h := range hits {
			fmt.Fprintf(w, "  would remove [%s] %s (%s)\n", h.source, h.entry.ID, h.path)
		}
		return nil
	}

	// Confirmed — delete file, drop manifest entry, rebuild inverted
	// indexes immediately so the next surface call doesn't accidentally
	// touch a phantom ID. Read-only sources are skipped with a warning.
	for _, h := range hits {
		s, _ := registry.Get(h.source)
		if s != nil && !s.Writable {
			fmt.Fprintf(w, "  ! [%s] %s — source is read-only; skipped.\n", h.source, h.entry.ID)
			continue
		}
		if err := os.Remove(h.path); err != nil && !os.IsNotExist(err) {
			fmt.Fprintf(w, "  ! [%s] remove %s: %v\n", h.source, h.path, err)
			continue
		}
		// Backups (.v1.bak from a prior migration) sit beside the file.
		// Remove them too — keeping a backup of a memory the user just
		// asked to forget defeats the purpose.
		_ = os.Remove(h.path + ".v1.bak")

		mf, _ := memory.Load(s.Path)
		if mf.Remove(h.entry.ID) {
			mf.RebuildInvertedIndexes()
			if err := mf.Save(s.Path); err != nil {
				fmt.Fprintf(w, "  ! [%s] update manifest: %v\n", h.source, err)
				continue
			}
		}
		fmt.Fprintf(w, "  ✓ removed [%s] %s — %s\n", h.source, h.entry.ID, h.entry.Title)
	}
	return nil
}

// slashMemoryReindex forces a manifest rebuild against one or every
// enabled source. Useful after a hand-authored .md file is added,
// after migration, or whenever the user suspects an index has drifted.
//
// Usage:
//
//	fg: memory reindex                    # rebuild every enabled source
//	fg: memory reindex --source <name>    # rebuild just that source
func slashMemoryReindex(w io.Writer, p paths, e *tfidf.Engine, arg string) error {
	wanted := parseFlag(arg, "--source")
	registry, _ := memory.LoadSources(p.dataDir, p.memoryDir)
	vocab := memory.NewVocabSnapshot(e)

	targets := registry.EnabledSources()
	if wanted != "" {
		s, ok := registry.Get(wanted)
		if !ok {
			fmt.Fprintf(w, "[Focus] Unknown source %q. See `fg: memory source list`.\n", wanted)
			return nil
		}
		targets = []memory.Source{*s}
	}
	if len(targets) == 0 {
		fmt.Fprintln(w, "[Focus] No enabled sources to reindex.")
		return nil
	}

	for _, s := range targets {
		mf, err := memory.Load(s.Path)
		if err != nil {
			fmt.Fprintf(w, "[Focus] Load manifest [%s]: %v (continuing with empty)\n", s.Name, err)
			mf = memory.NewManifest()
		}
		mf.Source = s.Name
		errs := mf.Rebuild(s.Path, vocab)
		// Stamp source name on every entry — Rebuild stamps them with
		// whatever was on the manifest at load time, but the registry's
		// canonical source name takes precedence.
		for i := range mf.Entries {
			mf.Entries[i].Source = s.Name
		}
		if mf.Dirty() {
			if saveErr := mf.Save(s.Path); saveErr != nil {
				errs = append(errs, fmt.Errorf("save manifest: %w", saveErr))
			}
		}
		fmt.Fprintf(w, "[Focus] Reindexed %d memorie(s) in %s [%s].\n", len(mf.Entries), s.Path, s.Name)
		if len(mf.ByAsset) > 0 || len(mf.ByInterest) > 0 || len(mf.ByTopic) > 0 {
			fmt.Fprintf(w, "  Inverted indexes: %d assets, %d interests, %d topics\n",
				len(mf.ByAsset), len(mf.ByInterest), len(mf.ByTopic))
		}
		for _, err := range errs {
			fmt.Fprintf(w, "  ! %v\n", err)
		}
	}
	return nil
}

// parseFlag extracts a `--name <value>` pair from a free-form argument
// string. Returns "" if the flag is absent or has no value following.
// Tolerant of `--name=value` form too.
func parseFlag(arg, name string) string {
	fields := strings.Fields(arg)
	for i, f := range fields {
		if f == name && i+1 < len(fields) {
			return fields[i+1]
		}
		if strings.HasPrefix(f, name+"=") {
			return strings.TrimPrefix(f, name+"=")
		}
	}
	return ""
}

// slashMemoryMigrateV1 walks the memory directory and converts every
// schemaVersion=1 file to v2 in place, preserving the original at
// `<file>.v1.bak`. After all migrations succeed the manifest is
// rebuilt so v2 entries are immediately surfaceable.
func slashMemoryMigrateV1(w io.Writer, p paths, e *tfidf.Engine) error {
	entries, err := os.ReadDir(p.memoryDir)
	if err != nil {
		fmt.Fprintf(w, "[Focus] Read %s: %v\n", p.memoryDir, err)
		return nil
	}
	vocab := memory.NewVocabSnapshot(e)
	migrated, skipped, failed := 0, 0, 0
	for _, ent := range entries {
		if ent.IsDir() || !strings.HasSuffix(ent.Name(), ".md") {
			continue
		}
		path := filepath.Join(p.memoryDir, ent.Name())
		data, err := os.ReadFile(path)
		if err != nil {
			fmt.Fprintf(w, "  ! read %s: %v\n", ent.Name(), err)
			failed++
			continue
		}
		if memory.IsV2File(data) {
			skipped++
			continue
		}
		if _, err := memory.MigrateV1FileToV2(path, vocab); err != nil {
			fmt.Fprintf(w, "  ! migrate %s: %v\n", ent.Name(), err)
			failed++
			continue
		}
		migrated++
		fmt.Fprintf(w, "  ✓ migrated %s (backup at %s.v1.bak)\n", ent.Name(), ent.Name())
	}
	fmt.Fprintf(w, "[Focus] Migration done: %d migrated, %d already v2, %d failed.\n",
		migrated, skipped, failed)
	if migrated > 0 {
		// Rebuild the manifest so the newly-migrated v2 files become
		// surfaceable immediately.
		mf, _ := memory.Load(p.memoryDir)
		_ = mf.Rebuild(p.memoryDir, vocab)
		_ = mf.Save(p.memoryDir)
	}
	return nil
}

// slashMemoryDiff renders a per-chapter timeline of an existing memory:
// for each chapter, the date, title, time-frame, the assets/topics/
// interests it introduced (de-duplicated against earlier chapters so the
// diff actually shows what's new), and a one-line snippet of the What
// section. Read-only — never mutates state.
//
// Usage: /focus memory diff <id-or-prefix>
//
// Closes the U-2 gap: users can see a story's evolution without
// reading the full Markdown file.
func slashMemoryDiff(w io.Writer, p paths, e *tfidf.Engine, idOrPrefix string) error {
	idOrPrefix = strings.TrimSpace(idOrPrefix)
	if idOrPrefix == "" {
		fmt.Fprintln(w, "[Focus] Usage: /focus memory diff <id-or-prefix>")
		return nil
	}

	registry, _ := memory.LoadSources(p.dataDir, p.memoryDir)
	vocab := memory.NewVocabSnapshot(e)

	type hit struct {
		source string
		path   string
		entry  memory.IndexEntry
	}
	var matches []hit
	for _, s := range registry.Sources {
		mf, _ := memory.EnsureFresh(s.Path, vocab)
		for _, ent := range mf.Entries {
			if ent.ID == idOrPrefix || strings.HasPrefix(ent.ID, idOrPrefix) {
				matches = append(matches, hit{
					source: s.Name,
					path:   filepath.Join(s.Path, ent.Path),
					entry:  ent,
				})
			}
		}
	}
	if len(matches) == 0 {
		fmt.Fprintf(w, "[Focus] No memory matches %q across attached sources.\n", idOrPrefix)
		return nil
	}
	if len(matches) > 1 && matches[0].entry.ID != idOrPrefix {
		fmt.Fprintf(w, "[Focus] Ambiguous prefix %q — matches %d memories. Use the full ID:\n", idOrPrefix, len(matches))
		for _, h := range matches {
			fmt.Fprintf(w, "  [%s] %s — %s\n", h.source, h.entry.ID, h.entry.Title)
		}
		return nil
	}
	match := matches[0]

	mem, err := memory.ReadFile(match.path)
	if err != nil {
		fmt.Fprintf(w, "[Focus] Failed to read %s: %v\n", match.path, err)
		return nil
	}
	if len(mem.ChaptersList) == 0 {
		fmt.Fprintf(w, "[Focus] %s has no chapters.\n", mem.ID)
		return nil
	}

	fmt.Fprintf(w, "=== %s — %s [%s] ===\n", mem.ID, mem.Title, match.source)
	fmt.Fprintf(w, "  %d chapter(s)\n\n", len(mem.ChaptersList))

	// Track set of items seen in earlier chapters so each per-chapter
	// section shows what's *newly introduced*, not the running union.
	seenAssets := map[string]bool{}
	seenInterests := map[string]bool{}
	seenTopics := map[string]bool{}
	for i, ch := range mem.ChaptersList {
		date := "(no date)"
		if !ch.Date.IsZero() {
			date = ch.Date.UTC().Format("2006-01-02")
		}
		title := ch.Title
		if title == "" {
			title = "(untitled)"
		}
		fmt.Fprintf(w, "  Chapter %d — %s — %s\n", i+1, date, title)
		if tm := strings.TrimSpace(ch.TimeMarker); tm != "" {
			fmt.Fprintf(w, "    time marker: %s\n", tm)
		}

		newAssets := diffNew(ch.Assets, seenAssets)
		newInterests := diffNew(ch.Interests, seenInterests)
		newTopics := diffNew(ch.Topics, seenTopics)
		if len(newAssets) > 0 {
			fmt.Fprintf(w, "    + assets:    %s\n", strings.Join(newAssets, ", "))
		}
		if len(newTopics) > 0 {
			fmt.Fprintf(w, "    + topics:    %s\n", strings.Join(newTopics, ", "))
		}
		if len(newInterests) > 0 {
			fmt.Fprintf(w, "    + interests: %s\n", strings.Join(newInterests, ", "))
		}
		if i == 0 && len(newAssets)+len(newTopics)+len(newInterests) == 0 {
			fmt.Fprintln(w, "    (no metadata recorded)")
		} else if i > 0 && len(newAssets)+len(newTopics)+len(newInterests) == 0 {
			fmt.Fprintln(w, "    (no new metadata; chapter extends prior coverage)")
		}
		// One-line What snippet — keep diffs compact; users who want the
		// full prose run `/focus memory show`.
		whatSnippet := strings.Join(strings.Fields(ch.What), " ")
		whatSnippet = text.TruncateRunesWithSuffix(whatSnippet, 100, "…")
		fmt.Fprintf(w, "    what: %s\n", whatSnippet)
		if i+1 < len(mem.ChaptersList) {
			fmt.Fprintln(w)
		}
	}
	return nil
}

// diffNew returns the entries from `items` that have not yet been
// recorded in `seen`. Comparison is case-insensitive; entries are
// reported in the casing of their first occurrence. Mutates `seen` so
// successive calls accumulate.
func diffNew(items []string, seen map[string]bool) []string {
	out := make([]string, 0, len(items))
	for _, it := range items {
		key := strings.ToLower(strings.TrimSpace(it))
		if key == "" || seen[key] {
			continue
		}
		seen[key] = true
		out = append(out, it)
	}
	return out
}

// slashMemoryWhy answers "why isn't memory X surfacing?". It runs the
// full scoring pipeline against the supplied prompt at threshold=0 so
// every entry's per-tier scores are visible. The configured threshold
// is shown alongside so the user sees which entries crossed the bar
// vs. fell short, with a marker on each row.
//
// Read-only — never touches manifests, never mutates state. Closes the
// "I know I have a memory about this; what's it scoring?" feedback gap.
func slashMemoryWhy(w io.Writer, p paths, cfg config, e *tfidf.Engine, prompt string) error {
	prompt = strings.TrimSpace(prompt)
	if prompt == "" {
		fmt.Fprintln(w, "[Focus] Usage: /focus memory why \"prompt text\"")
		return nil
	}

	registry, _ := memory.LoadSources(p.dataDir, p.memoryDir)
	vocab := memory.NewVocabSnapshot(e)
	manifests, errs := registry.LoadEnabledManifests(vocab)
	for _, err := range errs {
		fmt.Fprintf(w, "  ! %v\n", err)
	}
	msi := memory.NewMultiSourceIndex(manifests...)
	if msi.Empty() {
		fmt.Fprintln(w, "[Focus] No memories in any enabled source.")
		return nil
	}

	// Reproduce the runtime's scoring exactly — threshold=0 so we see
	// what *would* surface at any threshold, big TopK so nothing is
	// truncated, FrequencyBonus matched to runtime so the displayed
	// scores reflect what the hook actually injects.
	scoreCfg := memory.SurfaceConfig{
		Enabled:           true,
		Threshold:         0,
		TopK:              1000,
		MaxBlockChars:     1 << 20,
		AssetWeight:       cfg.Memory.Weights.Asset,
		TopicWeight:       cfg.Memory.Weights.Topic,
		InterestWeight:    cfg.Memory.Weights.Interest,
		FingerprintWeight: cfg.Memory.Weights.Fingerprint,
		FrequencyBonus:    cfg.Memory.FrequencyBonus,
	}

	// Tokenize via the live engine so the scoring uses the same vocab
	// the runtime would. We don't mutate state — just classify in
	// memory.
	promptVec := e.Vectorize(prompt)
	pv := make(tfidf.Vector, 0, len(promptVec))
	pv = append(pv, promptVec...)
	result := memory.Surface(memory.SurfaceInputs{
		PromptText: prompt,
		PromptVec:  pv,
		Vocab:      vocab,
		Index:      msi,
	}, scoreCfg)

	threshold := cfg.Memory.SurfaceThreshold
	if threshold <= 0 {
		threshold = 0.35
	}
	fmt.Fprintln(w, "=== Memory Why ===")
	fmt.Fprintf(w, "  Prompt:    %q\n", prompt)
	fmt.Fprintf(w, "  Threshold: %.3f\n", threshold)
	fmt.Fprintf(w, "  Scored:    %d candidate(s)\n\n", len(result.Selected))

	if len(result.Selected) == 0 {
		fmt.Fprintln(w, "  No memory had any tier hit. Asset extraction returned nothing,")
		fmt.Fprintln(w, "  prompt vector had no overlap with any fingerprint, and no topic")
		fmt.Fprintln(w, "  or interest cosine cleared zero. Tune `memory.surfaceThreshold`")
		fmt.Fprintln(w, "  or seed assets/topics/interests on the relevant memory.")
		return nil
	}

	for _, c := range result.Selected {
		marker := "  "
		if c.Score >= threshold {
			marker = "✓ "
		}
		fmt.Fprintf(w, "%s%s [%s] score=%.3f  %s\n",
			marker, c.Entry.ID, c.Entry.Source, c.Score, c.Entry.Title)
		for _, r := range c.Reasons {
			detail := r.Detail
			if detail == "" {
				detail = "(no detail)"
			}
			fmt.Fprintf(w, "      %-12s %.3f  %s\n", r.Tier, r.Score, detail)
		}
	}
	fmt.Fprintln(w)
	fmt.Fprintln(w, "  ✓ = would surface at the current threshold.")
	return nil
}

// slashMemoryHealth reports manifest state per attached source — counts,
// vocab freshness, missing-asset stats. Read-only diagnostic; never
// modifies files. Disabled sources are listed with a `(disabled)` tag
// so the user sees them but with all stats blank.
func slashMemoryHealth(w io.Writer, p paths, e *tfidf.Engine) error {
	registry, _ := memory.LoadSources(p.dataDir, p.memoryDir)
	vocab := memory.NewVocabSnapshot(e)
	cwd, _ := os.Getwd()

	fmt.Fprintln(w, "=== Memory Health ===")
	fmt.Fprintf(w, "  Default source: %s\n", registry.Default)
	fmt.Fprintf(w, "  Sources: %d total, %d enabled\n",
		len(registry.Sources), len(registry.EnabledSources()))
	fmt.Fprintln(w)

	overall := healthTotals{}
	for _, s := range registry.Sources {
		fmt.Fprintf(w, "  [%s] %s\n", s.Name, s.Path)
		if !s.Enabled {
			fmt.Fprintln(w, "    (disabled — skipped)")
			fmt.Fprintln(w)
			continue
		}
		mf, errs := memory.EnsureFresh(s.Path, vocab)
		for _, err := range errs {
			fmt.Fprintf(w, "    ! %v\n", err)
		}
		fmt.Fprintf(w, "    Manifest:   %d entries, schema v%s, rebuilt %s\n",
			len(mf.Entries), mf.SchemaVersion, formatRelTime(mf.RebuiltAt))
		fmt.Fprintf(w, "    Vocab hash: %s\n", shortHash(mf.VocabHash))
		if mf.VocabHash != "" && mf.VocabHash != vocab.Hash {
			fmt.Fprintln(w, "    ⚠ Vocab hash differs from current engine; surfacing may underrank.")
		}
		if len(mf.Entries) == 0 {
			fmt.Fprintln(w, "    (no memories)")
			fmt.Fprintln(w)
			continue
		}
		stats := perSourceHealth(mf, cwd)
		fmt.Fprintf(w, "    Touches:    %d (sum)\n", stats.touches)
		fmt.Fprintf(w, "    Assets:     %d total, %d missing on disk\n",
			stats.assets, stats.missingAssets)
		if stats.missingAssets > 0 {
			fmt.Fprintln(w, "    ⚠ Some asset paths no longer exist; their memories will render them with (missing).")
		}
		overall.add(stats)
		fmt.Fprintln(w)
	}

	if overall.entries > 0 {
		fmt.Fprintln(w, "  --- Totals across enabled sources ---")
		fmt.Fprintf(w, "    Memories: %d  Touches: %d  Assets: %d (%d missing)\n",
			overall.entries, overall.touches, overall.assets, overall.missingAssets)
	}
	return nil
}

// healthTotals aggregates per-source counts so the trailing summary can
// roll them up without re-iterating manifests.
type healthTotals struct {
	entries       int
	touches       int
	assets        int
	missingAssets int
}

func (a *healthTotals) add(b healthTotals) {
	a.entries += b.entries
	a.touches += b.touches
	a.assets += b.assets
	a.missingAssets += b.missingAssets
}

// perSourceHealth walks one manifest's entries and tallies the stats
// the health report displays.
func perSourceHealth(mf *memory.Manifest, cwd string) healthTotals {
	out := healthTotals{entries: len(mf.Entries)}
	for _, e := range mf.Entries {
		out.touches += e.TouchedBy
		for _, a := range e.Assets {
			out.assets++
			if !looksLikeFilePath(a) {
				continue
			}
			abs := a
			if !filepath.IsAbs(a) && cwd != "" {
				abs = filepath.Join(cwd, a)
			}
			if _, err := os.Stat(abs); err != nil {
				out.missingAssets++
			}
		}
	}
	return out
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
	if h == "" {
		return "(unset)"
	}
	return text.TruncateRunesWithSuffix(h, 15, "...")
}
