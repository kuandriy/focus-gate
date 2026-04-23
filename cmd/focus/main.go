package main

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/kuandriy/focus-gate/internal/forest"
	"github.com/kuandriy/focus-gate/internal/gate"
	"github.com/kuandriy/focus-gate/internal/guide"
	"github.com/kuandriy/focus-gate/internal/memory"
	"github.com/kuandriy/focus-gate/internal/persist"
	"github.com/kuandriy/focus-gate/internal/text"
	"github.com/kuandriy/focus-gate/internal/tfidf"
)

// paths resolves data file paths relative to the binary location.
type paths struct {
	dataDir    string
	intentFile string
	engineFile string
	guideFile  string
	lockFile   string
	configFile string
	memoryDir  string // <dataDir>/memories/ — long-term memory file store
}

// resolveDataDir determines the state directory using this priority:
//  1. --data-dir CLI flag (explicit override)
//  2. FOCUS_GATE_DATA_DIR environment variable
//  3. Per-project isolation: ~/.focus-gate/<sha256(cwd)[:12]>/
//
// Per-project isolation (option 3) prevents cross-contamination when the same
// binary is used across multiple projects. Each working directory gets its own
// state namespace under ~/.focus-gate/. Note: if a project directory is renamed
// or moved, the hash changes and the old state directory becomes orphaned under
// ~/.focus-gate/ — use --reset or delete the old slug directory manually.
func resolveDataDir() string {
	// 1. CLI flag
	if dir := flagValue(os.Args, "--data-dir"); dir != "" {
		return dir
	}

	// 2. Environment variable
	if dir := os.Getenv("FOCUS_GATE_DATA_DIR"); dir != "" {
		return dir
	}

	// 3. Per-project isolation based on CWD hash
	cwd, err := os.Getwd()
	if err != nil {
		// Fallback: relative to binary (legacy behavior)
		exe, _ := os.Executable()
		return filepath.Join(filepath.Dir(exe), "data")
	}

	home, err := os.UserHomeDir()
	if err != nil {
		exe, _ := os.Executable()
		return filepath.Join(filepath.Dir(exe), "data")
	}

	h := sha256.Sum256([]byte(cwd))
	slug := hex.EncodeToString(h[:])[:12]
	return filepath.Join(home, ".focus-gate", slug)
}

func resolvePaths() paths {
	dataDir := resolveDataDir()

	return paths{
		dataDir:    dataDir,
		intentFile: filepath.Join(dataDir, "intent.json"),
		engineFile: filepath.Join(dataDir, "engine.json"),
		guideFile:  filepath.Join(dataDir, "guide.json"),
		lockFile:   filepath.Join(dataDir, ".lock"),
		configFile: resolveConfigFile(),
		memoryDir:  filepath.Join(dataDir, "memories"),
	}
}

// resolveConfigFile picks the configuration file to load, in priority order:
//  1. ./.focus-gate.json — per-project override (current working directory).
//  2. $FOCUS_GATE_CONFIG — explicit path, useful for CI or non-standard layouts.
//  3. config.json alongside the binary — global fallback shared across projects.
//
// The first path that exists on disk wins. If none exist, we return the global
// path anyway so a missing-file Load silently yields defaults (same as before).
func resolveConfigFile() string {
	if cwd, err := os.Getwd(); err == nil {
		projectCfg := filepath.Join(cwd, ".focus-gate.json")
		if _, err := os.Stat(projectCfg); err == nil {
			return projectCfg
		}
	}

	if envCfg := os.Getenv("FOCUS_GATE_CONFIG"); envCfg != "" {
		return envCfg
	}

	exe, err := os.Executable()
	if err != nil {
		exe = "."
	}
	return filepath.Join(filepath.Dir(exe), "config.json")
}

// flagValue returns the value of a --key=value or --key value flag.
func flagValue(args []string, flag string) string {
	for i, a := range args {
		if a == flag && i+1 < len(args) {
			return args[i+1]
		}
		if strings.HasPrefix(a, flag+"=") {
			return a[len(flag)+1:]
		}
	}
	return ""
}

// config matches the JSON config file structure.
type config struct {
	MemorySize int     `json:"memorySize"`
	DecayRate  float64 `json:"decayRate"`
	Similarity struct {
		Extend float64 `json:"extend"`
		Branch float64 `json:"branch"`
	} `json:"similarity"`
	ContextLimit    int     `json:"contextLimit"`
	BubbleUpTerms   int     `json:"bubbleUpTerms"`
	MaxRefsPerNode  int     `json:"maxRefsPerNode"`
	GuideSize       int     `json:"guideSize"`
	SessionTimeout  float64 `json:"sessionTimeout"`  // hours; 0 = disabled
	MergeSimilarity float64 `json:"mergeSimilarity"` // threshold for cluster merging; 0 = disabled
	TypoTolerance   struct {
		Enabled          bool `json:"enabled"`
		MaxDistance      int  `json:"maxDistance"`
		MinWordLen       int  `json:"minWordLen"`
		MinEstablishedDF int  `json:"minEstablishedDF"`
	} `json:"typoTolerance"`
	Memory struct {
		Enabled            bool     `json:"enabled"`
		Dir                string   `json:"dir"`
		SurfaceThreshold   float64  `json:"surfaceThreshold"`
		TopK               int      `json:"topK"`
		MaxBlockChars      int      `json:"maxBlockChars"`
		MinLeaves          int      `json:"minLeaves"`
		MinPrompts         int      `json:"minPrompts"`
		PromotionThreshold float64  `json:"promotionThreshold"`
		RescueThreshold    float64  `json:"rescueThreshold"`
		PromotionCooldown  string   `json:"promotionCooldown"` // Go duration string, e.g. "4h"
		PendingMaxAge      string   `json:"pendingMaxAge"`     // Go duration string
		MergeSuggestCosine float64  `json:"mergeSuggestCosine"`
		AutoNudge          bool     `json:"autoNudge"`
		RedactPatterns     []string `json:"redactPatterns"` // regexes scrubbed from pending bundles
	} `json:"memory"`
}

func defaultConfig() config {
	c := config{
		MemorySize:      100,
		DecayRate:       0.05,
		ContextLimit:    600,
		BubbleUpTerms:   6,
		MaxRefsPerNode:  5,
		GuideSize:       15,
		SessionTimeout:  4.0, // 4 hours
		MergeSimilarity: 0.6,
	}
	c.Similarity.Extend = 0.55
	c.Similarity.Branch = 0.25
	// Typo tolerance defaults. maxDistance=2 catches realistic user typos
	// (e.g. "envaeron" ← "environ" is two edits); the other guards
	// (minWordLen, minEstablishedDF) keep false merges rare.
	c.TypoTolerance.Enabled = true
	c.TypoTolerance.MaxDistance = 2
	c.TypoTolerance.MinWordLen = 5
	c.TypoTolerance.MinEstablishedDF = 3
	// Long-term memory defaults (Session A — surface layer only; candidate
	// detection and promote/commit ship in later sessions). Safe to leave
	// enabled: if no memory files exist yet, the surface block is silent.
	c.Memory.Enabled = true
	c.Memory.Dir = "memories"
	c.Memory.SurfaceThreshold = 0.35
	c.Memory.TopK = 2
	c.Memory.MaxBlockChars = 250
	c.Memory.MinLeaves = 4
	c.Memory.MinPrompts = 3
	c.Memory.PromotionThreshold = 1.5
	c.Memory.RescueThreshold = 1.2
	c.Memory.PromotionCooldown = "4h"
	c.Memory.PendingMaxAge = "168h" // 7 days
	c.Memory.MergeSuggestCosine = 0.6
	c.Memory.AutoNudge = true
	return c
}

// loadConfig uses a two-phase JSON approach to distinguish "user set field to 0"
// from "field absent" (should use default). Phase 1 loads a raw map to detect
// which keys are present. Phase 2 loads the full struct. Only explicitly present
// keys override defaults, so users can intentionally set decayRate=0 without
// the value being silently replaced.
func loadConfig(path string) config {
	cfg := defaultConfig()

	// Phase 1: Detect which keys the user explicitly set.
	raw := make(map[string]json.RawMessage)
	if err := persist.Load(path, &raw); err != nil {
		fmt.Fprintf(os.Stderr, "focus-gate: load config: %v\n", err)
		return cfg
	}
	if len(raw) == 0 {
		return cfg
	}

	// Phase 2: Parse into full struct.
	var userCfg config
	if err := persist.Load(path, &userCfg); err != nil {
		fmt.Fprintf(os.Stderr, "focus-gate: parse config: %v\n", err)
		return cfg
	}

	// Phase 3: Apply only the keys the user explicitly wrote.
	if _, ok := raw["memorySize"]; ok {
		cfg.MemorySize = userCfg.MemorySize
	}
	if _, ok := raw["decayRate"]; ok {
		cfg.DecayRate = userCfg.DecayRate
	}
	if _, ok := raw["contextLimit"]; ok {
		cfg.ContextLimit = userCfg.ContextLimit
	}
	if _, ok := raw["bubbleUpTerms"]; ok {
		cfg.BubbleUpTerms = userCfg.BubbleUpTerms
	}
	if _, ok := raw["guideSize"]; ok {
		cfg.GuideSize = userCfg.GuideSize
	}
	if _, ok := raw["maxRefsPerNode"]; ok {
		cfg.MaxRefsPerNode = userCfg.MaxRefsPerNode
	}
	if _, ok := raw["sessionTimeout"]; ok {
		cfg.SessionTimeout = userCfg.SessionTimeout
	}
	if _, ok := raw["mergeSimilarity"]; ok {
		cfg.MergeSimilarity = userCfg.MergeSimilarity
	}
	// Handle nested "similarity" object.
	if simRaw, ok := raw["similarity"]; ok {
		var simMap map[string]json.RawMessage
		if json.Unmarshal(simRaw, &simMap) == nil {
			if _, ok := simMap["extend"]; ok {
				cfg.Similarity.Extend = userCfg.Similarity.Extend
			}
			if _, ok := simMap["branch"]; ok {
				cfg.Similarity.Branch = userCfg.Similarity.Branch
			}
		}
	}

	// Handle nested "typoTolerance" object using the same key-presence
	// protocol so users can set individual fields (e.g. disable with
	// "enabled": false) without having to spell out every default.
	if ttRaw, ok := raw["typoTolerance"]; ok {
		var ttMap map[string]json.RawMessage
		if json.Unmarshal(ttRaw, &ttMap) == nil {
			if _, ok := ttMap["enabled"]; ok {
				cfg.TypoTolerance.Enabled = userCfg.TypoTolerance.Enabled
			}
			if _, ok := ttMap["maxDistance"]; ok {
				cfg.TypoTolerance.MaxDistance = userCfg.TypoTolerance.MaxDistance
			}
			if _, ok := ttMap["minWordLen"]; ok {
				cfg.TypoTolerance.MinWordLen = userCfg.TypoTolerance.MinWordLen
			}
			if _, ok := ttMap["minEstablishedDF"]; ok {
				cfg.TypoTolerance.MinEstablishedDF = userCfg.TypoTolerance.MinEstablishedDF
			}
		}
	}

	// Handle nested "memory" object with the same explicit-zero protocol.
	if memRaw, ok := raw["memory"]; ok {
		var memMap map[string]json.RawMessage
		if json.Unmarshal(memRaw, &memMap) == nil {
			if _, ok := memMap["enabled"]; ok {
				cfg.Memory.Enabled = userCfg.Memory.Enabled
			}
			if _, ok := memMap["dir"]; ok {
				cfg.Memory.Dir = userCfg.Memory.Dir
			}
			if _, ok := memMap["surfaceThreshold"]; ok {
				cfg.Memory.SurfaceThreshold = userCfg.Memory.SurfaceThreshold
			}
			if _, ok := memMap["topK"]; ok {
				cfg.Memory.TopK = userCfg.Memory.TopK
			}
			if _, ok := memMap["maxBlockChars"]; ok {
				cfg.Memory.MaxBlockChars = userCfg.Memory.MaxBlockChars
			}
			if _, ok := memMap["minLeaves"]; ok {
				cfg.Memory.MinLeaves = userCfg.Memory.MinLeaves
			}
			if _, ok := memMap["minPrompts"]; ok {
				cfg.Memory.MinPrompts = userCfg.Memory.MinPrompts
			}
			if _, ok := memMap["promotionThreshold"]; ok {
				cfg.Memory.PromotionThreshold = userCfg.Memory.PromotionThreshold
			}
			if _, ok := memMap["rescueThreshold"]; ok {
				cfg.Memory.RescueThreshold = userCfg.Memory.RescueThreshold
			}
			if _, ok := memMap["promotionCooldown"]; ok {
				cfg.Memory.PromotionCooldown = userCfg.Memory.PromotionCooldown
			}
			if _, ok := memMap["pendingMaxAge"]; ok {
				cfg.Memory.PendingMaxAge = userCfg.Memory.PendingMaxAge
			}
			if _, ok := memMap["mergeSuggestCosine"]; ok {
				cfg.Memory.MergeSuggestCosine = userCfg.Memory.MergeSuggestCosine
			}
			if _, ok := memMap["autoNudge"]; ok {
				cfg.Memory.AutoNudge = userCfg.Memory.AutoNudge
			}
			if _, ok := memMap["redactPatterns"]; ok {
				cfg.Memory.RedactPatterns = userCfg.Memory.RedactPatterns
			}
		}
	}

	return cfg
}

// hookInput is the JSON structure sent by Claude Code on stdin.
type hookInput struct {
	Prompt         string `json:"prompt"`
	TranscriptPath string `json:"transcript_path"`
}

func main() {
	// Wrap everything in recovery — never block the user's prompt
	defer func() {
		if r := recover(); r != nil {
			fmt.Fprintf(os.Stderr, "focus-gate panic: %v\n", r)
		}
	}()

	if err := run(); err != nil {
		fmt.Fprintf(os.Stderr, "focus-gate: %v\n", err)
		os.Exit(1)
	}
}

func run() error {
	p := resolvePaths()

	// Recover .tmp files from interrupted saves before loading any state.
	persist.RecoverTmpFiles(p.intentFile, p.engineFile, p.guideFile)
	cfg := loadConfig(p.configFile)

	// --quiet suppresses stderr logging
	if hasFlag(os.Args, "--quiet") {
		if f, err := os.Open(os.DevNull); err == nil {
			os.Stderr = f
		}
	}

	// Parse CLI flags. --json is a modifier flag that can appear alongside
	// --inspect or --dry-run to switch output from human-readable text to
	// machine-readable JSON.
	jsonOutput := hasFlag(os.Args, "--json")

	if len(os.Args) > 1 {
		switch os.Args[1] {
		case "--reset":
			return handleReset(p)
		case "--list-projects":
			return handleListProjects()
		case "--status":
			return handleStatus(p, cfg)
		case "--inspect":
			return handleInspect(p, cfg, jsonOutput)
		case "--dry-run":
			// --dry-run expects the next argument to be the prompt string.
			prompt := ""
			if len(os.Args) > 2 && !strings.HasPrefix(os.Args[2], "--") {
				prompt = os.Args[2]
			}
			if prompt == "" {
				return fmt.Errorf("usage: focus --dry-run \"prompt text\" [--json]")
			}
			return handleDryRun(p, cfg, prompt, jsonOutput)
		case "--cmd":
			// Slash command mode for custom Claude Code slash commands.
			// Writes to stdout and exits 0 so output can be captured cleanly.
			sub := ""
			arg := ""
			if len(os.Args) > 2 {
				sub = os.Args[2]
			}
			if len(os.Args) > 3 {
				arg = strings.Join(os.Args[3:], " ")
			}
			return handleSlashCommand(slashCommand{sub: sub, arg: arg}, p, cfg, os.Stdout)
		}
	}

	// Default: hook mode — read prompt from stdin
	return handlePrompt(p, cfg)
}

func handleReset(p paths) error {
	persist.Remove(p.intentFile)
	persist.Remove(p.engineFile)
	persist.Remove(p.guideFile)
	fmt.Fprint(os.Stdout, "[Focus] Reset complete. All tracking data cleared.\n")
	return nil
}

// handleListProjects scans ~/.focus-gate/ and prints one row per per-project
// data directory, showing the sha256 slug, total size on disk, and last
// modification time. This is the discovery tool for orphaned state left
// behind when a project directory gets renamed or moved — the slug changes
// and the old data becomes invisible to --status/--inspect, but is still
// sitting on disk.
func handleListProjects() error {
	home, err := os.UserHomeDir()
	if err != nil {
		return fmt.Errorf("user home: %w", err)
	}
	root := filepath.Join(home, ".focus-gate")
	entries, err := os.ReadDir(root)
	if err != nil {
		if os.IsNotExist(err) {
			fmt.Fprintln(os.Stdout, "[Focus] No project data found — ~/.focus-gate/ does not exist yet.")
			return nil
		}
		return fmt.Errorf("read %s: %w", root, err)
	}

	type projectRow struct {
		slug    string
		size    int64
		modTime int64
	}
	var rows []projectRow
	for _, e := range entries {
		if !e.IsDir() {
			continue
		}
		dir := filepath.Join(root, e.Name())
		info, err := os.Stat(dir)
		if err != nil {
			continue
		}
		size := dirSize(dir)
		rows = append(rows, projectRow{
			slug:    e.Name(),
			size:    size,
			modTime: info.ModTime().UnixMilli(),
		})
	}

	if len(rows) == 0 {
		fmt.Fprintln(os.Stdout, "[Focus] No project data directories under ~/.focus-gate/.")
		return nil
	}

	// Newest first so currently-active projects are at the top.
	sort.Slice(rows, func(i, j int) bool { return rows[i].modTime > rows[j].modTime })

	fmt.Fprintln(os.Stdout, "[Focus] Known project data directories:")
	fmt.Fprintln(os.Stdout)
	fmt.Fprintf(os.Stdout, "  %-14s %10s  %s\n", "SLUG", "SIZE", "LAST MODIFIED")
	for _, r := range rows {
		fmt.Fprintf(os.Stdout, "  %-14s %10s  %s\n",
			r.slug,
			humanSize(r.size),
			time.UnixMilli(r.modTime).Format("2006-01-02 15:04:05"),
		)
	}
	fmt.Fprintln(os.Stdout)
	fmt.Fprintln(os.Stdout, "Slugs are sha256(cwd)[:12]. If a project was renamed, its old slug")
	fmt.Fprintln(os.Stdout, "will appear here but never be touched again — remove with:")
	fmt.Fprintln(os.Stdout, "  rm -rf ~/.focus-gate/<slug>")
	return nil
}

// dirSize returns the total size of all regular files under dir. Returns 0 on
// any error; used only for display so best-effort is fine.
func dirSize(dir string) int64 {
	var total int64
	_ = filepath.Walk(dir, func(_ string, info os.FileInfo, err error) error {
		if err == nil && !info.IsDir() {
			total += info.Size()
		}
		return nil
	})
	return total
}

// humanSize formats a byte count as KB/MB with one decimal.
func humanSize(b int64) string {
	switch {
	case b < 1024:
		return fmt.Sprintf("%d B", b)
	case b < 1024*1024:
		return fmt.Sprintf("%.1f KB", float64(b)/1024)
	default:
		return fmt.Sprintf("%.1f MB", float64(b)/(1024*1024))
	}
}

// logLoadErr logs non-nil persist.Load errors to stderr. Errors are logged
// rather than returned because a corrupt file should not block the user's
// prompt — the system continues with empty/default state and the user can
// --reset if needed.
func logLoadErr(name string, err error) {
	if err != nil {
		fmt.Fprintf(os.Stderr, "focus-gate: load %s: %v\n", name, err)
	}
}

// loadForest returns a forest loaded from disk with schema-version guarding.
// On error (missing file, corrupt data, or version mismatch) the returned
// forest is empty and the error is logged to stderr, never blocking the user.
func loadForest(path string) *forest.Forest {
	f := forest.NewForest()
	logLoadErr("intent", persist.LoadVersioned(path, f, forest.SchemaVersion))
	return f
}

// loadEngine mirrors loadForest for the TF-IDF engine state.
func loadEngine(path string) *tfidf.Engine {
	e := tfidf.NewEngine()
	logLoadErr("engine", persist.LoadVersioned(path, e, tfidf.SchemaVersion))
	return e
}

// loadGuide mirrors loadForest for the guide state.
func loadGuide(path string, maxSize int) *guide.Guide {
	g := guide.New(maxSize)
	logLoadErr("guide", persist.LoadVersioned(path, g, guide.SchemaVersion))
	return g
}

// loadMemoryManifest reads the long-term memory manifest from disk and
// auto-rebuilds it when the directory contents have drifted (hand-edited
// files, vocabulary shift, files added or removed outside the binary).
// Errors are logged rather than returned — a broken manifest must not
// block the user's prompt.
func loadMemoryManifest(dir string, e *tfidf.Engine) *memory.Manifest {
	mf, errs := memory.EnsureFresh(dir, memory.NewVocabSnapshot(e))
	for _, err := range errs {
		fmt.Fprintf(os.Stderr, "focus-gate: memory manifest: %v\n", err)
	}
	if mf == nil {
		return memory.NewManifest()
	}
	return mf
}

// parseDuration wraps time.ParseDuration with a zero-value fallback so
// a missing or malformed config field never panics the hook.
func parseDuration(s string, fallback time.Duration) time.Duration {
	if s == "" {
		return fallback
	}
	d, err := time.ParseDuration(s)
	if err != nil {
		return fallback
	}
	return d
}

// memorySelectConfig maps the CLI config block to the memory package's
// selection config. Done separately from toGateConfig so the memory
// pipeline has its own clean boundary.
func memorySelectConfig(cfg config) memory.SelectConfig {
	return memory.SelectConfig{
		MinLeaves:          cfg.Memory.MinLeaves,
		MinPrompts:         cfg.Memory.MinPrompts,
		PromotionThreshold: cfg.Memory.PromotionThreshold,
		RescueThreshold:    cfg.Memory.RescueThreshold,
		Cooldown:           parseDuration(cfg.Memory.PromotionCooldown, 4*time.Hour),
		MergeSuggestCosine: cfg.Memory.MergeSuggestCosine,
		RedactPatterns:     cfg.Memory.RedactPatterns,
	}
}

// guideSummariesForTree returns the subset of guide summaries whose
// IntentID resolves to a node inside the given tree. Entries without an
// IntentID (legacy or untracked) are skipped — SelectCandidate cares
// about *per-tree* AI reinforcement, so attaching anonymous summaries
// would re-introduce the global-guide leakage we're guarding against.
//
// O(entries × 1) with the map-membership check; called once per tree
// per prompt, so worst case is O(entries × trees) which is bounded by
// guide.MaxSize × MemorySize — negligible.
func guideSummariesForTree(gd *guide.Guide, tree *forest.Tree) []string {
	if gd == nil || tree == nil || len(gd.Entries) == 0 {
		return nil
	}
	var out []string
	for _, e := range gd.Entries {
		if e.IntentID == "" {
			continue
		}
		if _, ok := tree.Nodes[e.IntentID]; ok {
			out = append(out, e.Summary)
		}
	}
	return out
}

// attachMemoryCollector wires the long-term memory candidate collector
// onto a Gate instance so Prune and tryMerge can produce candidates
// inline. The returned function finalises the collector: it appends any
// collected candidates to the pending queue on disk, deduplicated.
//
// Returning a finaliser rather than a side-effectful collector keeps
// the IO off the hot classification path — the hook gets a single
// deterministic save call at end-of-prompt.
//
// Guide entries are scoped per-tree at candidate time via their
// IntentID so each candidate's floor check and fingerprint reflect only
// the AI reinforcement that actually touched that tree, not the whole
// guide buffer.
func attachMemoryCollector(gt *gate.Gate, p paths, cfg config, engine *tfidf.Engine, gd *guide.Guide) func() (int, []error) {
	if !cfg.Memory.Enabled {
		return func() (int, []error) { return 0, nil }
	}

	selectCfg := memorySelectConfig(cfg)
	vocab := memory.NewVocabSnapshot(engine)

	// Load existing manifest entries once so SelectCandidate can suggest
	// merge targets without re-reading per tree. Manifest drift is
	// tolerated — a false-positive merge suggestion is harmless; the
	// LLM ultimately picks create vs merge.
	var existing []memory.IndexEntry
	if mf, _ := memory.Load(p.memoryDir); mf != nil {
		existing = mf.Entries
	}

	// Load pending queue once so Cooldowns and in-queue dedup are honoured
	// even before any save fires.
	maxAge := parseDuration(cfg.Memory.PendingMaxAge, 168*time.Hour)
	pq, err := memory.LoadPending(p.dataDir, maxAge)
	if err != nil || pq == nil {
		pq = memory.NewPendingQueue()
	}

	var collected []*memory.Candidate
	gt.OnTreeAtRisk = func(tree *forest.Tree, reason string) {
		cand := memory.SelectCandidate(memory.SelectInputs{
			Tree:            tree,
			GuideSummaries:  guideSummariesForTree(gd, tree),
			Vocab:           vocab,
			DecayRate:       cfg.DecayRate,
			Reason:          reason,
			ExistingEntries: existing,
			Cooldowns:       pq.Cooldowns,
		}, selectCfg)
		if cand != nil {
			collected = append(collected, cand)
		}
	}

	return func() (int, []error) {
		if len(collected) == 0 {
			return 0, nil
		}
		var errs []error
		deduped := memory.DedupCandidates(collected, 0.85)
		added := pq.AppendCandidates(deduped, 0.85)
		if added == 0 {
			return 0, nil
		}
		if err := pq.Save(p.dataDir); err != nil {
			errs = append(errs, fmt.Errorf("save pending queue: %w", err))
		}
		return added, errs
	}
}

// appendPendingNudge adds a one-line hint to the existing memory surface
// block when the pending queue is non-empty. Kept as a separate helper
// so autoNudge can be toggled without threading the pending count through
// surfaceMemoryBlock. Zero-cost when the queue is empty — a single read
// of the pending file's mtime could be added later if the cost shows up
// in profiles, but the file is small and the hook is not on a hot loop.
func appendPendingNudge(p paths, cfg config, block string) string {
	if !cfg.Memory.Enabled {
		return block
	}
	maxAge := parseDuration(cfg.Memory.PendingMaxAge, 168*time.Hour)
	pq, err := memory.LoadPending(p.dataDir, maxAge)
	if err != nil || pq == nil || len(pq.Candidates) == 0 {
		return block
	}
	nudge := fmt.Sprintf("  (%d topic(s) queued for memory promotion — run `fg: memory promote`)\n", len(pq.Candidates))
	if block == "" {
		return "[Memory ↪ relevant prior context]\n" + nudge
	}
	return block + nudge
}

// surfaceMemoryBlock loads the memory manifest (if enabled), looks up
// entries whose fingerprint matches the prompt vector, and renders a
// pointer block suitable for splicing into the injected context.
//
// Returns the loaded manifest alongside the rendered block so the caller
// can save it at end-of-prompt — the manifest is mutated by touch-count
// increments on every surface, but those increments are debounced into
// a single write per hook invocation.
func surfaceMemoryBlock(p paths, cfg config, e *tfidf.Engine, promptVec tfidf.Vector) (*memory.Manifest, string) {
	if !cfg.Memory.Enabled || len(promptVec) == 0 {
		return nil, ""
	}
	mf := loadMemoryManifest(p.memoryDir, e)
	result := memory.Surface(promptVec, mf, memory.SurfaceConfig{
		Enabled:       cfg.Memory.Enabled,
		Threshold:     cfg.Memory.SurfaceThreshold,
		TopK:          cfg.Memory.TopK,
		MaxBlockChars: cfg.Memory.MaxBlockChars,
	})
	for _, ent := range result.Selected {
		mf.Touch(ent.ID)
	}
	return mf, result.Block
}

func handleStatus(p paths, cfg config) error {
	f := loadForest(p.intentFile)
	e := loadEngine(p.engineFile)
	g := loadGuide(p.guideFile, cfg.GuideSize)

	gateCfg := toGateConfig(cfg)
	gt := gate.New(f, e, gateCfg)
	ctx := gt.GenerateContext()
	if ctx != "" {
		fmt.Fprint(os.Stdout, ctx)
	} else {
		fmt.Fprintf(os.Stdout, "[Focus | %d prompts | %d/%d mem | %d trees]\n[/Focus]\n",
			f.Meta.TotalPrompts, f.NodeCount(), cfg.MemorySize, len(f.Trees))
	}

	guideCtx := g.Render(f)
	if guideCtx != "" {
		fmt.Fprint(os.Stdout, guideCtx)
	}

	return nil
}

func handlePrompt(p paths, cfg config) error {
	// Read all of stdin — works on Windows, Linux, macOS
	data, err := io.ReadAll(os.Stdin)
	if err != nil {
		return fmt.Errorf("read stdin: %w", err)
	}
	if len(data) == 0 {
		return nil
	}

	var input hookInput
	if err := json.Unmarshal(data, &input); err != nil {
		return fmt.Errorf("parse stdin: %w", err)
	}

	prompt := text.CleanPrompt(input.Prompt)
	if prompt == "" {
		return nil
	}

	// Slash command intercept — /focus or fg: runs the inspector.
	//
	// The raw hook-output context block is rendered reliably in the
	// Claude Code CLI but may be hidden from the user in the VSCode
	// extension (the extension shows hooks to the model, not to the user).
	// So we wrap the inspector output with a short instruction telling
	// the model to echo it back verbatim. The model's reply then appears
	// in the visible chat regardless of which UI the user is on.
	//
	// Exit 0 lets the user's literal "fg: status" text still reach the
	// model; the wrapper makes the model's response the inspector output
	// itself, not a conversational reply to "fg: status". The classifier
	// never runs because os.Exit halts main below.
	if cmd, ok := parseSlashCommand(input.Prompt); ok {
		var buf strings.Builder
		if err := handleSlashCommand(cmd, p, cfg, &buf); err != nil {
			fmt.Fprintf(os.Stderr, "focus-gate: %v\n", err)
		}
		fmt.Fprintln(os.Stdout, "[focus-gate]: The user invoked a focus-gate inspector command. Reply with EXACTLY the fenced block below, and nothing else — no preamble, no commentary.")
		fmt.Fprintln(os.Stdout)
		fmt.Fprintln(os.Stdout, "```")
		fmt.Fprint(os.Stdout, buf.String())
		fmt.Fprintln(os.Stdout, "```")
		os.Exit(0)
	}

	// Serialize concurrent UserPromptSubmit hooks on the same project. The
	// lock is held across load → mutate → save so two simultaneous prompts
	// cannot race on state files and silently drop one another's updates.
	lock, err := persist.Acquire(p.lockFile)
	if err != nil {
		fmt.Fprintf(os.Stderr, "focus-gate: acquire lock: %v\n", err)
	}
	defer func() {
		if lock != nil {
			_ = lock.Release()
		}
	}()

	// Load persisted state
	f := loadForest(p.intentFile)
	e := loadEngine(p.engineFile)
	g := loadGuide(p.guideFile, cfg.GuideSize)

	// Update guide from transcript (if available)
	if input.TranscriptPath != "" {
		updateGuide(g, input.TranscriptPath, f)
	}

	// Process prompt
	gateCfg := toGateConfig(cfg)
	gt := gate.New(f, e, gateCfg)

	// Reinforce the forest from new AI response summaries before classifying
	// the incoming prompt, so tree scores reflect recent assistant activity.
	if reinforced := gt.ReinforceFromGuide(g); reinforced > 0 {
		fmt.Fprintf(os.Stderr, "focus-gate: reinforced %d guide entries\n", reinforced)
	}

	// Attach the long-term memory candidate collector so Gate.Prune and
	// Gate.tryMerge can stash at-risk trees as pending candidates. The
	// finaliser is invoked after ProcessPrompt so collected candidates
	// are batch-written once per hook invocation. The guide is passed
	// whole so the collector can scope summaries to each tree's own
	// intent nodes rather than blanket-attaching the entire guide to
	// every candidate.
	finaliseMemoryCandidates := attachMemoryCollector(gt, p, cfg, e, g)

	// Process the new prompt
	ctx := gt.ProcessPrompt(prompt, fmt.Sprintf("p%d", f.Meta.TotalPrompts))

	// Persist any pending-memory candidates the collector gathered.
	if added, errs := finaliseMemoryCandidates(); added > 0 {
		for _, err := range errs {
			fmt.Fprintf(os.Stderr, "focus-gate: memory candidate: %v\n", err)
		}
		fmt.Fprintf(os.Stderr, "focus-gate: queued %d memory candidate(s) for promotion\n", added)
	}

	// Long-term memory surface: look up memories whose fingerprint matches
	// the prompt vector and render a pointer block before the guide. Silent
	// when disabled, when the manifest is empty, or when no entry meets
	// the similarity threshold.
	memManifest, memBlock := surfaceMemoryBlock(p, cfg, e, gt.LastPromptVector())
	if cfg.Memory.AutoNudge {
		memBlock = appendPendingNudge(p, cfg, memBlock)
	}
	if memBlock != "" {
		ctx = strings.Replace(ctx, "[/Focus]\n", memBlock+"[/Focus]\n", 1)
	}

	// Append guide context
	guideCtx := g.Render(f)
	if guideCtx != "" {
		// Insert guide before [/Focus]
		ctx = strings.Replace(ctx, "[/Focus]\n", guideCtx+"[/Focus]\n", 1)
	}

	// Save all state atomically
	if err := persist.SaveAtomic(p.intentFile, f); err != nil {
		fmt.Fprintf(os.Stderr, "focus-gate: save intent: %v\n", err)
	}
	if err := persist.SaveAtomic(p.engineFile, e); err != nil {
		fmt.Fprintf(os.Stderr, "focus-gate: save engine: %v\n", err)
	}
	if err := persist.SaveAtomic(p.guideFile, g); err != nil {
		fmt.Fprintf(os.Stderr, "focus-gate: save guide: %v\n", err)
	}
	// Touch counter increments from surfacing are batched into this single
	// manifest write at the end of the prompt.
	if memManifest != nil && memManifest.Dirty() {
		if err := memManifest.Save(p.memoryDir); err != nil {
			fmt.Fprintf(os.Stderr, "focus-gate: save memory manifest: %v\n", err)
		}
	}

	// Output context to stdout
	fmt.Fprint(os.Stdout, ctx)
	return nil
}

// updateGuide extracts the last assistant message from a Claude Code transcript
// and adds it to the guide. Uses structured JSON decoding to handle all valid
// transcript formats — plain string content, arrays of content blocks, nested
// objects, and escaped characters.
func updateGuide(g *guide.Guide, transcriptPath string, f *forest.Forest) {
	data, err := os.ReadFile(transcriptPath)
	if err != nil {
		return
	}

	// Claude Code transcript: JSON array of {role, message: {content}} objects.
	// content may be a plain string or an array of {type, text} blocks.
	type transcriptEntry struct {
		Role    string `json:"role"`
		Message struct {
			Content json.RawMessage `json:"content"`
		} `json:"message"`
	}

	var transcript []transcriptEntry
	if err := json.Unmarshal(data, &transcript); err != nil {
		return
	}

	// Walk backwards to find the last assistant message.
	snippet := ""
	for i := len(transcript) - 1; i >= 0; i-- {
		if transcript[i].Role != "assistant" {
			continue
		}

		raw := transcript[i].Message.Content
		if len(raw) == 0 {
			continue
		}

		// Try content as plain string first, then as array of content blocks.
		var contentStr string
		if json.Unmarshal(raw, &contentStr) == nil && contentStr != "" {
			snippet = contentStr
			break
		}

		// Array of content blocks (Claude format): [{type, text}, ...].
		var blocks []struct {
			Type string `json:"type"`
			Text string `json:"text"`
		}
		if json.Unmarshal(raw, &blocks) == nil {
			for _, block := range blocks {
				if block.Text != "" {
					snippet = block.Text
					break
				}
			}
			if snippet != "" {
				break
			}
		}
	}

	// Truncate to a summary length.
	if len(snippet) > 200 {
		snippet = snippet[:200] + "..."
	}
	snippet = strings.TrimSpace(snippet)
	if snippet == "" {
		return
	}

	// Link to the most recent leaf in the last tree.
	intentID := ""
	if len(f.Trees) > 0 {
		lastTree := f.Trees[len(f.Trees)-1]
		leaves := lastTree.GetLeaves()
		if len(leaves) > 0 {
			intentID = leaves[len(leaves)-1].ID
		}
	}

	g.Add(snippet, intentID, nil)
}

func toGateConfig(cfg config) gate.Config {
	// Use the current working directory as the base for file-ref validation.
	// In hook mode, Claude Code runs this binary from the project root, which
	// is exactly what we want. If Getwd fails (e.g. the dir was deleted out
	// from under us), fall back to empty — validation then becomes a no-op.
	projectDir, _ := os.Getwd()
	return gate.Config{
		ExtendThreshold: cfg.Similarity.Extend,
		BranchThreshold: cfg.Similarity.Branch,
		BubbleUpTerms:   cfg.BubbleUpTerms,
		MaxRefsPerNode:  cfg.MaxRefsPerNode,
		MemorySize:      cfg.MemorySize,
		DecayRate:       cfg.DecayRate,
		ContextLimit:    cfg.ContextLimit,
		SessionTimeout:  cfg.SessionTimeout,
		MergeSimilarity: cfg.MergeSimilarity,
		ProjectDir:      projectDir,
		TypoTolerance: text.CanonicalizeOpts{
			Enabled:          cfg.TypoTolerance.Enabled,
			MaxDistance:      cfg.TypoTolerance.MaxDistance,
			MinWordLen:       cfg.TypoTolerance.MinWordLen,
			MinEstablishedDF: cfg.TypoTolerance.MinEstablishedDF,
		},
	}
}
