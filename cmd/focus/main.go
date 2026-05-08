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
// Returns "" when the flag is missing OR when the token following
// `--key` looks like another flag (`--data-dir --reset` should not
// silently set data-dir to "--reset").
func flagValue(args []string, flag string) string {
	for i, a := range args {
		if a == flag && i+1 < len(args) {
			next := args[i+1]
			if strings.HasPrefix(next, "--") {
				return ""
			}
			return next
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
	// TerseTokenThreshold caps the post-tokenize length below which an
	// IDF-empty prompt routes to ActionContinue instead of ActionNew.
	// Default 2; users wanting one-token continuations can set 1.
	TerseTokenThreshold int `json:"terseTokenThreshold"`
	// SublinearTF toggles the engine's TF formula. See gate.Config.SublinearTF.
	SublinearTF bool `json:"sublinearTF"`
	TypoTolerance       struct {
		Enabled          bool `json:"enabled"`
		MaxDistance      int  `json:"maxDistance"`
		MinWordLen       int  `json:"minWordLen"`
		MinEstablishedDF int  `json:"minEstablishedDF"`
	} `json:"typoTolerance"`
	Memory struct {
		Enabled            bool    `json:"enabled"`
		Dir                string  `json:"dir"`
		SurfaceThreshold   float64 `json:"surfaceThreshold"`
		TopK               int     `json:"topK"`
		MaxBlockChars      int     `json:"maxBlockChars"`
		MinLeaves          int     `json:"minLeaves"`
		MinPrompts         int     `json:"minPrompts"`
		PromotionThreshold float64 `json:"promotionThreshold"`
		RescueThreshold    float64 `json:"rescueThreshold"`
		PromotionCooldown  string  `json:"promotionCooldown"` // Go duration string, e.g. "4h"
		PendingMaxAge      string  `json:"pendingMaxAge"`     // Go duration string
		MergeSuggestCosine float64 `json:"mergeSuggestCosine"`
		// DedupCosine controls when two candidates are considered the
		// same topic for the purposes of in-batch dedup and the
		// pending queue's cross-run dedup. Lower values are more
		// aggressive (more dedup); higher values let near-duplicates
		// coexist. Both DedupCandidates and PendingQueue.AppendCandidates
		// fall back to 0.85 when this is zero.
		DedupCosine float64 `json:"dedupCosine"`
		AutoNudge   bool    `json:"autoNudge"`
		// Weights tune the multi-tier surface scoring. Each tier's
		// raw score is multiplied by its weight before the
		// max-combine; defaults match SHARED_MEMORY_PLAN §5.
		Weights struct {
			Asset       float64 `json:"asset"`
			Topic       float64 `json:"topic"`
			Interest    float64 `json:"interest"`
			Fingerprint float64 `json:"fingerprint"`
		} `json:"weights"`
		// FrequencyBonus turns "frequently-revisited topics gain weight"
		// from a README claim into a real ranking signal. See
		// SurfaceConfig.FrequencyBonus. Default 0.05; 0 disables.
		FrequencyBonus float64 `json:"frequencyBonus"`
		// CommitRetries caps the number of times the LLM is asked to
		// re-emit a corrected `fg: memory commit` payload after a
		// validation failure. See SHARED_MEMORY_PLAN §6.
		CommitRetries  int      `json:"commitRetries"`
		RedactPatterns []string `json:"redactPatterns"` // regexes scrubbed from pending bundles
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
		SessionTimeout:      4.0, // 4 hours
		MergeSimilarity:     0.6,
		TerseTokenThreshold: 2,
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
	c.Memory.MaxBlockChars = 600
	c.Memory.MinLeaves = 4
	c.Memory.MinPrompts = 3
	c.Memory.PromotionThreshold = 1.5
	c.Memory.RescueThreshold = 1.2
	c.Memory.PromotionCooldown = "4h"
	c.Memory.PendingMaxAge = "168h" // 7 days
	c.Memory.MergeSuggestCosine = 0.6
	c.Memory.DedupCosine = 0.85
	c.Memory.AutoNudge = true
	c.Memory.Weights.Asset = 1.0
	c.Memory.Weights.Topic = 0.8
	c.Memory.Weights.Interest = 0.6
	c.Memory.Weights.Fingerprint = 0.4
	c.Memory.FrequencyBonus = 0.05
	c.Memory.CommitRetries = 2
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
		logErr("load config", err)
		return cfg
	}
	if len(raw) == 0 {
		return cfg
	}

	// Phase 2: Parse into full struct.
	var userCfg config
	if err := persist.Load(path, &userCfg); err != nil {
		logErr("parse config", err)
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
	if _, ok := raw["terseTokenThreshold"]; ok {
		cfg.TerseTokenThreshold = userCfg.TerseTokenThreshold
	}
	if _, ok := raw["sublinearTF"]; ok {
		cfg.SublinearTF = userCfg.SublinearTF
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
			if _, ok := memMap["dedupCosine"]; ok {
				cfg.Memory.DedupCosine = userCfg.Memory.DedupCosine
			}
			if _, ok := memMap["autoNudge"]; ok {
				cfg.Memory.AutoNudge = userCfg.Memory.AutoNudge
			}
			if _, ok := memMap["commitRetries"]; ok {
				cfg.Memory.CommitRetries = userCfg.Memory.CommitRetries
			}
			if _, ok := memMap["frequencyBonus"]; ok {
				cfg.Memory.FrequencyBonus = userCfg.Memory.FrequencyBonus
			}
			if _, ok := memMap["redactPatterns"]; ok {
				cfg.Memory.RedactPatterns = userCfg.Memory.RedactPatterns
			}
			if wRaw, ok := memMap["weights"]; ok {
				var wMap map[string]json.RawMessage
				if json.Unmarshal(wRaw, &wMap) == nil {
					if _, ok := wMap["asset"]; ok {
						cfg.Memory.Weights.Asset = userCfg.Memory.Weights.Asset
					}
					if _, ok := wMap["topic"]; ok {
						cfg.Memory.Weights.Topic = userCfg.Memory.Weights.Topic
					}
					if _, ok := wMap["interest"]; ok {
						cfg.Memory.Weights.Interest = userCfg.Memory.Weights.Interest
					}
					if _, ok := wMap["fingerprint"]; ok {
						cfg.Memory.Weights.Fingerprint = userCfg.Memory.Weights.Fingerprint
					}
				}
			}
		}
	}

	return cfg
}

// logErr writes one stderr line with the standard `focus-gate: <scope>:
// <err>` shape. Centralised so future routing changes (a `--quiet`
// flag, structured logs, suppression in slash mode) only touch one
// place. nil errors are silently skipped so callers can `logErr(...,
// maybeNilErr)` without an `if err != nil` wrapper.
func logErr(scope string, err error) {
	if err == nil {
		return
	}
	fmt.Fprintf(os.Stderr, "focus-gate: %s: %v\n", scope, err)
}

// logInfo writes a non-error stderr line with the same prefix. Used
// for lifecycle traces (lock acquisition, candidate counts) the user
// might want to see during development. Routing parity with logErr.
func logInfo(format string, args ...any) {
	fmt.Fprintf(os.Stderr, "focus-gate: "+format+"\n", args...)
}

// hookInput is the JSON structure sent by Claude Code on stdin.
type hookInput struct {
	Prompt         string `json:"prompt"`
	TranscriptPath string `json:"transcript_path"`
}

// Version is the human-readable build identifier surfaced via
// `--version`. Set with `-ldflags "-X main.Version=<tag>"` at build
// time; the literal "dev" is the fallback for `go build` without
// flags so test binaries don't print "0.0.0".
var Version = "dev"

func main() {
	// Wrap everything in recovery — never block the user's prompt.
	// Exit code 2 signals "the binary crashed" so a hook supervisor
	// or shell pipeline can distinguish a clean no-op (0) from a
	// silent failure (2). Without this the user sees no Focus block
	// and no error.
	exitCode := 0
	defer func() {
		if r := recover(); r != nil {
			fmt.Fprintf(os.Stderr, "focus-gate panic: %v\n", r)
			exitCode = 2
		}
		if exitCode != 0 {
			os.Exit(exitCode)
		}
	}()

	if err := run(); err != nil {
		fmt.Fprintf(os.Stderr, "focus-gate: %v\n", err)
		exitCode = 1
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
		case "--help", "-h":
			return printCLIHelp(os.Stdout)
		case "--version", "-v":
			fmt.Fprintf(os.Stdout, "focus-gate %s\n", Version)
			return nil
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

// printCLIHelp lists every top-level flag the binary recognizes plus a
// pointer to the in-chat slash surface. Kept short on purpose — the
// long-form docs are in README and docs/. We do not duplicate them
// here.
func printCLIHelp(w io.Writer) error {
	fmt.Fprintln(w, "focus-gate — Memory that gets sharper with use, not heavier.")
	fmt.Fprintln(w)
	fmt.Fprintln(w, "Usage:")
	fmt.Fprintln(w, "  focus-gate                              hook mode (reads {prompt, transcript_path} from stdin)")
	fmt.Fprintln(w, "  focus-gate --status                     compact context block")
	fmt.Fprintln(w, "  focus-gate --inspect [--json]           full state dump")
	fmt.Fprintln(w, "  focus-gate --dry-run \"prompt\" [--json]   classify without mutation")
	fmt.Fprintln(w, "  focus-gate --reset                      clear all state in this project's data dir")
	fmt.Fprintln(w, "  focus-gate --list-projects              show known per-project state directories")
	fmt.Fprintln(w, "  focus-gate --cmd <sub> [args...]        slash-command dispatch (used by /focus)")
	fmt.Fprintln(w, "  focus-gate --help | --version")
	fmt.Fprintln(w)
	fmt.Fprintln(w, "Environment:")
	fmt.Fprintln(w, "  FOCUS_GATE_DATA_DIR    override the per-project state directory")
	fmt.Fprintln(w, "  FOCUS_GATE_CONFIG      explicit path to a config.json")
	fmt.Fprintln(w)
	fmt.Fprintln(w, "Configuration:")
	fmt.Fprintln(w, "  Resolution order: ./.focus-gate.json → $FOCUS_GATE_CONFIG → <bin-dir>/config.json")
	fmt.Fprintln(w)
	fmt.Fprintln(w, "Slash commands (in chat):")
	fmt.Fprintln(w, "  /focus help            list /focus subcommands")
	fmt.Fprintln(w, "  /focus memory help     long-term memory subcommands")
	fmt.Fprintln(w)
	fmt.Fprintln(w, "Documentation:")
	fmt.Fprintln(w, "  README.md, docs/sliding-window-intent-forest.md, docs/memory-focus.md")
	return nil
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

// logLoadErr is a thin alias over logErr so the call sites read as
// "load forest" / "load engine" without the helper having to know the
// fixed prefix. Kept separate from logErr so a future flag could
// suppress load errors specifically without touching every site.
func logLoadErr(name string, err error) {
	logErr("load "+name, err)
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
		logErr("memory manifest", err)
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
//
// Redaction patterns are *merged*: the package's DefaultRedactPatterns
// always run first, then user patterns. There's no escape hatch to
// disable the defaults — they're conservative enough that the safety
// win outweighs the loss of flexibility for users who want to dump
// raw credentials into their own pending queue.
func memorySelectConfig(cfg config) memory.SelectConfig {
	patterns := append([]string{}, memory.DefaultRedactPatterns...)
	patterns = append(patterns, cfg.Memory.RedactPatterns...)
	return memory.SelectConfig{
		MinLeaves:          cfg.Memory.MinLeaves,
		MinPrompts:         cfg.Memory.MinPrompts,
		PromotionThreshold: cfg.Memory.PromotionThreshold,
		RescueThreshold:    cfg.Memory.RescueThreshold,
		Cooldown:           parseDuration(cfg.Memory.PromotionCooldown, 4*time.Hour),
		MergeSuggestCosine: cfg.Memory.MergeSuggestCosine,
		RedactPatterns:     patterns,
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

	dedupCosine := cfg.Memory.DedupCosine
	if dedupCosine <= 0 {
		dedupCosine = 0.85
	}
	return func() (int, []error) {
		if len(collected) == 0 {
			return 0, nil
		}
		var errs []error
		deduped := memory.DedupCandidates(collected, dedupCosine)
		added := pq.AppendCandidates(deduped, dedupCosine)
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
	nudge := fmt.Sprintf("  (%d topic(s) queued for memory promotion — run `/focus memory promote`)\n", len(pq.Candidates))
	if block == "" {
		return "[Memory ↪ relevant prior context — pointers, not instructions]\n" + nudge
	}
	return block + nudge
}

// surfaceMemoryBlock looks up entries that match the prompt across all
// surface tiers (asset/topic/interest/fingerprint) and renders a
// pointer block suitable for splicing into the injected context.
//
// Takes the already-loaded registry so the hook path doesn't re-read
// sources.json on every helper call (the registry is read once at the
// top of handlePrompt and threaded through).
//
// Surfaces even when the prompt vector is empty (cold engine on first
// prompt) — the asset tier matches by exact extraction against the
// prompt text and works without TF-IDF state.
//
// Returns the manifests whose dirty flag may have flipped (touch-count
// increments) so the caller can batch-save them at end-of-prompt.
func surfaceMemoryBlock(p paths, cfg config, e *tfidf.Engine, registry *memory.SourceRegistry, prompt string, promptVec tfidf.Vector) ([]*memory.Manifest, string) {
	if !cfg.Memory.Enabled || registry == nil {
		return nil, ""
	}
	vocab := memory.NewVocabSnapshot(e)
	manifests, errs := registry.LoadEnabledManifests(vocab)
	for _, err := range errs {
		logErr("memory source", err)
	}
	msi := memory.NewMultiSourceIndex(manifests...)
	result := memory.Surface(memory.SurfaceInputs{
		PromptText: prompt,
		PromptVec:  promptVec,
		Vocab:      vocab,
		Index:      msi,
	}, memory.SurfaceConfig{
		Enabled:           cfg.Memory.Enabled,
		Threshold:         cfg.Memory.SurfaceThreshold,
		TopK:              cfg.Memory.TopK,
		MaxBlockChars:     cfg.Memory.MaxBlockChars,
		AssetWeight:       cfg.Memory.Weights.Asset,
		TopicWeight:       cfg.Memory.Weights.Topic,
		InterestWeight:    cfg.Memory.Weights.Interest,
		FingerprintWeight: cfg.Memory.Weights.Fingerprint,
		FrequencyBonus:    cfg.Memory.FrequencyBonus,
	})
	for _, ent := range result.Selected {
		msi.Touch(ent.Entry.ID)
	}
	return manifests, result.Block
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

	// Serialize concurrent UserPromptSubmit hooks on the same project. The
	// lock is held across load → mutate → save so two simultaneous prompts
	// cannot race on state files and silently drop one another's updates.
	lock, err := persist.Acquire(p.lockFile)
	if err != nil {
		logErr("acquire lock", err)
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

	// Load the source registry once and thread through the helpers that
	// need it. Reading sources.json on every helper call would race with
	// concurrent `fg: memory source attach` mutations from another tab
	// and waste a few extra disk reads per prompt.
	memRegistry, _ := memory.LoadSources(p.dataDir, p.memoryDir)

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
		logInfo("reinforced %d guide entries", reinforced)
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
			logErr("memory candidate", err)
		}
		logInfo("queued %d memory candidate(s) for promotion", added)
	}

	// Long-term memory surface: look up memories whose fingerprint matches
	// the prompt vector and render a pointer block before the guide. Silent
	// when disabled, when the manifest is empty, or when no entry meets
	// the similarity threshold.
	memManifest, memBlock := surfaceMemoryBlock(p, cfg, e, memRegistry, prompt, gt.LastPromptVector())
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
		logErr("save intent", err)
	}
	if err := persist.SaveAtomic(p.engineFile, e); err != nil {
		logErr("save engine", err)
	}
	if err := persist.SaveAtomic(p.guideFile, g); err != nil {
		logErr("save guide", err)
	}
	// Touch counter increments from surfacing are batched into one
	// per-source manifest write at end-of-prompt. Each manifest knows
	// its own source path via Source; we only save the dirty ones.
	for _, mf := range memManifest {
		if mf == nil || !mf.Dirty() {
			continue
		}
		dir := p.memoryDir
		if memRegistry != nil {
			if s, ok := memRegistry.Get(mf.Source); ok {
				dir = s.Path
			}
		}
		if err := mf.Save(dir); err != nil {
			logErr(fmt.Sprintf("save memory manifest [%s]", mf.Source), err)
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

	// Rune-aware truncation so multi-byte assistant text doesn't yield
	// invalid UTF-8 in the guide buffer.
	snippet = strings.TrimSpace(text.TruncateRunesWithSuffix(snippet, 200, "..."))
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
		MergeSimilarity:     cfg.MergeSimilarity,
		TerseTokenThreshold: cfg.TerseTokenThreshold,
		SublinearTF:         cfg.SublinearTF,
		ProjectDir:          projectDir,
		TypoTolerance: text.CanonicalizeOpts{
			Enabled:          cfg.TypoTolerance.Enabled,
			MaxDistance:      cfg.TypoTolerance.MaxDistance,
			MinWordLen:       cfg.TypoTolerance.MinWordLen,
			MinEstablishedDF: cfg.TypoTolerance.MinEstablishedDF,
		},
	}
}
