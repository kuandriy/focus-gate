package main

import (
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"

	"github.com/kuandriy/focus-gate/internal/forest"
	"github.com/kuandriy/focus-gate/internal/gate"
	"github.com/kuandriy/focus-gate/internal/guide"
	"github.com/kuandriy/focus-gate/internal/markov"
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
	markovFile string
	configFile string
}

func resolvePaths() paths {
	exe, err := os.Executable()
	if err != nil {
		exe = "."
	}
	dir := filepath.Dir(exe)
	dataDir := filepath.Join(dir, "data")
	return paths{
		dataDir:    dataDir,
		intentFile: filepath.Join(dataDir, "intent.json"),
		engineFile: filepath.Join(dataDir, "engine.json"),
		guideFile:  filepath.Join(dataDir, "guide.json"),
		markovFile: filepath.Join(dataDir, "markov.json"),
		configFile: filepath.Join(dir, "config.json"),
	}
}

// config matches the JSON config file structure.
type config struct {
	MemorySize int     `json:"memorySize"`
	DecayRate  float64 `json:"decayRate"`
	Similarity struct {
		Extend float64 `json:"extend"`
		Branch float64 `json:"branch"`
	} `json:"similarity"`
	ContextLimit      int     `json:"contextLimit"`
	BubbleUpTerms     int     `json:"bubbleUpTerms"`
	MaxSourcesPerNode int     `json:"maxSourcesPerNode"`
	GuideSize         int     `json:"guideSize"`
	TransitionBoost   float64 `json:"transitionBoost"`
}

func defaultConfig() config {
	c := config{
		MemorySize:        100,
		DecayRate:         0.05,
		ContextLimit:      600,
		BubbleUpTerms:     6,
		MaxSourcesPerNode: 20,
		GuideSize:         15,
		TransitionBoost:   0.2,
	}
	c.Similarity.Extend = 0.55
	c.Similarity.Branch = 0.25
	return c
}

// loadConfig uses a two-phase JSON approach to distinguish "user set field to 0"
// from "field absent" (should use default). Phase 1 loads a raw map to detect
// which keys are present. Phase 2 loads the full struct. Only explicitly present
// keys override defaults, so users can intentionally set transitionBoost=0 or
// decayRate=0 without the value being silently replaced.
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
	if _, ok := raw["maxSourcesPerNode"]; ok {
		cfg.MaxSourcesPerNode = userCfg.MaxSourcesPerNode
	}
	if _, ok := raw["guideSize"]; ok {
		cfg.GuideSize = userCfg.GuideSize
	}
	if _, ok := raw["transitionBoost"]; ok {
		cfg.TransitionBoost = userCfg.TransitionBoost
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
	persist.RecoverTmpFiles(p.intentFile, p.engineFile, p.guideFile, p.markovFile)
	cfg := loadConfig(p.configFile)

	// Parse CLI flags
	if len(os.Args) > 1 {
		switch os.Args[1] {
		case "--reset":
			return handleReset(p)
		case "--status":
			return handleStatus(p, cfg)
		}
	}

	// Default: hook mode — read prompt from stdin
	return handlePrompt(p, cfg)
}

func handleReset(p paths) error {
	persist.Remove(p.intentFile)
	persist.Remove(p.engineFile)
	persist.Remove(p.guideFile)
	persist.Remove(p.markovFile)
	fmt.Fprint(os.Stdout, "[Focus] Reset complete. All tracking data cleared.\n")
	return nil
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

func handleStatus(p paths, cfg config) error {
	f := forest.NewForest()
	logLoadErr("intent", persist.Load(p.intentFile, f))

	e := tfidf.NewEngine()
	logLoadErr("engine", persist.Load(p.engineFile, e))

	g := guide.New(cfg.GuideSize)
	logLoadErr("guide", persist.Load(p.guideFile, g))

	c := markov.New()
	logLoadErr("markov", persist.Load(p.markovFile, c))

	gateCfg := toGateConfig(cfg)
	gt := gate.NewWithChain(f, e, c, gateCfg)
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

	// Load persisted state
	f := forest.NewForest()
	logLoadErr("intent", persist.Load(p.intentFile, f))

	e := tfidf.NewEngine()
	logLoadErr("engine", persist.Load(p.engineFile, e))

	g := guide.New(cfg.GuideSize)
	logLoadErr("guide", persist.Load(p.guideFile, g))

	c := markov.New()
	logLoadErr("markov", persist.Load(p.markovFile, c))

	// Update guide from transcript (if available)
	if input.TranscriptPath != "" {
		updateGuide(g, input.TranscriptPath, f)
	}

	// Process prompt
	gateCfg := toGateConfig(cfg)
	gt := gate.NewWithChain(f, e, c, gateCfg)

	// Reinforce the forest from new AI response summaries before classifying
	// the incoming prompt, so tree scores reflect recent assistant activity.
	if reinforced := gt.ReinforceFromGuide(g); reinforced > 0 {
		fmt.Fprintf(os.Stderr, "focus-gate: reinforced %d guide entries\n", reinforced)
	}

	// Process the new prompt
	ctx := gt.ProcessPrompt(prompt, fmt.Sprintf("p%d", f.Meta.TotalPrompts))

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
	if err := persist.SaveAtomic(p.markovFile, c); err != nil {
		fmt.Fprintf(os.Stderr, "focus-gate: save markov: %v\n", err)
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
	return gate.Config{
		ExtendThreshold:   cfg.Similarity.Extend,
		BranchThreshold:   cfg.Similarity.Branch,
		BubbleUpTerms:     cfg.BubbleUpTerms,
		MaxSourcesPerNode: cfg.MaxSourcesPerNode,
		MemorySize:        cfg.MemorySize,
		DecayRate:         cfg.DecayRate,
		ContextLimit:      cfg.ContextLimit,
		TransitionBoost:   cfg.TransitionBoost,
	}
}
