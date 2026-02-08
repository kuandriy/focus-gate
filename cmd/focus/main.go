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
		configFile: filepath.Join(dir, "config.json"),
	}
}

// config matches the JSON config file structure.
type config struct {
	MemorySize        int     `json:"memorySize"`
	DecayRate         float64 `json:"decayRate"`
	Similarity        struct {
		Extend float64 `json:"extend"`
		Branch float64 `json:"branch"`
	} `json:"similarity"`
	ContextLimit      int `json:"contextLimit"`
	BubbleUpTerms     int `json:"bubbleUpTerms"`
	MaxSourcesPerNode int `json:"maxSourcesPerNode"`
	GuideSize         int `json:"guideSize"`
}

func defaultConfig() config {
	c := config{
		MemorySize:        100,
		DecayRate:         0.05,
		ContextLimit:      600,
		BubbleUpTerms:     6,
		MaxSourcesPerNode: 20,
		GuideSize:         15,
	}
	c.Similarity.Extend = 0.55
	c.Similarity.Branch = 0.25
	return c
}

func loadConfig(path string) config {
	cfg := defaultConfig()
	_ = persist.Load(path, &cfg)
	// Apply defaults for any zero values
	d := defaultConfig()
	if cfg.MemorySize == 0 {
		cfg.MemorySize = d.MemorySize
	}
	if cfg.DecayRate == 0 {
		cfg.DecayRate = d.DecayRate
	}
	if cfg.Similarity.Extend == 0 {
		cfg.Similarity.Extend = d.Similarity.Extend
	}
	if cfg.Similarity.Branch == 0 {
		cfg.Similarity.Branch = d.Similarity.Branch
	}
	if cfg.ContextLimit == 0 {
		cfg.ContextLimit = d.ContextLimit
	}
	if cfg.BubbleUpTerms == 0 {
		cfg.BubbleUpTerms = d.BubbleUpTerms
	}
	if cfg.MaxSourcesPerNode == 0 {
		cfg.MaxSourcesPerNode = d.MaxSourcesPerNode
	}
	if cfg.GuideSize == 0 {
		cfg.GuideSize = d.GuideSize
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
	fmt.Fprint(os.Stdout, "[Focus] Reset complete. All tracking data cleared.\n")
	return nil
}

func handleStatus(p paths, cfg config) error {
	f := forest.NewForest()
	_ = persist.Load(p.intentFile, f)

	e := tfidf.NewEngine()
	_ = persist.Load(p.engineFile, e)

	g := guide.New(cfg.GuideSize)
	_ = persist.Load(p.guideFile, g)

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

	// Load persisted state
	f := forest.NewForest()
	_ = persist.Load(p.intentFile, f)

	e := tfidf.NewEngine()
	_ = persist.Load(p.engineFile, e)

	g := guide.New(cfg.GuideSize)
	_ = persist.Load(p.guideFile, g)

	// Update guide from transcript (if available)
	if input.TranscriptPath != "" {
		updateGuide(g, input.TranscriptPath, f)
	}

	// Process prompt
	gateCfg := toGateConfig(cfg)
	gt := gate.New(f, e, gateCfg)
	ctx := gt.GenerateContext()

	// Process the new prompt
	ctx = gt.ProcessPrompt(prompt, fmt.Sprintf("p%d", f.Meta.TotalPrompts))

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

	// Output context to stdout
	fmt.Fprint(os.Stdout, ctx)
	return nil
}

// updateGuide reads the last assistant response from the transcript and adds it to the guide.
func updateGuide(g *guide.Guide, transcriptPath string, f *forest.Forest) {
	data, err := os.ReadFile(transcriptPath)
	if err != nil {
		return
	}

	// Find last assistant message in the transcript
	content := string(data)
	idx := strings.LastIndex(content, `"role":"assistant"`)
	if idx < 0 {
		idx = strings.LastIndex(content, `"role": "assistant"`)
	}
	if idx < 0 {
		return
	}

	// Extract a summary (first ~200 chars of the response content)
	contentStart := strings.Index(content[idx:], `"content"`)
	if contentStart < 0 {
		return
	}
	snippet := content[idx+contentStart:]
	// Find the text value
	valStart := strings.Index(snippet, `":"`)
	if valStart < 0 {
		valStart = strings.Index(snippet, `": "`)
		if valStart < 0 {
			return
		}
		valStart += 4
	} else {
		valStart += 3
	}
	snippet = snippet[valStart:]
	// Take up to 200 chars, stop at quote
	if endQuote := strings.Index(snippet, `"`); endQuote > 0 {
		snippet = snippet[:endQuote]
	}
	if len(snippet) > 200 {
		snippet = snippet[:200] + "..."
	}
	snippet = strings.TrimSpace(snippet)
	if snippet == "" {
		return
	}

	// Find the closest intent node to link to
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
	}
}
