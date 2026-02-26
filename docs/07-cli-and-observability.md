# CLI Interface and Observability

**Files:** [cmd/focus/main.go](../cmd/focus/main.go),
[cmd/focus/inspect.go](../cmd/focus/inspect.go),
[cmd/focus/slash.go](../cmd/focus/slash.go)

Focus Gate exposes a rich set of observability tools across three access patterns:
CLI flags, in-chat slash commands, and a health dashboard. This document covers all of
them, the configuration loading logic, and the panic recovery design.

---

## Binary Modes

```
focus                     Hook mode (default) — reads JSON from stdin
focus --reset             Delete all data files
focus --status            Print current context block and exit
focus --inspect           Full state dump (human-readable text)
focus --inspect --json    Full state dump (JSON)
focus --dry-run "prompt"  Preview classification without mutation
focus --dry-run "prompt" --json  Same, as JSON
```

The mode dispatch is in [main.go:172-195](../cmd/focus/main.go#L172):

```go
switch os.Args[1] {
case "--reset":    return handleReset(p)
case "--status":   return handleStatus(p, cfg)
case "--inspect":  return handleInspect(p, cfg, jsonOutput)
case "--dry-run":  return handleDryRun(p, cfg, prompt, jsonOutput)
}
// Default: hook mode
return handlePrompt(p, cfg)
```

---

## Panic Recovery

The `main()` function wraps everything in a deferred recover:

```go
// main.go:147-158
func main() {
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
```

This is a critical safety property: **Focus Gate must never block the user's prompt**.
If any code panics (nil pointer dereference, slice out of bounds, etc.), the panic is
caught, logged to stderr, and the process exits cleanly. Claude Code continues with no
output from Focus Gate — not ideal, but infinitely better than hanging the user's
workflow.

Note the scope: `recover()` only catches panics in the same goroutine. Focus Gate is
single-threaded, so this covers everything.

---

## Configuration Loading: Two-Phase JSON

**File:** [main.go:82-138](../cmd/focus/main.go#L82)

This is one of the most elegant solutions in the codebase. The problem: how do you
distinguish `{"transitionBoost": 0}` (user explicitly disabled the feature) from
`{}` (user didn't set the field, should use default 0.2)?

Standard `json.Unmarshal` into a struct produces `0` in both cases — you cannot tell
which case you're in.

### The Solution

```go
func loadConfig(path string) config {
    cfg := defaultConfig()  // Start with defaults

    // Phase 1: Detect which keys the user explicitly set
    raw := make(map[string]json.RawMessage)
    persist.Load(path, &raw)

    if len(raw) == 0 {
        return cfg  // No config file — use all defaults
    }

    // Phase 2: Parse into full struct
    var userCfg config
    persist.Load(path, &userCfg)

    // Phase 3: Apply only explicitly written keys
    if _, ok := raw["memorySize"]; ok {
        cfg.MemorySize = userCfg.MemorySize
    }
    if _, ok := raw["transitionBoost"]; ok {
        cfg.TransitionBoost = userCfg.TransitionBoost  // Even if 0!
    }
    // ... etc. for all fields
```

**Phase 1** loads into `map[string]json.RawMessage`. The map only contains keys
actually present in the JSON. An absent key is simply not in the map.

**Phase 2** loads into the typed `config` struct for proper type conversion.

**Phase 3** applies only the keys found in Phase 1. If `"transitionBoost"` is in the
raw map (even as `0`), it's applied. If it's absent, the default is kept.

The nested `similarity` object requires special handling:

```go
// main.go:125-134
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
```

This handles partial similarity objects: `{"similarity": {"extend": 0.6}}` correctly
updates only `extend`, leaving `branch` at its default.

---

## Path Resolution

All file paths are resolved relative to the binary's directory, not the working
directory. This is essential because Claude Code may run the hook from any directory.

```go
// main.go:30-45
func resolvePaths() paths {
    exe, err := os.Executable()
    if err != nil { exe = "." }
    dir := filepath.Dir(exe)
    dataDir := filepath.Join(dir, "data")
    return paths{
        intentFile: filepath.Join(dataDir, "intent.json"),
        engineFile: filepath.Join(dataDir, "engine.json"),
        guideFile:  filepath.Join(dataDir, "guide.json"),
        markovFile: filepath.Join(dataDir, "markov.json"),
        configFile: filepath.Join(dir, "config.json"),
    }
}
```

`os.Executable()` returns the path to the binary, even when invoked from a shell script
or another directory. `filepath.Dir` extracts the containing directory.

---

## In-Chat Slash Commands

**File:** [cmd/focus/slash.go](../cmd/focus/slash.go)

Slash commands intercept the hook pipeline early:

```go
// main.go:269-271
if cmd, ok := parseSlashCommand(input.Prompt); ok {
    return handleSlashCommand(cmd, p, cfg)
}
```

If the prompt starts with `/focus`, it's parsed as a slash command and handled
entirely without mutating any state. No forest changes, no saves.

### Parsing

```go
// slash.go:30-58
func parseSlashCommand(raw string) (slashCommand, bool) {
    trimmed := strings.TrimSpace(raw)
    lower := strings.ToLower(trimmed)

    if !strings.HasPrefix(lower, "/focus") {
        return slashCommand{}, false
    }
    // Ensure "/focus" is the complete word
    if len(lower) > len("/focus") && lower[len("/focus")] != ' ' {
        return slashCommand{}, false
    }
    // Extract subcommand and argument
    rest := strings.TrimSpace(trimmed[len("/focus"):])
    if rest == "" {
        return slashCommand{sub: "help"}, true
    }
    parts := strings.SplitN(rest, " ", 2)
    return slashCommand{sub: strings.ToLower(parts[0]), arg: ...}, true
}
```

The `lower[len("/focus")] != ' '` check prevents `/focusgate` from matching as a
`/focus` command.

### Available Commands

#### `/focus status`

Generates and prints the same context block that would be prepended to the next prompt.
Uses `gate.GenerateContext()` to produce the output.

Implemented at [slash.go:112-128](../cmd/focus/slash.go#L112).

#### `/focus inspect`

Delegates to `inspectText(f, e, g, c, cfg)` from `inspect.go`, which produces a
comprehensive multi-section dump:

```
=== Focus Gate State ===

[Forest]
  Trees: 3  |  Nodes: 12  |  Prompts: 45
  ...

  Tree #0: auth | token | oauth [score=0.82, nodes=5]
    ├── [leaf] fix the oauth token refresh bug [f=3, score=0.71]
    └── [leaf] add token expiry validation [f=1, score=0.45]

[TF-IDF Engine]
  Documents: 45  |  Vocabulary: 128 unique terms
  ...

[Guide]
  15 entries (3 active, 12 pruned links)
  ...

[Markov Chain]
  3 topics, 12 transitions
  ...
```

#### `/focus tree [N]`

Deep-dive into a specific tree:

```go
// slash.go:133-244
func slashTree(w *os.File, f *forest.Forest, e *tfidf.Engine, cfg config, arg string) error {
```

With no argument, lists all trees with their IDs, scores, and node counts. With a
number or partial ID, shows:
- Full node hierarchy (recursive `writeNodeTree`)
- Root vector terms (TF-IDF weights)
- Per-leaf vectors and scores
- Pruning candidates (lowest-scoring leaves, labeled `[PRUNE?]`)

#### `/focus terms [N]`

Displays the TF-IDF vocabulary sorted by document frequency, with IDF values:

```
=== TF-IDF Vocabulary: 45 docs, 128 unique terms ===
  Showing top 30 by document frequency:

  TERM                     DF      IDF
  ----                     --      ---
  token                    12  1.9534
  authenticat               8  2.4854
  ...
```

Implemented at [slash.go:250-273](../cmd/focus/slash.go#L250).

#### `/focus markov`

Prints the transition matrix in human-readable form:

```
=== Markov Transition Matrix ===
  Last topic: m0x1a2b (auth | token | oauth)

  m0x1a2b (auth | token | oauth) ->
    m0x3c4d (database | query | index): 8/12 (67%)
    m0x5e6f (api | endpoint | route): 4/12 (33%)
```

Implemented at [slash.go:279-336](../cmd/focus/slash.go#L279).

#### `/focus score "prompt"`

Runs `gate.DryRun(prompt)` and formats the result:

```
=== Score ===
  Prompt: "fix the oauth token validation"
  Tokens: [fix, oauth, token, valid]

  TF-IDF Vector (4 terms):
    oauth                0.3821
    token                0.2341
    valid                0.2104
    fix                  0.1034

  Thresholds: extend >= 0.550, branch >= 0.250

  Tree #0 "auth | token | oauth"  [boost=1.120]
    Root m0x4d5e6f  cosine=0.6821  boosted=0.7639
    Leaf m0xa1b2c3  cosine=0.7234  boosted=0.8102  "fix oauth ..."  <- BEST
    Leaf m0xd4e5f6  cosine=0.2103  boosted=0.2355  "add token expiry..."

  Result: extend (score=0.8102)
```

Implemented at [slash.go:341-397](../cmd/focus/slash.go#L341).

#### `/focus health`

The system diagnostics command. Implemented at [slash.go:402-571](../cmd/focus/slash.go#L402).

```
=== Focus Health ===

  Memory:  12/100 nodes (12%) [██░░░░░░░░░░]

  Trees:   3 (nodes per tree: min=2 avg=4.0 max=6, max depth=3)
  Prompts: 45

  TF-IDF:  45 docs, 128 unique terms
           23 terms with df=1 (noise: 18%)

  Tree activity:
    #0 [HOT]  score=0.821  age=2m    "auth | token | oauth"
    #1 [WARM] score=0.412  age=6.3h  "database | query | index"
    #2 [COLD] score=0.089  age=38.1h "ci | pipeline | deploy"

  Pruning forecast (lowest-scoring leaves):
    [PRUNE?] tree#2 m0xd4e5f6  score=0.0023  "add github actions for ..."
    [PRUNE?] tree#1 m0xa1b2c3  score=0.0156  "optimize database query..."

    88 slots remaining before pruning triggers.

  Markov:  3 topics tracked, last=m0x1a2b (auth | token | oauth)
```

Key sections:

- **Memory bar** — visual `[████░░░░░░░░]` shows pressure at a glance
- **Tree balance** — detects unbalanced forests (one huge tree vs. many small ones)
- **Term diversity** — `df=1` terms are corpus noise (appeared in only one prompt);
  high noise percentage means many one-off prompts
- **Tree activity** — HOT (< 4h), WARM (4-24h), COLD (> 24h) classification
- **Pruning forecast** — shows which nodes would be removed next and how many slots remain

The **`memoryBar` helper** at [slash.go:612-619](../cmd/focus/slash.go#L612):

```go
func memoryBar(pct int) string {
    const width = 12
    filled := pct * width / 100
    return "[" + strings.Repeat("█", filled) + strings.Repeat("░", width-filled) + "]"
}
```

---

## --reset

```go
// main.go:197-204
func handleReset(p paths) error {
    persist.Remove(p.intentFile)
    persist.Remove(p.engineFile)
    persist.Remove(p.guideFile)
    persist.Remove(p.markovFile)
    fmt.Fprint(os.Stdout, "[Focus] Reset complete. All tracking data cleared.\n")
    return nil
}
```

Deletes all four data files. The config file (`config.json`) is deliberately not removed
— it contains user preferences, not learned state. After reset, the next prompt starts
from a completely blank state as if the first run.

---

## Error Strategy: Log to Stderr, Don't Block

Throughout the codebase, errors are logged to stderr rather than returned:

```go
// main.go:210-213
func logLoadErr(name string, err error) {
    if err != nil {
        fmt.Fprintf(os.Stderr, "focus-gate: load %s: %v\n", name, err)
    }
}
```

Claude Code's hook contract: stdout is the context to inject; stderr is for diagnostics.
Errors on stderr are visible to the developer but don't affect the hook's output. The
system continues with empty/default state rather than failing hard.

The only hard exit is for fundamental failures like "can't read stdin":

```go
// main.go:249-254
data, err := io.ReadAll(os.Stdin)
if err != nil {
    return fmt.Errorf("read stdin: %w", err)
}
```

This is correct because without stdin, there's no prompt to process at all.
