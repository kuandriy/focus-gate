# Focus Gate: Project Overview

## What Is This?

**Focus Gate** is a local, dependency-free Go program that functions as an intelligent
_prompt intent tracker_ for AI coding assistants, specifically Claude Code. It runs as a
**`UserPromptSubmit` hook**: before every prompt you type is sent to the LLM, Focus Gate
intercepts it, classifies its intent, maintains a live model of the conversation's topic
structure, and injects a compact context block back into the prompt.

The system answers a fundamental question that LLMs cannot answer for themselves:
_"What have we been talking about, at what level of detail, and what is related to what?"_

---

## The Hook Mechanism

Claude Code supports hooks — external processes that receive the prompt on `stdin`, do
something, and write output back to `stdout`, which is then prepended to the prompt.
Focus Gate registers itself as a `UserPromptSubmit` hook.

```
User types prompt
       │
       ▼
┌──────────────────────────────────────────────────────┐
│  Focus Gate (UserPromptSubmit hook)                  │
│                                                       │
│  1. Read JSON from stdin  {prompt, transcript_path}   │
│  2. Classify prompt intent                            │
│  3. Mutate forest                                     │
│  4. Generate [Focus...] context block                 │
│  5. Write context block to stdout                     │
└──────────────────────────────────────────────────────┘
       │
       ▼
 Context block prepended to prompt
       │
       ▼
 LLM receives: [Focus | 12 prompts | 8/100 mem | 3 trees]
               ...topic tree summary...
               [/Focus]
               <original user prompt>
```

The hook input JSON is defined in [cmd/focus/main.go:141-144](../cmd/focus/main.go#L141):

```go
type hookInput struct {
    Prompt         string `json:"prompt"`
    TranscriptPath string `json:"transcript_path"`
}
```

`TranscriptPath` points to Claude Code's conversation transcript file. Focus Gate reads
the last assistant message from it to feed the Guide (see [docs/08-feedback-loop.md](./08-feedback-loop.md)).

---

## High-Level Architecture

The system is organized into seven independent packages under `internal/`, orchestrated
by the CLI in `cmd/focus/`.

```
cmd/focus/
├── main.go          Hook entry point, config loader, prompt handler
├── inspect.go       --inspect / --dry-run CLI commands
└── slash.go         In-chat /focus <sub> commands

internal/
├── forest/          Intent forest data structure
│   ├── node.go      Atomic node with decay scoring
│   ├── tree.go      Rooted topic tree
│   ├── forest.go    Forest collection with pruning
│   └── heap.go      Min-heap for O(log n) pruning
│
├── tfidf/           TF-IDF vectorization engine
│   ├── engine.go    Incremental document frequency tracking
│   └── vector.go    Sparse vector + zero-allocation cosine similarity
│
├── text/            Text normalization pipeline
│   ├── tokenizer.go Lowercase, split, stop-word filter, TermFrequency
│   └── stemmer.go   Two-pass lightweight stemmer with override map
│
├── gate/            Classifier and context generator
│   ├── gate.go      ProcessPrompt, classify, apply, bubbleUp, ReinforceFromGuide
│   └── dryrun.go    Non-mutating scoring trace
│
├── guide/           AI response ring buffer
│   └── guide.go     Add, Render, UnreinforcedEntries
│
├── markov/          Topic transition model
│   └── chain.go     Sparse transition matrix, Record, Probability, TopTransitions
│
└── persist/         Atomic JSON I/O
    └── persist.go   SaveAtomic (Windows-safe), RecoverTmpFiles, Load
```

---

## Data Flow on Every Prompt

Below is the exact sequence executed in
[cmd/focus/main.go:247-328](../cmd/focus/main.go#L247):

```
stdin JSON
    │
    ├── text.CleanPrompt(input.Prompt)        Strip IDE tags
    │
    ├── parseSlashCommand()                   /focus <sub> → short-circuit
    │
    ├── persist.Load(intentFile, &f)          Restore forest
    ├── persist.Load(engineFile, &e)          Restore TF-IDF engine
    ├── persist.Load(guideFile, &g)           Restore guide ring buffer
    ├── persist.Load(markovFile, &c)          Restore Markov chain
    │
    ├── updateGuide(g, transcriptPath, f)     Harvest last AI response
    │
    ├── gate.ReinforceFromGuide(g)            AI responses → tree touches
    │
    ├── gate.ProcessPrompt(prompt, source)
    │       │
    │       ├── text.Tokenize(prompt)
    │       ├── engine.VectorizeTokens(tokens)
    │       ├── classify(vec)                 Score all trees + Markov boost
    │       ├── apply(cls, ...)               Mutate forest
    │       ├── Chain.Record(lastTopic, treeID)
    │       ├── engine.AddDocument(tokens)
    │       ├── forest.Prune(memorySize, ...)  If over limit
    │       └── GenerateContext()
    │
    ├── g.Render(f)                           Append guide to context
    │
    ├── persist.SaveAtomic(intentFile, f)
    ├── persist.SaveAtomic(engineFile, e)
    ├── persist.SaveAtomic(guideFile, g)
    ├── persist.SaveAtomic(markovFile, c)
    │
    └── stdout → context block
```

---

## Persistent State Files

All state lives in a `data/` directory alongside the binary:

| File | Content | Struct |
|------|---------|--------|
| `data/intent.json` | The topic forest | `forest.Forest` |
| `data/engine.json` | TF-IDF document frequencies | `tfidf.Engine` |
| `data/guide.json` | AI response ring buffer | `guide.Guide` |
| `data/markov.json` | Transition counts and totals | `markov.Chain` |
| `config.json` | User configuration (optional) | `config` struct |

---

## Configuration

The full config struct is defined in [cmd/focus/main.go:48-60](../cmd/focus/main.go#L48)
and defaults in [cmd/focus/main.go:62-75](../cmd/focus/main.go#L62):

| Key | Default | Meaning |
|-----|---------|---------|
| `memorySize` | `100` | Max total nodes across all trees before pruning |
| `decayRate` | `0.05` | Exponential decay rate per hour for node scoring |
| `similarity.extend` | `0.55` | Cosine threshold to extend an existing leaf |
| `similarity.branch` | `0.25` | Cosine threshold to branch under a root |
| `contextLimit` | `600` | Max bytes in the generated context block |
| `bubbleUpTerms` | `6` | Terms to include in a synthetic parent abstraction |
| `maxSourcesPerNode` | `20` | Max prompt-source labels stored per node |
| `guideSize` | `15` | Ring-buffer capacity for AI response summaries |
| `transitionBoost` | `0.2` | Markov boost weight α in `score *= (1 + α*P)` |

The config loader uses a clever two-phase approach (see [docs/07-cli-and-observability.md](./07-cli-and-observability.md))
to distinguish an explicitly-written `0` from an absent field, so that
`"transitionBoost": 0` correctly disables the Markov boost rather than silently reverting
to the default.

---

## Key Design Decisions

### 1. No External Dependencies

The entire project uses only the Go standard library. No NLP libraries, no embedding
models, no databases. The tradeoff is a simpler, lighter system that can be cross-compiled
to any platform with a single `go build`.

### 2. Bounded Memory

The forest is capped at `memorySize` nodes. When the cap is exceeded, the
pruning algorithm ([forest/forest.go:48-163](../internal/forest/forest.go#L48))
removes the lowest-scoring leaves first, with parent cascading — a leaf's removal can
expose its parent as a new pruning candidate. This ensures the forest never grows
unboundedly, regardless of session length.

### 3. Multiplicative Markov Boost

The Markov chain boosts classification scores multiplicatively:
`score *= (1 + α × P(tree | last_topic))`. Multiplicative form is essential: if the raw
cosine is zero, no amount of history can create a spurious match. The boost only
amplifies existing semantic similarity, it cannot invent it.
See [gate/gate.go:175-232](../internal/gate/gate.go#L175).

### 4. Indexed Flag on Nodes

Nodes hold an `Indexed bool` field ([forest/node.go:29](../internal/forest/node.go#L29)).
Only nodes containing real user prompt text are indexed in the TF-IDF engine; synthetic
`bubbleUp` abstractions are not. This prevents the pruning code from calling
`RemoveDocument` on terms that were never added, which would corrupt IDF values.

### 5. Windows-Aware Atomic Saves

`os.Rename` on Windows fails if the target file already exists (unlike POSIX where rename
is atomic and replaces the target). The persistence layer handles this explicitly in
[persist/persist.go:34-38](../internal/persist/persist.go#L34).

---

## Package Dependency Graph

```
cmd/focus
    ├── internal/forest
    ├── internal/tfidf
    │       └── internal/text
    ├── internal/text
    ├── internal/gate
    │       ├── internal/forest
    │       ├── internal/tfidf
    │       ├── internal/text
    │       ├── internal/guide
    │       └── internal/markov
    ├── internal/guide
    │       └── internal/forest
    ├── internal/markov
    └── internal/persist
```

No circular dependencies. Each internal package is independently testable and has its
own `*_test.go` file.

---

## Module and Build

Module path: `github.com/kuandriy/focus-gate` (see `go.mod`).
Go version: `1.23.4`.
Build: `go build ./cmd/focus/` → single binary.
Tests: `go test ./...` → 71 test cases across all packages.
