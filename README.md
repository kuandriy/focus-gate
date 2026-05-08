# Focus Gate

*Memory that gets sharper with use, not heavier.*

Your AI coding assistant has the wrong notion of memory. Context windows compress, summarize, and eventually forget. The layers most teams add on top — skills, system prompts, hand-curated project docs — try to compensate by **prescribing** knowledge ("do X this way"), and they drift from reality the moment the codebase changes. Session recorders try the opposite — capture everything — and bury the signal under their own transcripts. Neither one survives long-term contact with the work.

Focus Gate takes a third bet: **the user is the only honest signal of what matters.** Every prompt you send is a vote for what's important right now. Every reply the assistant gives is a vote for what's actually being learned. We treat your attention itself as the unit of memory and refine it at every step of its journey from "this prompt" to "this is part of how we work here."

The promise sounds extreme until you watch it run for a few weeks:

- **No user experience is ever lost.** Every session is captured.
- **No intent is ever lost.** When you return to a topic months later, the system recognizes the continuation and updates an existing story instead of duplicating it.
- **Memory stays manageable for years.** Rarely-revisited topics fade out of surface ranking. Frequently-revisited topics gain weight and become cornerstones.
- **Refinement happens organically, in stages.** No backend pipeline decides what's important — *you* do, just by working.

Zero external dependencies. Single binary. Built entirely on Go's standard library. No network calls. No telemetry.

---

## Table of Contents

- [Four stages of organic refinement](#four-stages-of-organic-refinement)
- [What it isn't](#what-it-isnt)
- [What the AI sees](#what-the-ai-sees)
- [Read deeper](#read-deeper)
- [Install](#install)
- [Usage](#usage)
- [Configuration](#configuration)
- [Architecture](#architecture)
- [License](#license)

---

## Four stages of organic refinement

A piece of conversation passes through up to four refineries before it joins long-lived memory. Each one cuts what doesn't matter and reinforces what does. The user's attention is the only judge.

### Stage 1 — Focus Gate (the sliding window)

Every prompt you send and every meaningful AI reply is classified into a **forest of intents** — a small collection of topic trees you're working on right now. Topics you keep returning to gain weight; topics you mention once fade. When the forest fills up, the lowest-scoring leaves are pruned.

This is your working set: small, fresh, sharpened from both ends — your prompts shape it, the assistant's replies feed back to reinforce the trees you're actively climbing. The AI sees a compact summary on every turn, so it always knows what you've been focused on, even on day three of a long thread. Pruning is not loss; it's the first refinery saying *this didn't earn its place yet.*

→ Full mechanism: [docs/sliding-window-intent-forest.md](docs/sliding-window-intent-forest.md)

### Stage 2 — Focus Memory submodule (crystallization)

When a topic falls out of the sliding window but mattered enough — enough leaves, enough touches, enough peak score during its lifetime — it doesn't disappear. It crystallizes into a **story**: an append-only Markdown file with frontmatter that carries an index of what it's about (time markers, interests, topics, assets).

Once a chapter is written, it never edits. Corrections come as new chapters. The story grows; the trail itself is the point. This is the second refinery: *what mattered enough to be revisited becomes durable.*

→ Full mechanism: [docs/memory-focus.md](docs/memory-focus.md)

### Stage 3 — Local memory harvesting (the second refine)

A new candidate isn't blindly written to disk. The system shows the candidate alongside the index of every memory you already have, and the LLM is asked one question: is this a continuation of a story you've already started, or genuinely new?

- Continuations become a **new chapter on the existing story** — frontmatter list fields grow, but identity is preserved.
- Genuinely-new content becomes a **new story** with full registration metadata and Chapter 1.
- One-off curiosities, typo storms, and dead ends are **discarded**.

This is the third refinery, and it is what keeps your personal memory from accumulating duplicates as the same patterns recur over months. By the time something has earned its place, it has survived two filters.

### Stage 4 — Merge into shared memory (the cornerstone test)

When you open a PR against a company-shared memory repo, the same logic runs at a larger scale. You may have ten years of valuable experience to commit, but most of it is *already there*. The shared repo's review tooling recognizes overlaps: 99% of your contribution might just bump the "last reported" timestamp on a story everyone already knows — which is itself valuable, because frequency is the cornerstone signal. The 1% that's actually new becomes a fresh entry. Over time, frequently-touched stories accumulate weight; rarely-touched ones fade out of surface ranking even though they're still on disk.

This is the explicit shared-memory analogy: a *team's* shared memory is one collective brain assembled from hundreds of overlapping experiences across many contributors. When many people independently tell the same story, the story doesn't duplicate — it gains weight. That "already-load-bearing" vote is exactly what makes a shared corpus stay useful over years instead of collapsing under its own volume.

This is the fourth refinery, and it is the only reason a *team's* shared memory can stay useful across hundreds of contributors over years instead of collapsing under its own volume. The merge gate is the cornerstone test, and frequency is its scoring function.

The combined effect of all four stages: a memory that grows with experience but stays manageable in size and surface ranking. Hundreds of sessions don't translate into hundreds of new files. They translate into a small number of well-weighted stories that the team keeps coming back to — exactly the ones that turned out to matter.

---

## What it isn't

- **Not skills.** Skills are someone's *idea* of how things should be done. Memory is what was actually done, why, and what bit you. The AI decides whether and how to act on a memory; the memory itself is descriptive, not prescriptive.
- **Not a session recorder.** Recorders preserve transcripts and ask a backend pipeline to figure out what's important. Focus Gate flips the locus of judgment: *you* refine intent (just by working), *the system* reacts to your attention. There is no separate "importance" model trying to read your mind.
- **Not a RAG store.** Bodies are never inlined into prompts. Each story carries its own index; the AI traverses index pointers and Reads the body on demand only when something looks worth opening.
- **Not an importance scorer.** Focus Gate doesn't try to predict what's important about a topic. The forest tracks attention; the LLM judges fit at promote-time (append / create / discard against the full aggregate index) and judges relevance at recall-time (which pointer is worth opening, weighted by the `t:N` cornerstone count). Personal memory is what one developer compiles from real experience; shared memory is the analog of a single human memory built from hundreds of overlapping experiences across a team — frequency of independent tellings *is* the importance signal, and the only one that scales across years and contributors.

---

## What the AI sees

On every prompt, Focus Gate emits a single compact block of context — the live working-set summary plus pointers into long-term memory. Bodies are never inlined; everything is small enough to read at a glance.

```
[Focus | 23 prompts | 18/100 mem | 3 trees]
  [0.95] token | authentica | session | jwt
    - add refresh token rotation
    - fix the session expiry bug
  [0.82] migrat | schema | user | email
    - add index on email column
[/Focus]

[Memory ↪ relevant prior context — pointers, not instructions; t:N = times this story has been brought to mind]
  mem_20260322_a1b2c3 [personal] (score 0.92 via asset, t:14) Auth & session model
    matched: asset POST /auth/refresh (1.00), topic JWT authentication (0.71)
    → mem_20260322_a1b2c3.md
```

The first half is the sliding-window forest — what you've been focused on, with similarity scores and recent leaves. The second half is the long-term memory pointer index — what previously-crystallized stories may apply, with provenance and the matched reasons. The `t:N` count on each pointer is the cornerstone signal: how many times this memory has been brought to mind across all your sessions; the LLM weights ambiguous pointers by it. The combined block is hard-capped (default 600 chars for the forest summary plus 600 for the memory block) so it never crowds the conversation.

---

## Read deeper

- **[docs/sliding-window-intent-forest.md](docs/sliding-window-intent-forest.md)** — Stage 1 in depth. Story, then the math: TF-IDF classification, cluster merging, decay scoring, bubble-up abstraction, pruning with parent cascading, typo tolerance, session boundaries, bidirectional guide reinforcement, and the in-chat inspector commands.
- **[docs/memory-focus.md](docs/memory-focus.md)** — Stages 2–3 in depth. Story, then the mechanism: append-only chapters, the three-tier index (topics / interests / assets), surface ranking, the candidate-review lifecycle, attaching multiple sources, A/B comparison, personal study, contributing back to a shared source.
- **[docs/SHARED_MEMORY_PLAN.md](docs/SHARED_MEMORY_PLAN.md)** — Implementation plan for the Memory Layer: schema details, the LLM-facing prompt and commit protocol, migration from v1, the bootstrap / learning-loop protocol for Stage 4 contributions to a shared source.

---

## Install

Download the binary for your platform from [Releases](https://github.com/kuandriy/focus-gate/releases), or build from source:

```bash
go build -o focus-gate ./cmd/focus
```

Single binary, zero runtime dependencies, no network calls. Drop it on `$PATH` and reference it from your hook config.

---

## Usage

### As a Claude Code hook

Add to `.claude/settings.local.json` in the target project:

```json
{
  "hooks": {
    "UserPromptSubmit": [
      { "hooks": [ { "type": "command", "command": "/absolute/path/to/focus-gate" } ] }
    ]
  }
}
```

The hook fires on every prompt submission. Focus Gate reads the prompt from stdin, updates state, and writes the context block to stdout. Claude Code receives your prompt with that block prepended.

> Focus Gate is a CLI-level hook. It runs in terminal-based Claude Code sessions. The VSCode extension does not fire `UserPromptSubmit` hooks, so the in-chat `/focus` slash commands and `fg:` aliases route through the hook only when the hook itself runs (CLI sessions).

### In-chat commands

Type any command directly in chat. The command is intercepted before classification — **no state is modified**, the output appears inline as a fenced block.

| CLI form | Alias | Description |
|:---|:---|:---|
| `/focus status` | `fg: status` | Compact context (same block the AI sees) |
| `/focus inspect` | `fg: inspect` | Full state dump |
| `/focus tree [N\|prefix]` | `fg: tree [N\|prefix]` | List or deep-dive into trees |
| `/focus terms [N]` | `fg: terms [N]` | TF-IDF vocabulary |
| `/focus last` | `fg: last` | Recent classifications |
| `/focus score "prompt"` | `fg: score "prompt"` | Dry-run classification |
| `/focus health` | `fg: health` | Diagnostics — pressure, balance, pruning forecast |
| `/focus memory <sub>` | `fg: memory <sub>` | Memory-layer commands (see [docs/memory-focus.md](docs/memory-focus.md)) |
| `/focus help` | `fg: help` | List all commands |

> **Two paths, one handler.** `/focus` is a registered Claude Code slash command. `fg:` is intercepted by the hook itself and works in any environment where the hook runs, including the VSCode extension where the slash picker would otherwise capture `/`. Both route to the same code; pick whichever feels natural. Examples and field-by-field output reference live in the subdocs.

### CLI flags

```bash
./focus-gate --status                                # current context
./focus-gate --inspect [--json]                      # full state dump
./focus-gate --dry-run "prompt text" [--json]        # dry-run classification
./focus-gate --reset                                 # clear all state
./focus-gate --list-projects                         # known projects
echo '{"prompt":"..."}' | ./focus-gate               # hook mode
```

---

## Configuration

Resolution order (first match wins): `.focus-gate.json` in the project, then `$FOCUS_GATE_CONFIG`, then a global `config.json` next to the binary. Only fields present in the file override defaults; an explicit `0` is respected.

The most-tuned knobs:

| Parameter | Default | Description |
|:---|:---:|:---|
| `memorySize` | 100 | Forest node ceiling — trigger for pruning |
| `decayRate` | 0.05 | Per-hour exponential decay; higher = faster forgetting |
| `similarity.extend` / `branch` | 0.55 / 0.25 | Cosine thresholds for "extend leaf" vs "branch into tree" |
| `sessionTimeout` | 4.0 | Hours of inactivity before frequencies halve (`0` disables) |
| `mergeSimilarity` | 0.6 | Two trees merge when their roots are this close (`0` disables) |
| `typoTolerance.*` | see subdoc | Levenshtein-based canonicalization at tokenize time |
| `memory.*` | see subdoc | Surface threshold, candidate floors, sources, weights |

Full tables, defaults, and tuning recipes live in [docs/sliding-window-intent-forest.md §Configuration](docs/sliding-window-intent-forest.md#configuration) and [docs/memory-focus.md §Configuration](docs/memory-focus.md#configuration).

---

## Architecture

```
cmd/focus/         Entry: CLI, stdin/stdout, /focus subcommands
internal/
  text/            Tokenizer, stemmer, stop words, refs, typo canonicalizer
  tfidf/           TF-IDF engine, sparse vectors, cosine merge-join
  forest/          Node, Tree, Forest, heap-based pruning, peak score
  gate/            Classifier (extend / branch / new / continue), bubble-up, merge
  guide/           AI response tracking + forest reinforcement
  memory/          Long-term memory (story format, manifest, surface, candidate review)
  persist/         Atomic JSON, schema versioning, file lock, recovery pass
```

State persists per project under `~/.focus-gate/<sha256(cwd)[:12]>/`, so state never leaks between projects. Override with `--data-dir` or `$FOCUS_GATE_DATA_DIR`. Writes are atomic (`.tmp` + rename) and version-checked on load — corruption never blocks a prompt; the system falls back to empty state and the user can `--reset` if needed.

> **Concurrency.** On macOS and Linux, concurrent hook invocations are serialized by a `flock`-style file lock. On Windows the lock is currently a no-op — concurrent prompt submissions in two terminals could interleave state writes. CLI Claude Code typically issues one hook at a time, so this is rarely observable in practice; if it bites, run prompts serially or open an issue.

| File | Purpose |
|:---|:---|
| `intent.json` | Intent forest — what the user is asking about |
| `engine.json` | TF-IDF document frequency counts |
| `guide.json` | AI response summaries with intent links and reinforcement state |
| `pending_memories.json` | Append-only queue of Stage 1 → Stage 2 candidates |
| `sources.json` | Memory-source registry (attach/detach state, default-source pointer) |
| `memories/` | Long-term memory Markdown files + `index.json` manifest (the synthesized "personal" source) |

The repo also ships a small bootstrap source under [seed-memories/](seed-memories/) — five memories about Focus Gate's own internals. Attach it as a read-only source if you'd like the system to surface its own knowledge alongside your personal notes:

```
/focus memory source attach focus-gate /absolute/path/to/focus-gate/seed-memories --read-only
```

---

## License

MIT
