# Focus Gate — Documentation Index

This directory contains a detailed technical review of the Focus Gate codebase.
Each document covers one layer of the system with direct references to source code
line numbers.

---

## Documents

| # | File | What It Covers |
|---|------|---------------|
| 1 | [01-overview.md](./01-overview.md) | Architecture, hook mechanism, data flow, configuration reference |
| 2 | [02-data-structures.md](./02-data-structures.md) | `Node`, `Tree`, `Forest`, `LeafHeap` — fields, invariants, O-complexity |
| 3 | [03-algorithms.md](./03-algorithms.md) | TF-IDF, cosine similarity, decay scoring, Markov chain, bubbleUp, pruning |
| 4 | [04-classification-pipeline.md](./04-classification-pipeline.md) | `ProcessPrompt`, `classify`, `apply`, `preserveRoot`, `DryRun` |
| 5 | [05-text-processing.md](./05-text-processing.md) | `CleanPrompt`, `Tokenize`, stop words, two-pass stemmer, override map |
| 6 | [06-persistence.md](./06-persistence.md) | `SaveAtomic`, `RecoverTmpFiles`, Windows rename semantics, JSON layout |
| 7 | [07-cli-and-observability.md](./07-cli-and-observability.md) | CLI flags, slash commands, two-phase config loading, panic recovery |
| 8 | [08-feedback-loop.md](./08-feedback-loop.md) | Guide ring buffer, `ReinforceFromGuide`, bidirectional prompt↔response loop |
| 9 | [09-code-quality.md](./09-code-quality.md) | Design decisions, test coverage, failure modes, style conventions |

---

## Quick Reference: Key Source Locations

| Concept | File | Lines |
|---------|------|-------|
| Node scoring formula | [forest/node.go](../internal/forest/node.go) | 60–68 |
| Pruning with parent cascade | [forest/forest.go](../internal/forest/forest.go) | 48–163 |
| IDF virtual floor | [tfidf/engine.go](../internal/tfidf/engine.go) | 9–14 |
| Zero-allocation cosine | [tfidf/vector.go](../internal/tfidf/vector.go) | 37–74 |
| Tokenizer + stop words | [text/tokenizer.go](../internal/text/tokenizer.go) | 9–71 |
| Two-pass stemmer | [text/stemmer.go](../internal/text/stemmer.go) | 38–64 |
| Stemmer override map | [text/stemmer.go](../internal/text/stemmer.go) | 18–30 |
| Gate: classify() | [gate/gate.go](../internal/gate/gate.go) | 182–233 |
| Gate: apply() | [gate/gate.go](../internal/gate/gate.go) | 236–276 |
| Gate: preserveRoot() | [gate/gate.go](../internal/gate/gate.go) | 281–297 |
| Gate: bubbleUp() | [gate/gate.go](../internal/gate/gate.go) | 299–377 |
| Gate: GenerateContext() | [gate/gate.go](../internal/gate/gate.go) | 380–486 |
| Gate: ReinforceFromGuide() | [gate/gate.go](../internal/gate/gate.go) | 499–547 |
| DryRun | [gate/dryrun.go](../internal/gate/dryrun.go) | 62–158 |
| Guide ring buffer | [guide/guide.go](../internal/guide/guide.go) | 34–60 |
| Guide: Render + dead links | [guide/guide.go](../internal/guide/guide.go) | 77–106 |
| Markov: self-skip | [markov/chain.go](../internal/markov/chain.go) | 31–34 |
| Markov: PruneTopic | [markov/chain.go](../internal/markov/chain.go) | 101–126 |
| Atomic save (Windows) | [persist/persist.go](../internal/persist/persist.go) | 17–39 |
| Tmp file recovery | [persist/persist.go](../internal/persist/persist.go) | 45–69 |
| Two-phase config load | [cmd/focus/main.go](../cmd/focus/main.go) | 82–138 |
| Panic recovery | [cmd/focus/main.go](../cmd/focus/main.go) | 147–157 |
| updateGuide (transcript) | [cmd/focus/main.go](../cmd/focus/main.go) | 334–411 |
| /focus health | [cmd/focus/slash.go](../cmd/focus/slash.go) | 402–571 |
| /focus score | [cmd/focus/slash.go](../cmd/focus/slash.go) | 341–397 |
