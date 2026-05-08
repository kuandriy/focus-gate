# Focus Memory

**Stages 2–3 of Focus Gate's organic refinement pipeline. Where transient conversation focus crystallizes into durable, indexable, append-only stories — and where developers turn personal memory into a living company knowledge base.**

---

## Table of Contents

- [The story](#the-story)
- [Why we don't score importance](#why-we-dont-score-importance)
- [Why a Memory Layer?](#why-a-memory-layer)
- [Story format](#story-format)
- [The Index Tree](#the-index-tree)
- [Surface: traversing the index](#surface-traversing-the-index)
- [Lifecycle: how a memory is born](#lifecycle-how-a-memory-is-born)
- [Sources: personal, domain, anything attachable](#sources-personal-domain-anything-attachable)
- [Personal study: writing without contaminating shared sources](#personal-study-writing-without-contaminating-shared-sources)
- [Contributing back to a shared source](#contributing-back-to-a-shared-source)
- [The company knowledge base vision](#the-company-knowledge-base-vision)
- [Slash command reference](#slash-command-reference)
- [Configuration](#configuration)
- [Anti-goals](#anti-goals)
- [Implementation details](#implementation-details)

---

## The story

Imagine the trail of every coding session you've ever had. Day one: you set up authentication. Day forty: you debug the same authentication system, but you've forgotten why you picked RS256. Day three hundred: a new teammate asks why the refresh token rotates on use, and the original conversation is buried under a year of unrelated work. The decisions are still in the code, but the *reasoning* is gone.

Focus Memory is the part of Focus Gate that refuses to let that happen — and refuses to do it the obvious wrong way. The obvious wrong way is to record everything: every session transcribed, every decision logged, every keystroke archived. That gives you a giant pile of text that grows linearly with how much you work and is, in practice, unreadable. The valuable signal drowns in the volume.

Focus Memory takes the opposite bet. **The user is the only honest judge of what matters.** You don't tell the system what's important. You just work. The [sliding-window forest](sliding-window-intent-forest.md) refines first — topics you keep returning to grow weight, topics you mention once fade. When a topic finally falls out of the working set, only what mattered enough to survive that pruning is even *considered* for crystallization. That's refinement number one.

Refinement number two happens at crystallization itself. A surviving candidate isn't blindly written to disk. It's compared against the index of every memory you already have, and the system asks: is this a continuation of something you've already started, or genuinely new? Continuations become a new chapter on the existing story. Genuinely-new content becomes a new story. Junk — typo storms, single-prompt curiosities, dead ends — is dropped.

By the time a memory exists in your personal store, two filters have run on it. By the time it shows up in a *shared* memory store, a third has run: the merge review. You may have ten years of valuable experience to commit, but most of it is *already there*. The shared repo's review tooling recognizes overlaps: 99% of your contribution might just bump the "last reported" timestamp on a story everyone already knows — which is itself valuable, because frequency is the cornerstone signal. The 1% that's actually new becomes a fresh entry. Frequently-touched stories accumulate weight; rarely-touched ones fade out of surface ranking even though they remain on disk.

The result is a memory that grows with experience but stays manageable in size and ranking, even after hundreds of sessions, even across years, even across a team. **No content is lost. No intent is lost. But nothing extraneous is ever surfaced.** That is the promise.

A memory is not a transcript. It is a **story** with append-only chapters, a frontmatter index, and a stable identity. New chapters extend; old chapters never edit. Corrections come as new chapters that reference the older ones. The trail itself is the point — five years from now, the second half of a story may matter more than the first, but neither version is rewritten retroactively. It is the closest a software system can come to how human professional knowledge actually accumulates: by doing, by repeating, by occasionally being wrong and noticing.

This document explains how Focus Memory works — first the mental model, then the mechanism. The technical companion to Stage 1 lives in [docs/sliding-window-intent-forest.md](sliding-window-intent-forest.md). The implementation plan, schema deltas, and LLM-facing protocols live in [docs/SHARED_MEMORY_PLAN.md](SHARED_MEMORY_PLAN.md).

---

## Why we don't score importance

Focus Gate has no `importance` field. The forest scores **attention** — frequency, recency, depth — not correctness or importance. We made that choice deliberately, and a careful reader will notice the gap and want it filled. It isn't a gap; it's a deferral. Importance gets judged in three different places, by three different judges, none of which is a number on disk:

- **Promote-time — the LLM is the importance gate.** When a tree is about to be pruned and meets the floor checks, the candidate (its prompts AND the AI's reinforcing replies) is shown to the LLM alongside the full aggregate index of every memory you already have. The LLM decides one of three things: append a chapter to an existing story, create a new story, or discard. That decision *is* the importance judgment. We don't precompute it; we ask, with full context, at the only moment a writer needs to be opinionated.
- **Recall-time — the LLM is the relevance judge.** On every prompt, Surface emits short pointer rows with the title, the matched reasons, the dominant tier, and the `t:N` count (how many times this memory has been brought to mind across all sessions). The LLM reads those rows, decides which pointers are worth opening, and ignores the rest. The cornerstone signal isn't "this story has importance 0.87" — it's "this story keeps coming back; the team keeps reaching for it." Frequency does the talking; the LLM listens.
- **Personal vs. shared — the human-memory analogy.** A *personal* memory store is what one developer compiles from their own real experiences. A *shared* memory store is the analog of a single human memory built from hundreds of overlapping experiences across a team — when many people independently tell the same story, the story doesn't duplicate, it gains weight. The cornerstone test at Stage 4 turns frequency-of-telling into the only importance signal that scales across hundreds of contributors over years. A new chapter that 99% overlaps an existing story is a vote that the story is still load-bearing; only the 1% that's genuinely new becomes a fresh entry. That, too, is importance — emergent, not prescribed.

This is why the chapter `what` field is asked to capture **both** what stuck and what was tried-and-abandoned. Past-tense fact includes the dead ends. The reader (human or model) gets the full trail of how the kept design earned its place; we don't need a separate `outcome: superseded` flag because the next chapter, written when the team retried and learned more, will say so in plain prose. Append-only chapters are the audit trail; they record what humans actually do, which is retell the story differently next time.

---

## Why a Memory Layer?

Your AI assistant's context window forgets. Skills layered on top try to compensate by **prescribing** knowledge — "do X this way." But prescriptions decouple from the codebase the moment the codebase changes, and nobody pays the cost of keeping them in sync.

The Memory Layer takes the opposite approach: it captures **what was actually learned** during your conversations, not what someone thought should be true. Knowledge is refined twice on the way in:

1. **Forest pruning.** Focus Gate's sliding-window Forest already forgets one-off prompts and reinforces topics you keep returning to. Only what mattered enough to survive the prune is a candidate for memory.
2. **Crystallization.** Surviving candidates pass through a review (LLM-mediated) that turns them into stories with explicit indexes. One-off curiosities are discarded. Continuations of existing stories become new chapters. Genuinely new knowledge becomes new stories.

The result is **knowledge**, not skills. Skills are a manifest of how things should be done; knowledge is a record of what was done, why, and what bit. The AI decides whether and how to act on it — never instructed by the memory itself.

---

## Story format

A memory is a Markdown file with frontmatter and an append-only chaptered body. Once a chapter is written, it never edits. Corrections, refinements, and contradictions arrive as new chapters that reference the older ones.

```markdown
---
schemaVersion: "2"
id: "mem_20260322_a1b2c3"
title: "Auth & session model"
version: 3
chapters: 3
created: "2026-03-22T14:00:00Z"
updated: "2026-05-02T09:14:00Z"

timeMarkers:
  - "2026-03-15..2026-03-22"
  - "2026-04-10..2026-04-12"
  - "2026-04-30..2026-05-02"

interests:
  - "session lifecycle"
  - "rate limiting"
  - "/auth/refresh"
  - "RS256"

topics:
  - "JWT authentication with refresh-token rotation"
  - "session expiry handling"
  - "RS256 key management for multi-service auth"

assets:
  - "cmd/api/auth.go"
  - "internal/session/store.go"
  - "POST /auth/refresh"
  - "JWT_PRIVATE_KEY"

topTerms: ["jwt", "session", "refresh", "token", "middleware", "rotation"]
fingerprint: "jwt:0.4821 session:0.3912 refresh:0.3654 ..."
vocabHash: "ed96811f47add21e"
touchedBy: 14
---

## Chapter 1 — 2026-03-22 — Initial design
**Time marker:** 2026-03-15..2026-03-22
**Assets introduced:** cmd/api/auth.go, internal/session/store.go
**Interests:** session lifecycle, RS256
**Topics:** JWT authentication, session model

### What
Used JWT with RS256, 15-minute access tokens issued at /auth/login.

### Why
RS256 over HS256 because multi-service rotation needs asymmetric keypairs.

## Chapter 2 — 2026-04-12 — Refresh token rotation
**Time marker:** 2026-04-10..2026-04-12
**Assets introduced:** POST /auth/refresh
**Interests added:** /auth/refresh, replay protection
**Topics added:** refresh-token rotation

### What
Tried client-side refresh-token rotation first; abandoned after request-ordering races corrupted the rotation chain in two staging incidents. Settled on server-side single-use rotation: /auth/refresh swaps an access token for a new one and atomically rotates the refresh token in store.

### Why
Session length stretched from 15m to "until idle" while keeping tokens short-lived. Server-side state is the only place the rotation chain can be ordered safely under concurrency — client-side won the speed argument and lost the correctness one.

## Chapter 3 — 2026-05-02 — Rate-limit gotcha
**Time marker:** 2026-04-30..2026-05-02
**Assets touched:** middleware/rate_limit.go
**Interests added:** rate limiting
**Topics added:** auth middleware short-circuit on missing Authorization header

### What
Discovered that the rate-limit middleware bumped counters before auth ran,
so unauthenticated traffic could exhaust the per-user budget for an attacker's chosen victim.

### Why
Order of middlewares matters. Auth → rate-limit, never the other way.
```

Notice Chapter 2's `What`: it captures both the design that *stuck* (server-side rotation) and the design that was *tried and abandoned* (client-side rotation). That's how the system records outcome — narratively, in the chapter prose, not in a structural `status: superseded` field. Append-only chapters are the audit trail; the most recent telling wins by recency, the persistently-relevant one wins by `touchedBy` count.

### What's required vs binary-managed

| Field | Required | Mutable? | Note |
|---|---|---|---|
| `id`, `title`, `created` | yes | no | stamped once |
| `version`, `chapters` | yes | counter | bumps on append |
| `updated` | yes | yes | = latest chapter's date |
| `timeMarkers`, `interests`, `topics`, `assets` | yes (≥1 each) | append-only | new entries may arrive; old never disappear |
| `topTerms`, `fingerprint`, `vocabHash` | yes | yes | binary recomputes on every save |
| `touchedBy` | yes | yes | counter, bumps every time the memory is surfaced (the cornerstone signal) |

### Append-only invariants (hard rules)

1. **Chapters never disappear.** Once written, chapter N is part of the file forever.
2. **Chapters never edit.** Corrections go in a *new chapter* — the trail is the point.
3. **Frontmatter list fields only grow.** Removing entries breaks shared-source consumers downstream.
4. **`id` is immutable.** Renames break cross-source references.

These invariants exist so the index can trust what it sees: yesterday's truth is still today's, and a stable `id` means cross-source pointers keep working over time.

---

## The Index Tree

Each memory carries an embedded index in its frontmatter — three tiers, from broadest to most precise:

```
┌──────────────────────────────────────────────────────────────┐
│  TIER 1 — TOPICS                                             │
│  Phrases in user-prompt vocabulary. Cosine-matched.          │
│    e.g.  "session expiry handling"                           │
│          "auth middleware short-circuit"                     │
└──────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────┐
│  TIER 2 — INTERESTS                                          │
│  Things future prompts may want to think about, without      │
│  promising any particular asset is implemented.              │
│    e.g.  "rate limiting"                                     │
│          "/auth/refresh"  (the URL pattern, not the file)    │
│          "RS256"                                             │
└──────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────┐
│  ASSETS                                                      │
│  Exactly-matchable identifiers the story has actually        │
│  worked with.                                                │
│    e.g.  "cmd/api/auth.go"                                   │
│          "POST /auth/refresh"                                │
│          "JWT_PRIVATE_KEY"                                   │
└──────────────────────────────────────────────────────────────┘
```

Tier semantics in plain English:

- **Topics** — *"what is this story about, in the words a developer would type?"* Free-text phrases. Used for fuzzy cosine matching against the prompt vector.
- **Interests** — *"what concepts and entities does this story let me reason about, even if today's prompt doesn't name a specific file?"* Soft-suggestive: a prompt mentioning "rate limiting" should pull this story up even if no asset overlaps.
- **Assets** — *"what concrete things in the codebase / API surface has this story actually touched?"* Exact-match keys. A prompt mentioning `POST /auth/refresh` lands here directly.

### Probability scores (soft suggestion)

Index entries can carry an optional weight in `[0, 1]` — how confidently this entry represents the story. The weight is computed at write time from how often it appears across chapters and how central it is to the story's main thrust. Weights are non-prescriptive — they multiply the cosine score during surface ranking but never veto a match outright.

### Per-memory index, then aggregate index

Each memory's frontmatter is its own index. The Memory Layer also maintains an **aggregate inverted index** across all memories in all attached sources:

```json
{
  "byTopic":    { "session expiry": ["mem_a1b2c3"] },
  "byInterest": { "rate limiting":  ["mem_a1b2c3", "mem_d4e5f6"] },
  "byAsset":    { "cmd/api/auth.go": ["mem_a1b2c3"] }
}
```

The aggregate index is what Surface reads on every prompt — fast, cheap, body-free.

---

## Surface: traversing the index

On every prompt, Focus Gate:

1. **Vectorizes** the prompt and **extracts assets** from its text (file paths, API endpoints, env vars, function names).
2. **Walks the aggregate index** in tier order:
   - Asset hits (exact match) — highest confidence.
   - Topic cosine hits — strong fuzzy match against prompt vocabulary.
   - Interest cosine hits — soft suggestion.
3. **Pools, dedupes, ranks** memories. Final score is `max(asset_match × W_a, topic_cosine × W_t, interest_cosine × W_i, fingerprint_cosine × W_f)`.
4. **Renders pointers** — never bodies — into the Focus block, with provenance and the matched reasons:

```
[Memory ↪ relevant prior context — pointers, not instructions; t:N = times this story has been brought to mind]
  mem_20260322_a1b2c3 [personal] (score 0.92 via asset, t:14) Auth & session model
    matched: asset POST /auth/refresh (1.00), topic session expiry handling (0.62)
    → mem_20260322_a1b2c3.md
  mem_20260415_d4e5f6 [team] (score 0.57 via topic, t:3) Multi-service signing keys
    matched: topic RS256 key rotation (0.71)
    → mem_20260415_d4e5f6.md
```

The arrow line emits the memory's path **relative to its source's memories directory** — bare filename only, no source-name or absolute prefix. The source-name in brackets (`[personal]` / `[team]`) is what tells the AI which directory to look in; resolution into a full path happens via `/focus memory show <id>` or by reading the source's registered path from `fg: memory source list`. Keeping the row short means three matches still fit comfortably under the 600-char block budget.

The framing line at the top is deliberate: it tells the AI **what these pointers are**, **how to read them**, and — critically — **that they are non-prescriptive**. The dominant-tier annotation (`via asset` / `via topic`) signals which match drove the score so the LLM can sanity-check whether the reason is the one that actually matters for this prompt. The `t:N` count is the cornerstone signal: a memory with `t:14` has been brought to mind fourteen times across all your sessions — it has earned its place by being repeatedly relevant, not by being recently authored. Higher `t` = stronger "this story is still load-bearing" signal; the LLM uses it to weight ambiguous pointers. The AI Read-tools whichever pointer looks worth opening; Surface itself never inlines body content.

---

## Lifecycle: how a memory is born

```
┌──────────────────────────────┐
│  TREE ABOUT TO PRUNE         │   (Focus Gate's normal pruning trigger)
└──────────────┬───────────────┘
               │   Forest content scoring + thresholds
               │   (see memory.minLeaves, memory.promotionThreshold)
               ▼
┌──────────────────────────────┐
│  CANDIDATE QUEUED            │   pending_memories.json
└──────────────┬───────────────┘
               │   user runs `fg: memory promote`
               ▼
┌──────────────────────────────┐
│  REVIEW BUNDLE FOR LLM       │
│   • candidate (forest snap)  │
│   • full aggregate index     │   ← bodies NOT included
│   • prompt: continue/new/    │
│     discard?                 │
└──────────────┬───────────────┘
               │   AI replies with `fg: memory commit <json>`
               ▼
┌──────────────────────────────┐
│  REGISTER + WRITE            │
│   • action: append → existing memory gets a new chapter           │
│   • action: create → new memory with full registration metadata   │
│   • action: discard → drop from queue, no write                   │
└──────────────┬───────────────┘
               │   on validation failure → re-prompt LLM with the error,
               │   re-register; retry budget caps loop length
               ▼
┌──────────────────────────────┐
│  CANDIDATE PRUNED FROM QUEUE │
└──────────────────────────────┘
```

### What the LLM sees during review

The LLM is given the candidate **and the full aggregate index** — *no memory bodies*. It decides:

- **Continue an existing story.** The candidate's interests/topics/assets overlap with some memory's index → write a new chapter. Output: `targetId`, chapter content, new metadata to merge.
- **New story.** No meaningful overlap → create a fresh memory with full registration metadata + Chapter 1.
- **Discard.** Junk (typo storm, single-prompt curiosity, no durable signal) → drop from queue.

The candidate carries both the user's prompts AND the AI's reinforcing replies (the Guide), so the LLM sees the full story-shape of the topic, not just one half of the dialogue. The aggregate index lists every memory in every enabled source, so the append-or-create decision is made against the *whole* corpus the user has access to.

When the LLM writes a chapter's `what`, it is asked to capture **both** what stuck and what was tried-and-abandoned. Past-tense fact includes the dead ends — they are how the kept design earned its place. There is no separate `outcome` or `supersedes` field; the narrative carries it. A later chapter that retries and learns more will say so in plain prose, and the index keeps growing.

This is the same loop that runs in bootstrap sessions (see [SHARED_MEMORY_PLAN.md §10](SHARED_MEMORY_PLAN.md)). The LLM is a **student**, not a transcriber: skills, code, docs are inputs to learn from, not source-of-truth recipes to copy.

### Registration retry

When the LLM's commit fails validation (missing required fields, broken append-only invariants, bad frontmatter), Focus Gate replies with the specific error and asks the LLM to fix it. The retry budget is small (default 2). After it's exhausted, the candidate stays in the pending queue and the user is told to inspect.

---

## Sources: personal, domain, anything attachable

A **source** is any directory containing memory files (and optionally a manifest). Personal memory is one source by default. Shared / domain memories live in separate Git repos that you clone locally and attach by path.

The source registry is its own state file at `<dataDir>/sources.json` — *not* part of `config.json`. You shape it via slash commands rather than by hand-editing JSON:

```
/focus memory source attach team /path/to/team-shared-memories
/focus memory source attach alice-snapshot /path/to/alice-published-memories --read-only
/focus memory source disable alice-snapshot
/focus memory source default personal
```

After those commands the registry on disk looks like this (shown for reference; the binary owns the format):

```json
{
  "schemaVersion": "2",
  "default": "personal",
  "sources": [
    { "name": "personal",       "path": "<projectDataDir>/memories",                          "enabled": true,  "writable": true  },
    { "name": "alice-snapshot", "path": "/path/to/alice-published-memories",  "enabled": false, "writable": false },
    { "name": "team",         "path": "/path/to/team-shared-memories",    "enabled": true,  "writable": true  }
  ]
}
```

There is no "personal vs shared" special case in code — every source is a directory. Personal happens to be the one Focus Gate synthesizes on first load if no `sources.json` exists yet, and it cannot be detached (only disabled) so a hand-edit can't accidentally orphan your local memories.

### Attach / detach / enable / disable

| Command | Effect |
|---|---|
| `fg: memory source attach <name> <path> [--read-only]` | Register a new source |
| `fg: memory source detach <name>` | Unregister; files untouched |
| `fg: memory source enable <name>` / `disable <name>` | Toggle without unregistering |
| `fg: memory source default <name>` | Set the destination for new memories |
| `fg: memory source list` | Table: name, path, enabled, writable, count |

`personal` cannot be detached — only disabled — so you don't accidentally lose the path to your local memory directory.

### A/B comparison flow

Run the same prompt against different memory mixes:

```
fg: memory source disable team
# ask your question — only personal surfaces
fg: memory source enable team
fg: memory source disable personal
# ask the same question — only team surfaces
fg: memory source enable personal
```

Useful when you want to know whether a domain source actually helps with a given task, or when you want to "borrow" another developer's published memories without your own muddying the answer.

---

## Personal study: writing without contaminating shared sources

When you run `fg: memory promote`, the LLM sees the **aggregate index across all enabled sources**. It can decide to append a chapter to a memory in *any* writable source — including a shared one. Locally, that's just a write to your clone of the shared repo; it doesn't affect anyone else until you push and PR (see next section).

Sometimes you want to make sure the new chapter lands only in personal — for example, when you're learning a domain and don't yet trust your own conclusions enough to suggest them as company knowledge. The mechanism is the same as A/B comparison: **disable shared sources before promoting**.

```
fg: memory source disable team
fg: memory promote        # candidates promote into personal only
fg: memory source enable team
```

We call this **personal study** — the framing matches the §7 student loop: you're learning the domain by working in it, and the chapters you produce stay in your own scratchpad until you've earned enough confidence to PR them upstream.

---

## Contributing back to a shared source

This part is **deliberately not Focus Gate's concern**. The shared source is just a Git repo. Once you've appended chapters or created memories in your local clone, you push and open a PR like anywhere else:

```bash
cd /path/to/team-shared-memories
git status                              # shows the new chapter / memory file
git checkout -b candidates/your-username-2026-05-02
git add memories/ && git commit
gh pr create
```

The shared repo's review process — chat-bot, human, or both — decides what merges. The PR review works on **memory files**: it can validate frontmatter, run cross-memory consistency checks, suggest condensations across overlapping stories, or flag drift from the rest of the corpus. Focus Gate's contribution is producing well-formed, index-compatible local writes; everything beyond that is the shared repo's tooling.

There is intentionally no `fg: memory share push` and no PR automation in Focus Gate. Removing those keeps the boundary clean: Focus Gate writes locally; Git shares globally; review tooling lives where it belongs.

This is also where the *fourth* refinement fires. You may have ten years of valuable experience to commit, but the shared source's review tooling will recognize that 99% of it is already there. Most of your PR will collapse into "this story is still load-bearing — bump its last-reported timestamp and increment its weight." Only the genuinely-new 1% becomes new stories. That's how the shared corpus stays manageable across hundreds of contributors over years: the merge gate is the cornerstone test, and frequency is the cornerstone signal.

---

## The company knowledge base vision

The trajectory looks like this:

1. **Single developer.** You run Focus Gate. Memories crystallize in your personal source.
2. **Sharing kicks in.** A subset of your memories — domain conventions, recurring decisions, gotchas that aren't tied to your specific work — are obvious candidates to share. You PR them into a company-shared repo.
3. **More developers.** Each developer runs Focus Gate. Each contributes to the shared repo. Curation tooling on the repo side merges, condenses, retires, re-organizes.
4. **A new developer arrives.** They clone Focus Gate, attach the shared source, and on their first session the AI already has access to dozens of crystallized stories about the codebase, the domain, the pitfalls, and the decisions.

This is the explicit alternative to maintaining skills:

| Skills | Memory |
|---|---|
| Manually written, prescriptive | Captured from real work, descriptive |
| Drift from reality between updates | Append-only history; old chapters preserve traceability |
| Hand-curated taxonomy | Index emerges from what was actually used |
| Discrete bundles per task | Stories that span tasks and grow over time |
| Costly to maintain at scale | Maintenance is a side effect of doing the work |

Skills are a **manifest** of how someone thinks knowledge should be organized. Memory is the knowledge itself, captured at the point of being used. The Memory Layer is our bet that the second one wins.

---

## Slash command reference

### Inspect

| Command | Purpose |
|---|---|
| `fg: memory list` | Table of memories across all enabled sources |
| `fg: memory show <id-or-prefix>` | Pretty-print one memory's frontmatter + body |
| `fg: memory pending` | Queued candidates awaiting promotion |
| `fg: memory health` | Counts, stale-ref warnings, manifest state |

### Lifecycle

| Command | Purpose |
|---|---|
| `fg: memory promote [tempId]` | Render pending + aggregate index for the LLM (one prompt per candidate, or just the named one) |
| `fg: memory commit <tempId> '<json>'` | Persist the LLM's append/create/discard decision |
| `fg: memory discard <tempId\|all>` | Clear pending without promoting |
| `fg: memory forget <id-or-prefix> [--yes]` | Delete a memory file + manifest entry. Dry-runs without `--yes`; refuses to remove from read-only sources. |

### Sources

| Command | Purpose |
|---|---|
| `fg: memory source list` | Table: name, enabled, writable, **count** (memory file total per source), path |
| `fg: memory source attach <name> <path> [--read-only]` | Register a new source |
| `fg: memory source detach <name>` | Remove from config |
| `fg: memory source enable <name>` / `disable <name>` | Toggle without detaching |
| `fg: memory source default <name>` | Set the destination for new memories |
| `fg: memory reindex [--source <name>]` | Rebuild a source's manifest + inverted indexes |

### Migration

| Command | Purpose |
|---|---|
| `fg: memory migrate-v1` | Convert v1 single-doc memories to v2 stories. Each migrated file gets a `.v1.bak` backup; explicit invocation only — Focus Gate never auto-rewrites memory files. |

All `/focus memory ...` equivalents route to the same handlers.

---

## Configuration

```json
{
  "memory": {
    "enabled": true,
    "surfaceThreshold": 0.35,
    "topK": 2,
    "maxBlockChars": 600,
    "minLeaves": 4,
    "minPrompts": 3,
    "promotionThreshold": 1.5,
    "rescueThreshold": 1.2,
    "promotionCooldown": "4h",
    "pendingMaxAge": "168h",
    "mergeSuggestCosine": 0.6,
    "commitRetries": 2,
    "autoNudge": true,
    "weights": {
      "asset": 1.0,
      "topic": 0.8,
      "interest": 0.6,
      "fingerprint": 0.4
    }
  }
}
```

> **Note on sources.** The source registry lives in its own state file at `<dataDir>/sources.json`, not in `config.json`. Mutations go through `fg: memory source attach/detach/enable/disable/default`; the registry is synthesized with a single `personal` entry on first load. Hand-editing the JSON is supported but generally not necessary.

### Surface

| Parameter | Default | Description |
|---|---|---|
| `enabled` | `true` | Master switch. `false` → no detection, no surface, slash subcommands respond "disabled." |
| `surfaceThreshold` | 0.35 | Minimum cosine for a memory to surface (post-weighting). |
| `topK` | 2 | Max memories surfaced per prompt. |
| `maxBlockChars` | 600 | Soft cap on the rendered surface block. |
| `weights.asset` | 1.0 | Multiplier for asset-match score (exact match). |
| `weights.topic` | 0.8 | Multiplier for topic-cosine score. |
| `weights.interest` | 0.6 | Multiplier for interest-cosine score. |
| `weights.fingerprint` | 0.4 | Multiplier for aggregate fingerprint cosine (fallback). |

### Candidate selection

| Parameter | Default | Description |
|---|---|---|
| `minLeaves` | 4 | Floor: tree must have this many real leaves to qualify. |
| `minPrompts` | 3 | Floor: tree must have this many indexed prompts. |
| `promotionThreshold` | 1.5 | Minimum candidate score to queue. |
| `rescueThreshold` | 1.2 | Trees below threshold but with historical PeakScore above this are rescued on prune. |
| `promotionCooldown` | `"4h"` | Per-tree cooldown to prevent re-promotion storms. |
| `pendingMaxAge` | `"168h"` | Candidates older than this are dropped on load. |
| `mergeSuggestCosine` | 0.6 | Threshold at which a candidate is suggested to continue an existing memory. |
| `commitRetries` | 2 | Per-candidate retry budget when the LLM emits an invalid `fg: memory commit` payload. After exhaustion the candidate is parked in pending. |
| `autoNudge` | `true` | Append the "N topic(s) queued…" line to the Focus block when pending is non-empty. |

### Sources

The source registry is persisted at `<dataDir>/sources.json`, not in `config.json`. Each entry is `{ name, path, enabled, writable }`; the registry also stores a top-level `default` naming the destination for new memories. Missing → Focus Gate synthesizes `[{name: "personal", path: "<projectDataDir>/memories", enabled: true, writable: true}]` with `default: "personal"`. Mutate via the `fg: memory source ...` slash commands rather than editing JSON.

### Tuning recipes

- **Memories never surface?** Lower `surfaceThreshold` to 0.20. Verify with `fg: memory list` that they exist and have non-empty fingerprints.
- **Too many candidates?** Raise `minLeaves` to 6 and `promotionThreshold` to 2.0.
- **Asset hits too aggressive?** Lower `weights.asset` to 0.8 — equal weighting with topics.
- **Surface block feels noisy?** Lower `topK` to 1.
- **Want to revert to pre-memory behavior?** Set `"enabled": false`.

---

## Anti-goals

- **No body inlining at surface time.** Always pointers + matched reasons. The Focus block size stays predictable.
- **No mutation of past chapters.** Ever. Corrections are new chapters; old chapters are the audit trail.
- **No automatic push to shared repos.** Git is the user's tool, not Focus Gate's. PR review tooling lives in the shared repo.
- **No skill-aware code paths.** Skills are out of scope. The bootstrap that seeds the first shared repo from skill files is a one-time mocked LLM session, not a Focus Gate feature.
- **No cross-project memory sync** beyond what Git gives you.
- **No LLM HTTP client in Focus Gate.** Slash-mediated remains the only LLM integration path. The binary stays single-binary, zero-network, deterministic.
- **No automatic memory deletion.** Stale → flagged in `fg: memory health`. Removal is always a deliberate user action via `fg: memory forget`.

---

## Implementation details

This document is the user-facing mental model. For:

- The phased implementation TODO
- Schema / Go-struct deltas to `internal/memory/`
- The exact prompt template Stage B emits for the LLM
- Open design questions and tradeoffs
- Migration plan from v1 → v2 schema

see **[SHARED_MEMORY_PLAN.md](SHARED_MEMORY_PLAN.md)**.
