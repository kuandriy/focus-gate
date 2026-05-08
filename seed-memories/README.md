# Seed memories

Five v2 memory files plus a manifest (`index.json`) describing Focus Gate
itself — the philosophy, sliding-window forest, append-only chapter
format, multi-tier surface scoring, and the candidate-review loop. They
exist so a new install can demonstrate the long-term memory layer end-
to-end without first having to use the tool for weeks to grow a corpus.

## How to use

Attach this directory as a memory source from any project where you'd
like Focus Gate's own knowledge surfaced alongside your personal notes:

```
/focus memory source attach focus-gate /absolute/path/to/seed-memories --read-only
```

Subsequent prompts that mention `internal/memory/`, `surface.go`,
"append-only chapters", or any of the assets/topics indexed in
[index.json](index.json) will surface the corresponding seed memory as
a `[focus-gate]` pointer next to your usual `[personal]` entries.

`--read-only` is recommended — the seed catalog is shipped with the
binary and shouldn't accept new chapters. Detach at any time with
`/focus memory source detach focus-gate`.

## What's in here

| File | Topic |
|---|---|
| `mem_20260507_a1b2c3.md` | Four-stage organic refinement (philosophy) |
| `mem_20260507_b4c5d6.md` | Sliding-window intent forest mechanics |
| `mem_20260507_c7d8e9.md` | Memory v2 format: append-only chapters |
| `mem_20260507_d0e1f2.md` | Multi-tier surface scoring |
| `mem_20260507_e3f4a5.md` | Stage B candidate review (append/create/discard) |

`index.json` is the manifest the binary would generate on first scan.
It is committed alongside the .md files so attaching the directory
produces an immediately-surfaceable index without a rebuild round-trip.

## Regenerating the manifest

If you edit any `.md` file, regenerate the manifest with:

```
/focus memory reindex --source focus-gate
```

(Or delete `index.json` — the next surface call will rebuild it via
`EnsureFresh` since the directory contents will appear newer than the
missing index.)
