# Persistence Layer

**File:** [internal/persist/persist.go](../internal/persist/persist.go)

Focus Gate maintains its entire state across sessions through atomic JSON files.
The persistence layer solves three problems: safe writing (atomic updates), graceful
startup recovery (handling interrupted writes), and cross-platform compatibility
(Windows vs. Unix rename semantics).

---

## The Atomic Write Problem

When saving state, the naive approach is:

```
os.WriteFile("intent.json", data, 0644)
```

This is **not safe**. If the process is killed mid-write, the file is left partially
written and corrupted. On the next startup, `json.Unmarshal` would fail, and the user
would lose all their conversation intent history.

The correct approach is **write-then-rename**:

```
1. Write data to "intent.json.tmp"
2. Rename "intent.json.tmp" → "intent.json"
```

On POSIX systems (Linux, macOS), `rename(2)` is an atomic kernel operation — the old
file is replaced instantaneously from the filesystem's perspective. If the process dies
between step 1 and step 2, the `.tmp` file exists with complete data and the original
`intent.json` is untouched.

---

## SaveAtomic

```go
// persist.go:17-39
func SaveAtomic(path string, v any) error {
    dir := filepath.Dir(path)
    if err := os.MkdirAll(dir, 0755); err != nil {
        return err
    }

    data, err := json.MarshalIndent(v, "", "  ")
    if err != nil {
        return err
    }

    tmp := path + ".tmp"
    if err := os.WriteFile(tmp, data, 0644); err != nil {
        return err
    }

    // On Windows, os.Rename fails when the target already exists.
    if runtime.GOOS == "windows" {
        _ = os.Remove(path)
    }

    return os.Rename(tmp, path)
}
```

### Step by Step

1. **`MkdirAll(dir, 0755)`** — Creates the `data/` directory if it doesn't exist.
   First run creates the directory automatically. Permission 0755 = owner can
   read/write/execute, group/others can read/execute.

2. **`json.MarshalIndent(v, "", "  ")`** — Produces pretty-printed JSON. Using
   indented JSON rather than compact JSON has two benefits:
   - Files are human-readable in a text editor for debugging
   - Git diffs are meaningful if the user version-controls their data directory

3. **`os.WriteFile(tmp, data, 0644)`** — Writes to `intent.json.tmp`. Permission 0644
   = owner can read/write, group/others can read. If this fails (e.g. disk full), the
   original file is untouched.

4. **Windows `os.Remove(path)`** — The critical Windows-specific step. On Windows,
   `os.Rename` returns an error if the destination file already exists. There is no
   atomic-replace equivalent to POSIX rename on Windows without using advanced Win32 API
   calls. The workaround: remove the target first, then rename.

   This creates a brief window where neither file exists. Hence the recovery mechanism
   below.

5. **`os.Rename(tmp, path)`** — On Unix: atomic replace. On Windows: move after the
   remove.

### Error Handling at Call Sites

Errors from `SaveAtomic` are logged to stderr but not returned as fatal:

```go
// main.go:312-323
if err := persist.SaveAtomic(p.intentFile, f); err != nil {
    fmt.Fprintf(os.Stderr, "focus-gate: save intent: %v\n", err)
}
```

A save failure is not catastrophic — the user's prompt still proceeds. The next prompt
will attempt to save again. This design prioritizes **not blocking the user** over
**guaranteed persistence**.

---

## RecoverTmpFiles — Startup Recovery

```go
// persist.go:45-69
func RecoverTmpFiles(paths ...string) {
    for _, path := range paths {
        tmp := path + ".tmp"
        tmpExists := exists(tmp)
        targetExists := exists(path)

        if !tmpExists {
            continue
        }

        if !targetExists {
            // .tmp without target — interrupted save. Promote .tmp to target.
            if err := os.Rename(tmp, path); err != nil {
                fmt.Fprintf(os.Stderr, "focus-gate: recover %s: %v\n", path, err)
            } else {
                fmt.Fprintf(os.Stderr, "focus-gate: recovered %s from tmp\n", path)
            }
        } else {
            // Both exist — target is authoritative, remove stale .tmp.
            if err := os.Remove(tmp); err != nil {
                fmt.Fprintf(os.Stderr, "focus-gate: cleanup %s.tmp: %v\n", path, err)
            }
        }
    }
}
```

Called at startup in [main.go:164](../cmd/focus/main.go#L164) before any `Load`:

```go
persist.RecoverTmpFiles(p.intentFile, p.engineFile, p.guideFile, p.markovFile)
```

### Recovery Decision Table

| `.tmp` exists | Target exists | Action |
|---------------|---------------|--------|
| No | — | Do nothing (normal case) |
| Yes | No | **Promote .tmp** (interrupted Windows save) |
| Yes | Yes | **Remove .tmp** (stale from previous crash before rename) |

**Case 2 (Promote):** The Windows-specific window between `os.Remove(path)` and
`os.Rename(tmp, path)`. If the process dies here, `.tmp` has the complete new data and
the original is gone. Promoting `.tmp` restores the most recent complete save.

**Case 3 (Remove):** The process was killed after writing `.tmp` but before `os.Rename`.
Both files exist. The target is authoritative (it was the last successfully saved state).
The `.tmp` is stale and should be removed.

---

## Load — Graceful Missing File

```go
// persist.go:73-82
func Load(path string, v any) error {
    data, err := os.ReadFile(path)
    if err != nil {
        if errors.Is(err, os.ErrNotExist) {
            return nil  // Graceful: missing file = empty state
        }
        return err
    }
    return json.Unmarshal(data, v)
}
```

A missing file is not an error — it is treated as an empty state. This is the correct
behavior for first run (no data files exist yet) and for after a `--reset`.

The caller passes a pre-initialized struct:

```go
f := forest.NewForest()
logLoadErr("intent", persist.Load(p.intentFile, f))
```

If the file doesn't exist, `f` retains its zero/default values. If the file exists,
`json.Unmarshal` populates `f` in place.

---

## Remove

```go
// persist.go:85-91
func Remove(path string) error {
    err := os.Remove(path)
    if err != nil && !errors.Is(err, os.ErrNotExist) {
        return err
    }
    return nil
}
```

Used by `--reset` to delete all four data files. `os.ErrNotExist` is silently ignored —
idempotent removal is the correct behavior for a reset command.

---

## File Encoding: Indented JSON

All state is stored as pretty-printed JSON. The four files and their top-level structure:

### `data/intent.json` → `forest.Forest`

```json
{
  "trees": [
    {
      "id": "m0x1a2b3c",
      "rootId": "m0x4d5e6f",
      "nodes": {
        "m0x4d5e6f": {
          "id": "m0x4d5e6f",
          "content": "authenticat | token | oauth | user",
          "depth": 0,
          "weight": 2.807,
          "frequency": 7,
          ...
        }
      },
      ...
    }
  ],
  "meta": { "totalPrompts": 42, ... }
}
```

### `data/engine.json` → `tfidf.Engine`

```json
{
  "docFreq": {
    "authenticat": 5,
    "token": 8,
    "oauth": 3,
    ...
  },
  "totalDocs": 42
}
```

### `data/guide.json` → `guide.Guide`

```json
{
  "entries": [
    {
      "summary": "Here's the OAuth token refresh endpoint implementation...",
      "intentId": "m0x4d5e6f",
      "timestamp": 1706000000000,
      "reinforced": true
    }
  ],
  "maxSize": 15
}
```

### `data/markov.json` → `markov.Chain`

```json
{
  "counts": {
    "tree-id-A": { "tree-id-B": 3, "tree-id-C": 1 }
  },
  "totals": { "tree-id-A": 4 },
  "lastTopic": "tree-id-B"
}
```

---

## Why JSON and Not a Database?

Several alternatives were considered:

- **SQLite**: Excellent for structured queries, but requires CGo on most Go setups
  (or a pure-Go driver with feature tradeoffs). Adds significant complexity.
- **BoltDB/bbolt**: Pure-Go key-value store, but yet another dependency.
- **Binary encoding (gob, protobuf)**: Not human-readable, harder to inspect/debug.
- **Plain JSON (no atomicity)**: Fast to implement, but data corruption on crash.

**Atomic JSON** is the perfect fit for this system:
- Single dependency: Go standard library only
- Human-readable: user can inspect and understand their data
- Debuggable: `cat data/intent.json | jq` works
- Safe: write-then-rename guarantees no corruption
- Fast enough: the forest is typically <10KB; JSON marshal/unmarshal is negligible

---

## Cross-Platform Notes

The project explicitly handles Windows file semantics:

```go
// persist.go:34-38
if runtime.GOOS == "windows" {
    _ = os.Remove(path)
}
```

This is the only Windows-specific code in the codebase. All other code uses
standard library functions that abstract platform differences. File paths use
`filepath.Join` throughout for OS-appropriate separators.

File permissions (`0755`, `0644`) are set for Unix semantics. On Windows, these values
are accepted by `os.MkdirAll` / `os.WriteFile` but the Windows permission model
(ACLs) is entirely different — the values are effectively advisory on Windows. This is
acceptable since Focus Gate is a local, single-user tool.
