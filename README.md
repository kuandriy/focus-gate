# Focus Gate

Prompt intent tracking for AI coding assistants. Zero dependencies. Single binary.

Focus Gate intercepts every prompt you send to your AI coding assistant, compares it against your accumulated conversation intent using TF-IDF cosine similarity, and injects a compact context summary back into the conversation. The AI always knows what you've been working on, even across long sessions.

## Install

Download the binary for your platform from [Releases](https://github.com/kuandriy/focus-gate/releases), or build from source:

```bash
go build -o focus-gate ./cmd/focus
```

## Usage

### As a Claude Code Hook

Add to `.claude/settings.local.json`:

```json
{
  "hooks": {
    "UserPromptSubmit": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "/path/to/focus-gate"
          }
        ]
      }
    ]
  }
}
```

### CLI

```bash
# Show current forest state
./focus-gate --status

# Reset all tracking data
./focus-gate --reset

# Process a prompt (hook mode, reads JSON from stdin)
echo '{"prompt":"your prompt text"}' | ./focus-gate
```

## How It Works

Your prompts are organized into a **forest** of topic trees. Each new prompt is classified by TF-IDF cosine similarity:

| Score | Action | Meaning |
|:---:|:---:|:---|
| >= 0.55 | Extend | Closely related to an existing topic |
| 0.25 - 0.55 | Branch | Broadly related to a topic theme |
| < 0.25 | New Tree | Unrelated — starts a new topic |

The forest self-cleans: nodes decay over time, and when memory fills up, the least relevant topics are pruned automatically.

## Configuration

Create a `config.json` alongside the binary:

```json
{
  "memorySize": 100,
  "decayRate": 0.05,
  "similarity": { "extend": 0.55, "branch": 0.25 },
  "contextLimit": 600,
  "bubbleUpTerms": 6
}
```

## Architecture

```
cmd/focus/          Entry point (CLI, stdin/stdout)
internal/
  text/             Tokenizer, stemmer, stop words
  tfidf/            TF-IDF engine, sparse vectors, cosine similarity
  forest/           Node, Tree, Forest, heap-based pruning
  gate/             FocusGate classifier (classify, apply, bubble-up)
  guide/            AI response tracking (ring buffer)
  persist/          Atomic JSON persistence
```

Zero external dependencies. Built entirely on Go's standard library.

## License

MIT
