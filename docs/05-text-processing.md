# Text Processing Pipeline

**Files:** [internal/text/tokenizer.go](../internal/text/tokenizer.go),
[internal/text/stemmer.go](../internal/text/stemmer.go)

Before any prompt can be vectorized or compared, it must pass through a normalization
pipeline that reduces noisy, variable surface forms into a consistent set of meaningful
tokens. This is where raw English text becomes the discrete vocabulary that the TF-IDF
engine operates on.

---

## The Pipeline

Every prompt flows through the same sequence:

```
raw prompt
    │
    ├── text.CleanPrompt()     Strip XML-style IDE/system tags
    │
    └── text.Tokenize()
            │
            ├── strings.ToLower()
            │
            ├── strings.FieldsFunc()   Split on non-alphanumeric
            │   (keeps hyphens and underscores inside tokens)
            │
            ├── Stem(t)               Two-pass stemmer per token
            │
            └── filter:
                ├── len(t) <= 1  → discard
                └── stopWords[t] → discard

    └── text.TermFrequency()   token count → normalized frequency
```

---

## CleanPrompt — IDE Tag Stripping

```go
// tokenizer.go:41-43
var tagPattern = regexp.MustCompile(`<[a-z_-]+>[\s\S]*?</[a-z_-]+>`)

func CleanPrompt(raw string) string {
    return strings.TrimSpace(tagPattern.ReplaceAllString(raw, ""))
}
```

Claude Code injects IDE context into prompts via XML-style tags, e.g.:

```
<file_content>
package main
...
</file_content>

Please refactor the auth handler.
```

If the file content were tokenized, it would massively inflate the prompt's token set
with code syntax, variable names, and structural keywords — completely drowning out
the user's actual intent ("refactor the auth handler").

`CleanPrompt` removes all `<tag>...</tag>` blocks before classification. It uses a
non-greedy `[\s\S]*?` match to avoid consuming content between two separate tags.

This is called at [cmd/focus/main.go:262](../cmd/focus/main.go#L262) before prompt
processing, and in [gate/dryrun.go](../internal/gate/dryrun.go) documentation notes that
callers should apply it before passing to `DryRun`.

---

## Tokenize — Split, Stem, Filter

```go
// tokenizer.go:46-71
func Tokenize(text string) []string {
    lower := strings.ToLower(text)

    // Split on boundaries, keeping hyphens and underscores within tokens.
    raw := strings.FieldsFunc(lower, func(r rune) bool {
        return !unicode.IsLetter(r) && !unicode.IsDigit(r) && r != '-' && r != '_'
    })

    var tokens []string
    for _, t := range raw {
        t = Stem(t)
        if len(t) > 1 && !stopWords[t] {
            tokens = append(tokens, t)
        }
    }
    return tokens
}
```

### Hyphen/Underscore Preservation

The split function keeps hyphens and underscores inside tokens. This prevents a specific
class of stemmer errors. Consider `"session-expiry"`:

- **Without** hyphen preservation: splits to `["session", "expiry"]`, then stemming
  produces `["ses", "expiry"]` — "session" incorrectly strips to `"ses"` via the `-sion` rule.
- **With** hyphen preservation: stays as `"session-expiry"` as a single token.

The comment at [tokenizer.go:53-56](../internal/text/tokenizer.go#L53) notes:
> This prevents compound-word fragments from false-stemming (e.g. "session-expiry" stays
> whole instead of "session" → "ses" via -sion).

Whether compound words or single words are better for classification is situation-
dependent, but keeping them whole at least avoids the stemmer's known weakness with
word fragments.

### Stop Words

The stop word list at [tokenizer.go:9-38](../internal/text/tokenizer.go#L9) contains
~100 terms covering:

- **Grammatical function words**: articles (a, an, the), conjunctions (and, or, but),
  prepositions (in, on, at, to, for, of, with, by)
- **Pronouns**: I, we, you, he, she, they, me, my, our, your, his, her
- **Auxiliary verbs**: is, be, was, are, been, have, has, had, do, does, did,
  will, would, could, should, may, might, can, shall, must
- **Common software-talk filler**: let, need, want, like, make, think, know, see,
  get, go, going, new, well, now, way, one, two, first

The list includes interesting entries that deserve note:

- `"use"` / `"used"` — "use this library", "used in production" would both be stripped
  to avoid polluting vectors with the ubiquitous "use" token.
- `"re"`, `"ve"`, `"ll"` — contracted forms after apostrophe-splitting (e.g. "we're"
  → "we" + "re"; "we've" → "we" + "ve").
- `"don"`, `"didn"`, `"doesn"`, `"won"`, `"isn"`, `"aren"`, `"wasn"`, `"weren"` —
  negation contractions after apostrophe split. Including negations as stop words is
  a deliberate choice: "don't add X" and "add X" both classify into the "add X" topic.

---

## Stem — Two-Pass Lightweight Stemmer

**File:** [internal/text/stemmer.go](../internal/text/stemmer.go)

The stemmer maps surface forms to a common root form. For example, "authenticating",
"authentication", and "authenticated" all reduce to `"authenticat"` via suffix stripping
(`-ing`, `-tion`, `-ed`). The base form "authenticate" has no matching suffix and remains
unchanged.

```go
// stemmer.go:38-64
func Stem(word string) string {
    if len(word) < 4 { return word }

    if override, ok := stemOverrides[word]; ok {
        return override
    }

    // Pass 1: plurals
    if len(word) > 4 && strings.HasSuffix(word, "ies") {
        word = word[:len(word)-3] + "y"
    } else if len(word) > 4 && strings.HasSuffix(word, "es") && word[len(word)-3] != 's' {
        word = word[:len(word)-2]
    } else if len(word) > 3 && word[len(word)-1] == 's' && word[len(word)-2] != 's' {
        word = word[:len(word)-1]
    }

    // Pass 2: derivational suffix (longest match)
    for _, suf := range derivational {
        if len(word) > len(suf)+2 && strings.HasSuffix(word, suf) {
            return word[:len(word)-len(suf)]
        }
    }
    return word
}
```

### Pass 1: Plurals

Three patterns, applied in order:

| Pattern | Example | Result |
|---------|---------|--------|
| `-ies` (len > 4) | `"libraries"` | `"library"` |
| `-es` where `word[-3] != 's'` | `"fixes"` | `"fix"` |
| `-s` where `word[-2] != 's'` | `"tokens"` | `"token"` |

The double-s guard (`word[len(word)-2] != 's'`) prevents "less" from becoming "le" or
"class" from becoming "clas". The length guard (`len(word) > 4`) prevents very short
words like "goes" from being incorrectly stripped.

### Pass 2: Derivational Suffixes

```go
// stemmer.go:8-13
var derivational = []string{
    "ization", "ising", "izing", "ional",
    "ment", "ness", "less",
    "able", "ible", "tion", "sion", "ling", "ally",
    "ful", "ous", "ive", "ing", "ed", "ly",
}
```

These are sorted **longest-first**. The loop uses the first matching suffix:

```go
for _, suf := range derivational {
    if len(word) > len(suf)+2 && strings.HasSuffix(word, suf) {
        return word[:len(word)-len(suf)]
    }
}
```

Longest-first matching prevents partial stripping. For example:
- `"containerization"` should strip `"ization"` (→ `"container"`)
- If `"tion"` were tried first, it would strip only `"tion"` and leave `"containeriza"` — wrong

The `len(word) > len(suf)+2` guard ensures we don't strip a suffix from a word shorter
than the suffix + 2 characters, preventing edge cases like stripping `"ing"` from `"ring"`.

### Why `-er` Is Excluded

The comment at [stemmer.go:6-7](../internal/text/stemmer.go#L6) explains:
> "er" is intentionally excluded — too many English root words end in "er"
> (container, server, computer, docker) causing false conflation.

If `-er` were included:
- `"container"` → `"contain"` (wrong — "container" is a distinct concept)
- `"server"` → `"serv"` (wrong — "server" is a distinct concept)
- `"docker"` → `"dock"` (very wrong — completely different domain)

### Override Map

```go
// stemmer.go:18-30
var stemOverrides = map[string]string{
    "organization":   "organiz",
    "organizations":  "organiz",
    "organize":       "organiz",
    // ...
    "authorization":  "authoriz",
    "authorizations": "authoriz",
    "authorize":      "authoriz",
    "authorized":     "authoriz",
    "authorizing":    "authoriz",
    "unauthorized":   "authoriz",
}
```

Both `"authorization"` and `"organization"` end in `"ization"`, so the mechanical
stemmer's first matching suffix is `-ization`:

- `"authorization"` (13 chars) → strip `-ization` → `"author"` — false conflation.
  Author (a writer) is an unrelated concept.
- `"organization"` (12 chars) → strip `-ization` → `"organ"` — false conflation.
  Organ (a body part or instrument) is an unrelated concept.

The overrides short-circuit the mechanical rules and map both families to a consistent
stem (`"authoriz"`, `"organiz"`) that groups related variants correctly without
conflating them with unrelated words.

### Stemming Examples in Context

| Surface Form | Stem Result | How |
|-------------|-------------|-----|
| `"authentication"` | `"authenticat"` | Pass 2: suffix `-tion` → strip → `"authenticat"` |
| `"authenticate"` | `"authenticate"` | No suffix in derivational list matches → unchanged |
| `"authenticating"` | `"authenticat"` | suffix `-ing` → strip → `"authenticat"` |
| `"authenticated"` | `"authenticat"` | suffix `-ed` → strip → `"authenticat"` |
| `"containers"` | `"container"` | Pass 1: `-s` → `"container"` |
| `"containerization"` | `"container"` | Pass 2: `-ization` → `"container"` |
| `"fixing"` | `"fix"` | Pass 2: `-ing` → `"fix"` |
| `"fixes"` | `"fix"` | Pass 1: `-es` → `"fix"` |

The stemmer successfully conflates related forms while correctly refusing to conflate
unrelated words like "container" (the `-er` omission) and "author"/"authorization"
(the override map).

---

## TermFrequency

```go
// tokenizer.go:79-92
func TermFrequency(tokens []string) map[string]float64 {
    tf := make(map[string]float64, len(tokens))
    for _, t := range tokens {
        tf[t]++
    }
    n := float64(len(tokens))
    for k := range tf { tf[k] /= n }
    return tf
}
```

Produces normalized TF: each term's count divided by the total token count.
Used by both `engine.Vectorize` and `engine.VectorizeTokens`.

---

## Design Philosophy

The text processing pipeline follows a principle of **sufficient but not over-engineered
normalization**:

- **Stop words are aggressive**: Common software-vocabulary filler ("use", "let", "need",
  "get") is removed, keeping only domain-meaningful terms.
- **Stemmer is conservative**: Only unambiguous suffixes are stripped; `-er` is excluded
  to avoid false conflation of technical terms.
- **No lemmatization**: Full lemmatization (e.g. Porter Stemmer, Snowball) would require
  either a dictionary or complex rules. The two-pass stemmer captures 80% of the benefit
  with 5% of the complexity.
- **No semantic embeddings**: Using pre-trained word embeddings would require bundling a
  model file (100MB+) or making network calls. The TF-IDF approach is fully local and
  zero-dependency.
