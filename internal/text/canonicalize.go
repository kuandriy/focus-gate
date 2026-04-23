package text

// CanonicalizeOpts controls typo-tolerant token canonicalization. Zero value
// (all fields zero) is equivalent to Enabled=false — the canonicalizer
// becomes a pass-through and Tokenize behaviour is unchanged.
type CanonicalizeOpts struct {
	Enabled          bool
	MaxDistance      int // max Levenshtein edits allowed between typo and canonical term
	MinWordLen       int // skip canonicalization for tokens shorter than this (after stem)
	MinEstablishedDF int // only rewrite toward terms with at least this many occurrences
}

// TokenizeWithCorpus runs the standard tokenization pipeline and then, when
// opts.Enabled is true, canonicalizes each new token against the given corpus
// of document-frequency counts. Novel tokens that are within opts.MaxDistance
// edits of an already-established term (DF ≥ opts.MinEstablishedDF) are
// rewritten to that established term. This keeps repeated typos of the same
// word from each becoming a distinct TF-IDF entry and splintering the forest.
//
// When opts.Enabled is false, corpus is nil, or corpus is empty, the function
// returns exactly the same result as Tokenize.
func TokenizeWithCorpus(text string, corpus map[string]int, opts CanonicalizeOpts) []string {
	tokens := Tokenize(text)
	if !opts.Enabled || len(corpus) == 0 || len(tokens) == 0 {
		return tokens
	}
	// Sensible floors so a half-populated opts struct never produces wild
	// behaviour: at minimum require 5-char words and one established
	// neighbour. Callers supply full defaults in gate.toGateConfig.
	if opts.MaxDistance <= 0 {
		return tokens
	}
	if opts.MinWordLen < 2 {
		opts.MinWordLen = 2
	}
	if opts.MinEstablishedDF < 1 {
		opts.MinEstablishedDF = 1
	}

	out := make([]string, len(tokens))
	for i, t := range tokens {
		out[i] = canonicalize(t, corpus, opts)
	}
	return out
}

// canonicalize picks the best-matching established term for a single token.
// Cheap guards first: short tokens and already-known tokens pass through
// unchanged. Only truly novel-and-long tokens pay the edit-distance scan.
//
// Ties break deterministically by alphabetical order so the same session
// replayed against the same corpus always produces the same canonical form.
func canonicalize(token string, corpus map[string]int, opts CanonicalizeOpts) string {
	if len([]rune(token)) < opts.MinWordLen {
		return token
	}
	if corpus[token] > 0 {
		return token // already a known term
	}

	best := ""
	bestDist := opts.MaxDistance + 1
	for term, df := range corpus {
		if df < opts.MinEstablishedDF {
			continue
		}
		if term == token {
			return term
		}
		d := LevenshteinCutoff(token, term, opts.MaxDistance)
		if d > opts.MaxDistance {
			continue
		}
		if d < bestDist || (d == bestDist && term < best) {
			bestDist = d
			best = term
		}
	}
	if best == "" {
		return token
	}
	return best
}
