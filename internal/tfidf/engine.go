package tfidf

import (
	"math"

	"github.com/kuandriy/focus-gate/internal/text"
)

// minVirtualDocs is the minimum effective corpus size used in the IDF
// denominator. With fewer than this many real documents, IDF pretends the
// corpus is this large so that term weights can discriminate even during
// the first few prompts of a session. Once TotalDocs >= minVirtualDocs,
// this floor has no effect.
const minVirtualDocs = 5

// SchemaVersion is the on-disk schema version for engine.json.
const SchemaVersion = "1"

// Engine is an incremental TF-IDF engine. Unlike rebuilding the entire corpus
// on every invocation, it persists document frequency counts and updates them
// incrementally as documents are added or removed (during pruning).
type Engine struct {
	Schema    string         `json:"schemaVersion"`
	DocFreq   map[string]int `json:"docFreq"`
	TotalDocs int            `json:"totalDocs"`

	// Sublinear toggles `1 + log2(count)` term-frequency weighting in
	// place of the linear `count / total` form. Standard IR practice
	// for dampening repeated-term dominance — a word appearing 100x
	// shouldn't outweigh one appearing 10x by a factor of 10. Off by
	// default to preserve existing classification behaviour;
	// runtime-only (not persisted) so callers wire it from config
	// after Load.
	Sublinear bool `json:"-"`
}

// SetSchemaVersion implements persist.SchemaVersioner.
func (e *Engine) SetSchemaVersion(v string) { e.Schema = v }

// NewEngine creates an empty TF-IDF engine.
func NewEngine() *Engine {
	return &Engine{
		Schema:  SchemaVersion,
		DocFreq: make(map[string]int),
	}
}

// AddDocument updates document frequency counts for a new document's tokens.
// Each unique token increments its DF by 1.
func (e *Engine) AddDocument(tokens []string) {
	seen := make(map[string]bool, len(tokens))
	for _, t := range tokens {
		if !seen[t] {
			e.DocFreq[t]++
			seen[t] = true
		}
	}
	e.TotalDocs++
}

// RemoveDocument decrements document frequency counts when a document is pruned.
// Tokens that reach zero DF are deleted from the map to prevent unbounded growth.
func (e *Engine) RemoveDocument(tokens []string) {
	seen := make(map[string]bool, len(tokens))
	for _, t := range tokens {
		if !seen[t] {
			e.DocFreq[t]--
			if e.DocFreq[t] <= 0 {
				delete(e.DocFreq, t)
			}
			seen[t] = true
		}
	}
	e.TotalDocs--
	if e.TotalDocs < 0 {
		e.TotalDocs = 0
	}
}

// IDF computes the inverse document frequency for a term.
// Uses smoothed formula: log2(1 + effectiveDocs/df) where effectiveDocs
// is max(TotalDocs, MinVirtualDocs) to ensure discrimination at low
// document counts. Returns 0 for unknown terms.
func (e *Engine) IDF(term string) float64 {
	df := e.DocFreq[term]
	if df == 0 {
		return 0
	}
	effectiveDocs := e.TotalDocs
	if effectiveDocs < minVirtualDocs {
		effectiveDocs = minVirtualDocs
	}
	return math.Log2(1 + float64(effectiveDocs)/float64(df))
}

// Vectorize converts raw text into a sorted TF-IDF Vector.
// Tokenizes the text, computes term frequencies, multiplies by IDF weights,
// and returns a sorted sparse vector ready for cosine similarity.
func (e *Engine) Vectorize(rawText string) Vector {
	tokens := text.Tokenize(rawText)
	return e.VectorizeTokens(tokens)
}

// VectorizeTokens converts pre-tokenized text into a sorted TF-IDF Vector.
// Uses sublinear `1 + log2(count)` weighting when Engine.Sublinear is set,
// linear `count / total` otherwise.
func (e *Engine) VectorizeTokens(tokens []string) Vector {
	if len(tokens) == 0 {
		return nil
	}
	if e.Sublinear {
		return e.vectorizeSublinear(tokens)
	}
	tf := text.TermFrequency(tokens)
	weights := make(map[string]float64, len(tf))
	for term, freq := range tf {
		idf := e.IDF(term)
		if idf > 0 {
			weights[term] = freq * idf
		}
	}
	return NewVector(weights)
}

// vectorizeSublinear builds a vector with sublinear TF weighting:
// `tf = 1 + log2(count)`. Cosine cares only about direction, so the
// missing /total normalization the linear form does is harmless.
func (e *Engine) vectorizeSublinear(tokens []string) Vector {
	counts := make(map[string]int, len(tokens))
	for _, t := range tokens {
		counts[t]++
	}
	weights := make(map[string]float64, len(counts))
	for term, count := range counts {
		idf := e.IDF(term)
		if idf <= 0 {
			continue
		}
		weights[term] = (1.0 + math.Log2(float64(count))) * idf
	}
	return NewVector(weights)
}
