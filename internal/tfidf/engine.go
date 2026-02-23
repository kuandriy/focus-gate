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

// Engine is an incremental TF-IDF engine. Unlike rebuilding the entire corpus
// on every invocation, it persists document frequency counts and updates them
// incrementally as documents are added or removed (during pruning).
type Engine struct {
	DocFreq   map[string]int `json:"docFreq"`
	TotalDocs int            `json:"totalDocs"`
}

// NewEngine creates an empty TF-IDF engine.
func NewEngine() *Engine {
	return &Engine{
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
	if len(tokens) == 0 {
		return nil
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

// VectorizeTokens converts pre-tokenized text into a sorted TF-IDF Vector.
func (e *Engine) VectorizeTokens(tokens []string) Vector {
	if len(tokens) == 0 {
		return nil
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
