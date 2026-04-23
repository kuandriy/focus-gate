package memory

import (
	"crypto/sha256"
	"encoding/hex"
	"sort"

	"github.com/kuandriy/focus-gate/internal/tfidf"
)

// VocabSnapshot is a thin adapter decoupling the memory package from the
// concrete TF-IDF engine. Callers build one per operation and pass it to
// WriteFile / RefreshDerived / Surface. The memory package never sees
// the engine struct directly, which keeps tests trivial — a test can
// construct a VocabSnapshot with any closure and custom hash.
type VocabSnapshot struct {
	Hash      string                               // identifies the vocabulary state
	Vectorize func(text string) map[string]float64 // returns term→weight for the text
}

// NewVocabSnapshot builds a VocabSnapshot from a tfidf.Engine. The hash
// is computed once at snapshot time over the engine's sorted DocFreq
// keys — enough to detect vocabulary drift between a memory's write time
// and a later load time. The closure re-uses Engine.Vectorize under the
// hood so memory lookup matches classifier behaviour exactly.
func NewVocabSnapshot(e *tfidf.Engine) VocabSnapshot {
	return VocabSnapshot{
		Hash: hashVocab(e),
		Vectorize: func(text string) map[string]float64 {
			v := e.Vectorize(text)
			out := make(map[string]float64, len(v))
			for _, t := range v {
				out[t.Word] = t.Weight
			}
			return out
		},
	}
}

// hashVocab produces a short stable fingerprint over the engine's current
// vocabulary. Only keys are hashed, not counts — DF rising and falling
// doesn't matter for "is this vector still addressable?", only whether
// new terms exist that weren't there before.
func hashVocab(e *tfidf.Engine) string {
	terms := make([]string, 0, len(e.DocFreq))
	for t := range e.DocFreq {
		terms = append(terms, t)
	}
	sort.Strings(terms)
	h := sha256.New()
	for _, t := range terms {
		h.Write([]byte(t))
		h.Write([]byte{0})
	}
	sum := h.Sum(nil)
	return hex.EncodeToString(sum[:8]) // 16 hex chars is plenty of signal
}
