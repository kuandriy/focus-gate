package tfidf

import (
	"math"
	"sort"
)

// Term is a single term-weight pair in a sparse vector.
type Term struct {
	Word   string
	Weight float64
}

// Vector is a sparse TF-IDF vector, always sorted by Word for merge-join operations.
type Vector []Term

// NewVector creates a sorted Vector from a term-weight map.
func NewVector(weights map[string]float64) Vector {
	if len(weights) == 0 {
		return nil
	}
	v := make(Vector, 0, len(weights))
	for word, w := range weights {
		v = append(v, Term{Word: word, Weight: w})
	}
	sort.Slice(v, func(i, j int) bool {
		return v[i].Word < v[j].Word
	})
	return v
}

// CosineSimilarity computes the cosine of the angle between two sorted sparse vectors
// using a merge-join. Zero allocations, O(n+m) time.
//
// Returns 0.0 if either vector is empty (undefined angle).
// Returns 1.0 for identical vectors.
func CosineSimilarity(a, b Vector) float64 {
	if len(a) == 0 || len(b) == 0 {
		return 0
	}

	var dot, normA, normB float64
	i, j := 0, 0

	for i < len(a) && j < len(b) {
		if a[i].Word == b[j].Word {
			dot += a[i].Weight * b[j].Weight
			normA += a[i].Weight * a[i].Weight
			normB += b[j].Weight * b[j].Weight
			i++
			j++
		} else if a[i].Word < b[j].Word {
			normA += a[i].Weight * a[i].Weight
			i++
		} else {
			normB += b[j].Weight * b[j].Weight
			j++
		}
	}

	// Drain remaining elements
	for ; i < len(a); i++ {
		normA += a[i].Weight * a[i].Weight
	}
	for ; j < len(b); j++ {
		normB += b[j].Weight * b[j].Weight
	}

	denom := math.Sqrt(normA) * math.Sqrt(normB)
	if denom == 0 {
		return 0
	}
	return dot / denom
}
