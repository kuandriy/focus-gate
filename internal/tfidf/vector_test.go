package tfidf

import (
	"math"
	"testing"
)

func TestNewVector(t *testing.T) {
	v := NewVector(map[string]float64{
		"beta": 2.0, "alpha": 1.0, "gamma": 3.0,
	})

	if len(v) != 3 {
		t.Fatalf("len = %d, want 3", len(v))
	}
	// Must be sorted by Word
	if v[0].Word != "alpha" || v[1].Word != "beta" || v[2].Word != "gamma" {
		t.Errorf("not sorted: %v", v)
	}
}

func TestNewVectorEmpty(t *testing.T) {
	v := NewVector(nil)
	if v != nil {
		t.Errorf("NewVector(nil) = %v, want nil", v)
	}
	v = NewVector(map[string]float64{})
	if v != nil {
		t.Errorf("NewVector(empty) = %v, want nil", v)
	}
}

func TestCosineSimilarityIdentical(t *testing.T) {
	v := NewVector(map[string]float64{"auth": 1.0, "token": 2.0})
	sim := CosineSimilarity(v, v)
	if math.Abs(sim-1.0) > 1e-10 {
		t.Errorf("identical vectors: similarity = %f, want 1.0", sim)
	}
}

func TestCosineSimilarityOrthogonal(t *testing.T) {
	a := NewVector(map[string]float64{"auth": 1.0, "token": 1.0})
	b := NewVector(map[string]float64{"database": 1.0, "schema": 1.0})
	sim := CosineSimilarity(a, b)
	if sim != 0.0 {
		t.Errorf("orthogonal vectors: similarity = %f, want 0.0", sim)
	}
}

func TestCosineSimilarityPartial(t *testing.T) {
	a := NewVector(map[string]float64{"auth": 1.0, "token": 1.0})
	b := NewVector(map[string]float64{"auth": 1.0, "session": 1.0})

	sim := CosineSimilarity(a, b)

	// Shared: auth(1*1)=1. normA = 1+1=2. normB = 1+1=2.
	// cos = 1 / (sqrt(2) * sqrt(2)) = 1/2 = 0.5
	if math.Abs(sim-0.5) > 1e-10 {
		t.Errorf("partial overlap: similarity = %f, want 0.5", sim)
	}
}

func TestCosineSimilarityEmpty(t *testing.T) {
	a := NewVector(map[string]float64{"auth": 1.0})
	if CosineSimilarity(a, nil) != 0.0 {
		t.Error("similarity with nil should be 0")
	}
	if CosineSimilarity(nil, a) != 0.0 {
		t.Error("nil similarity should be 0")
	}
	if CosineSimilarity(nil, nil) != 0.0 {
		t.Error("nil/nil similarity should be 0")
	}
}

func TestCosineSimilarityKnownValue(t *testing.T) {
	// Manual calculation: a=[3,4,0], b=[0,4,3] in dimensions alpha,beta,gamma
	a := NewVector(map[string]float64{"alpha": 3.0, "beta": 4.0})
	b := NewVector(map[string]float64{"beta": 4.0, "gamma": 3.0})

	// dot = 4*4 = 16
	// normA = sqrt(9+16) = 5
	// normB = sqrt(16+9) = 5
	// cos = 16/25 = 0.64
	sim := CosineSimilarity(a, b)
	if math.Abs(sim-0.64) > 1e-10 {
		t.Errorf("known value: similarity = %f, want 0.64", sim)
	}
}
