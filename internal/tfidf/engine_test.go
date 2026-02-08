package tfidf

import (
	"math"
	"testing"
)

func TestEngineAddDocument(t *testing.T) {
	e := NewEngine()
	e.AddDocument([]string{"auth", "token", "jwt"})
	e.AddDocument([]string{"auth", "session", "login"})
	e.AddDocument([]string{"database", "schema", "migration"})

	if e.TotalDocs != 3 {
		t.Errorf("TotalDocs = %d, want 3", e.TotalDocs)
	}
	if e.DocFreq["auth"] != 2 {
		t.Errorf("DocFreq[auth] = %d, want 2", e.DocFreq["auth"])
	}
	if e.DocFreq["token"] != 1 {
		t.Errorf("DocFreq[token] = %d, want 1", e.DocFreq["token"])
	}
	if e.DocFreq["database"] != 1 {
		t.Errorf("DocFreq[database] = %d, want 1", e.DocFreq["database"])
	}
}

func TestEngineAddDocumentDeduplicates(t *testing.T) {
	e := NewEngine()
	// Same token repeated in one document should only count once
	e.AddDocument([]string{"auth", "auth", "auth"})

	if e.TotalDocs != 1 {
		t.Errorf("TotalDocs = %d, want 1", e.TotalDocs)
	}
	if e.DocFreq["auth"] != 1 {
		t.Errorf("DocFreq[auth] = %d, want 1 (deduplicated)", e.DocFreq["auth"])
	}
}

func TestEngineRemoveDocument(t *testing.T) {
	e := NewEngine()
	e.AddDocument([]string{"auth", "token"})
	e.AddDocument([]string{"auth", "session"})

	e.RemoveDocument([]string{"auth", "token"})

	if e.TotalDocs != 1 {
		t.Errorf("TotalDocs = %d, want 1", e.TotalDocs)
	}
	if e.DocFreq["auth"] != 1 {
		t.Errorf("DocFreq[auth] = %d, want 1", e.DocFreq["auth"])
	}
	// "token" should be deleted (DF reached 0)
	if _, exists := e.DocFreq["token"]; exists {
		t.Error("DocFreq[token] should be deleted after removal")
	}
}

func TestEngineRemoveDocumentFloor(t *testing.T) {
	e := NewEngine()
	// Removing from empty shouldn't go negative
	e.RemoveDocument([]string{"ghost"})
	if e.TotalDocs != 0 {
		t.Errorf("TotalDocs = %d, want 0 (floor)", e.TotalDocs)
	}
}

func TestEngineIDF(t *testing.T) {
	e := NewEngine()
	e.AddDocument([]string{"auth", "token"})
	e.AddDocument([]string{"auth", "session"})
	e.AddDocument([]string{"database", "schema"})

	// "auth" appears in 2/3 docs: log2(1 + 3/2) = log2(2.5) ≈ 1.322
	authIDF := e.IDF("auth")
	expected := math.Log2(1 + 3.0/2.0)
	if math.Abs(authIDF-expected) > 1e-10 {
		t.Errorf("IDF(auth) = %f, want %f", authIDF, expected)
	}

	// "token" appears in 1/3 docs: log2(1 + 3/1) = log2(4) = 2.0
	tokenIDF := e.IDF("token")
	if math.Abs(tokenIDF-2.0) > 1e-10 {
		t.Errorf("IDF(token) = %f, want 2.0", tokenIDF)
	}

	// Unknown term
	if e.IDF("unknown") != 0 {
		t.Error("IDF of unknown term should be 0")
	}
}

func TestEngineVectorize(t *testing.T) {
	e := NewEngine()
	e.AddDocument([]string{"auth", "token", "jwt"})
	e.AddDocument([]string{"auth", "session"})
	e.AddDocument([]string{"database", "schema"})

	v := e.Vectorize("add JWT authentication")
	if v == nil {
		t.Fatal("Vectorize returned nil")
	}

	// Should have non-zero weights for terms that exist in the corpus
	hasWeight := false
	for _, term := range v {
		if term.Weight > 0 {
			hasWeight = true
			break
		}
	}
	if !hasWeight {
		t.Error("Vector should have at least one non-zero weight")
	}
}

func TestEngineVectorizeEmpty(t *testing.T) {
	e := NewEngine()
	v := e.Vectorize("")
	if v != nil {
		t.Errorf("Vectorize empty should be nil, got %v", v)
	}
}

func TestEngineVectorizeRareTermHigher(t *testing.T) {
	e := NewEngine()
	e.AddDocument([]string{"auth", "token"})
	e.AddDocument([]string{"auth", "session"})
	e.AddDocument([]string{"auth", "database"})

	// "auth" is in all 3 docs (common), "token" is in 1 doc (rare)
	// For the text "auth token", "token" should have higher TF-IDF weight
	v := e.VectorizeTokens([]string{"auth", "token"})

	var authWeight, tokenWeight float64
	for _, term := range v {
		switch term.Word {
		case "auth":
			authWeight = term.Weight
		case "token":
			tokenWeight = term.Weight
		}
	}

	if tokenWeight <= authWeight {
		t.Errorf("rare term 'token' (%f) should have higher weight than common term 'auth' (%f)",
			tokenWeight, authWeight)
	}
}
