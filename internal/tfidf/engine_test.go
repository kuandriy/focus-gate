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

	// TotalDocs=3, but effective is max(3, MinVirtualDocs=5)
	// "auth" appears in 2/3 docs: log2(1 + 5/2) = log2(3.5) ≈ 1.807
	authIDF := e.IDF("auth")
	expected := math.Log2(1 + 5.0/2.0)
	if math.Abs(authIDF-expected) > 1e-10 {
		t.Errorf("IDF(auth) = %f, want %f", authIDF, expected)
	}

	// "token" appears in 1/3 docs: log2(1 + 5/1) = log2(6) ≈ 2.585
	tokenIDF := e.IDF("token")
	expectedToken := math.Log2(1 + 5.0/1.0)
	if math.Abs(tokenIDF-expectedToken) > 1e-10 {
		t.Errorf("IDF(token) = %f, want %f", tokenIDF, expectedToken)
	}

	// Unknown term
	if e.IDF("unknown") != 0 {
		t.Error("IDF of unknown term should be 0")
	}
}

func TestEngineIDFFloorDisappearsAtScale(t *testing.T) {
	e := NewEngine()
	for i := 0; i < 10; i++ {
		e.AddDocument([]string{"common"})
	}
	e.AddDocument([]string{"rare"})

	// TotalDocs=11, above MinVirtualDocs=5 — floor has no effect
	// "rare" in 1/11: log2(1 + 11/1) = log2(12)
	rareIDF := e.IDF("rare")
	expected := math.Log2(1 + 11.0/1.0)
	if math.Abs(rareIDF-expected) > 1e-10 {
		t.Errorf("IDF(rare) at scale = %f, want %f", rareIDF, expected)
	}
}

func TestColdStartDiscrimination(t *testing.T) {
	e := NewEngine()
	// Single document about auth
	e.AddDocument([]string{"auth", "jwt", "token"})

	// With MinVirtualDocs floor, terms in 1 doc should get IDF > 1.0
	// (without the floor, all terms would get log2(1+1/1) = 1.0)
	idf := e.IDF("auth")
	if idf <= 1.0 {
		t.Errorf("cold-start IDF = %f, want > 1.0 (floor should boost discrimination)", idf)
	}

	// Two different prompts should produce low cosine similarity even at 1 doc
	authVec := e.VectorizeTokens([]string{"auth", "jwt", "token"})
	dbVec := e.VectorizeTokens([]string{"databas", "migrat", "schema"})
	sim := CosineSimilarity(authVec, dbVec)
	if sim > 0.01 {
		t.Errorf("dissimilar prompts at cold start: cosine = %f, want ~ 0.0", sim)
	}
}

// SublinearTF must dampen the effect of repeated terms. With linear TF,
// repetition is "free" once the formula normalizes by total tokens; with
// sublinear, repetition contributes to signal but proportionally less than
// linearly (1 + log2(count)).
func TestEngineVectorize_SublinearDampensRepetition(t *testing.T) {
	e := NewEngine()
	e.AddDocument([]string{"auth", "jwt"})
	e.AddDocument([]string{"unrelated", "stuff"})

	authW := func(v Vector) float64 {
		for _, t := range v {
			if t.Word == "auth" {
				return t.Weight
			}
		}
		return 0
	}

	e.Sublinear = false
	wLinearMany := authW(e.VectorizeTokens([]string{"auth", "auth", "auth", "auth"}))

	e.Sublinear = true
	wSublinOne := authW(e.VectorizeTokens([]string{"auth"}))
	wSublinMany := authW(e.VectorizeTokens([]string{"auth", "auth", "auth", "auth"}))

	if wSublinOne == 0 {
		t.Fatal("sublinear auth weight zero for single-token vector")
	}
	if wSublinMany <= wLinearMany {
		t.Errorf("sublinear should weight repetition higher than linear normalized TF: linear=%.3f sublin=%.3f",
			wLinearMany, wSublinMany)
	}
	if wSublinMany <= wSublinOne {
		t.Errorf("sublinear-many (%.3f) should exceed sublinear-one (%.3f)", wSublinMany, wSublinOne)
	}
	if wSublinMany > 4*wSublinOne {
		t.Errorf("ratio many/one = %.2f exceeds 4× — not sublinear", wSublinMany/wSublinOne)
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
