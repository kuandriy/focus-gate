package memory

import (
	"strings"
	"testing"

	"github.com/kuandriy/focus-gate/internal/tfidf"
)

// vec is a tiny helper that builds a tfidf.Vector from a term→weight map.
func vec(weights map[string]float64) tfidf.Vector {
	return tfidf.NewVector(weights)
}

// surfaceFixture wraps a one-source MultiSourceIndex with the supplied
// entries, ready for Surface tests.
func surfaceFixture(entries ...IndexEntry) (*MultiSourceIndex, VocabSnapshot) {
	mf := NewManifest()
	for _, e := range entries {
		if e.Source == "" {
			e.Source = mf.Source
		}
		mf.Entries = append(mf.Entries, e)
	}
	mf.RebuildInvertedIndexes()
	return NewMultiSourceIndex(mf), fakeVocab("vh1")
}

func TestSurface_DisabledOrEmpty(t *testing.T) {
	msi, vocab := surfaceFixture(
		IndexEntry{ID: "mem_x", Title: "x", Fingerprint: map[string]float64{"a": 1.0}},
	)
	cfg := DefaultSurfaceConfig()

	// Disabled.
	cfg.Enabled = false
	r := Surface(SurfaceInputs{
		PromptVec: vec(map[string]float64{"a": 1.0}),
		Vocab:     vocab,
		Index:     msi,
	}, cfg)
	if r.Block != "" || len(r.Selected) != 0 {
		t.Error("disabled config should return empty SurfaceResult")
	}

	// Empty index.
	cfg = DefaultSurfaceConfig()
	r = Surface(SurfaceInputs{
		PromptVec: vec(map[string]float64{"a": 1.0}),
		Vocab:     vocab,
		Index:     NewMultiSourceIndex(),
	}, cfg)
	if r.Block != "" || len(r.Selected) != 0 {
		t.Error("empty index should return empty SurfaceResult")
	}

	// Empty prompt vector and no asset hits — no signal.
	r = Surface(SurfaceInputs{
		PromptText: "",
		PromptVec:  nil,
		Vocab:      vocab,
		Index:      msi,
	}, cfg)
	if r.Block != "" || len(r.Selected) != 0 {
		t.Error("nil prompt vector with no assets should return empty SurfaceResult")
	}
}

func TestSurface_AssetHitDominatesFingerprint(t *testing.T) {
	// Memory A has the asset hit. Memory B has high fingerprint similarity
	// only. Asset (weight 1.0) should beat fingerprint (weight 0.4) at
	// equal raw scores.
	msi, vocab := surfaceFixture(
		IndexEntry{
			ID:          "mem_asset",
			Title:       "Auth",
			Path:        "mem_asset.md",
			Assets:      []string{"cmd/api/auth.go"},
			Fingerprint: map[string]float64{"foo": 0.1},
		},
		IndexEntry{
			ID:          "mem_fp",
			Title:       "Other",
			Path:        "mem_fp.md",
			Fingerprint: map[string]float64{"jwt": 1.0},
		},
	)
	// Need to RebuildInvertedIndexes after fixture mutation; surfaceFixture
	// already did that, but the Asset case requires the prompt to mention
	// it.
	r := Surface(SurfaceInputs{
		PromptText: "looking at cmd/api/auth.go for jwt",
		PromptVec:  vec(map[string]float64{"jwt": 1.0}),
		Vocab:      vocab,
		Index:      msi,
	}, DefaultSurfaceConfig())
	if len(r.Selected) == 0 {
		t.Fatal("expected at least one selected entry")
	}
	if r.Selected[0].Entry.ID != "mem_asset" {
		t.Errorf("asset hit should rank first, got %s", r.Selected[0].Entry.ID)
	}
}

func TestSurface_TopicWeightedScore(t *testing.T) {
	// Topic match: weight × cosine. Two memories with the same topic
	// keyword in the prompt; the one with topic weight 1.0 should beat
	// the one with weight 0.5.
	msi, vocab := surfaceFixture(
		IndexEntry{
			ID:    "mem_high",
			Title: "high",
			Path:  "mem_high.md",
			Topics: []WeightedEntry{
				{Name: "JWT authentication", Weight: 1.0},
			},
			Fingerprint: map[string]float64{"jwt": 0.1},
		},
		IndexEntry{
			ID:    "mem_low",
			Title: "low",
			Path:  "mem_low.md",
			Topics: []WeightedEntry{
				{Name: "JWT authentication", Weight: 0.5},
			},
			Fingerprint: map[string]float64{"jwt": 0.1},
		},
	)
	cfg := DefaultSurfaceConfig()
	cfg.Threshold = 0.0
	cfg.TopK = 2
	r := Surface(SurfaceInputs{
		PromptText: "jwt authentication",
		PromptVec:  vec(map[string]float64{"jwt": 1.0, "authentication": 1.0}),
		Vocab:      vocab,
		Index:      msi,
	}, cfg)
	if len(r.Selected) != 2 {
		t.Fatalf("got %d selected, want 2", len(r.Selected))
	}
	if r.Selected[0].Entry.ID != "mem_high" {
		t.Errorf("higher-weight topic should rank first, got %s", r.Selected[0].Entry.ID)
	}
}

func TestSurface_BelowThreshold(t *testing.T) {
	msi, vocab := surfaceFixture(IndexEntry{
		ID:          "mem_x",
		Title:       "Auth",
		Fingerprint: map[string]float64{"unrelated": 1.0},
	})
	r := Surface(SurfaceInputs{
		PromptVec: vec(map[string]float64{"jwt": 1.0, "session": 1.0}),
		Vocab:     vocab,
		Index:     msi,
	}, DefaultSurfaceConfig())
	if r.Block != "" {
		t.Errorf("orthogonal vectors should not surface, got %q", r.Block)
	}
}

func TestSurface_AboveThresholdRendersPointer(t *testing.T) {
	msi, vocab := surfaceFixture(IndexEntry{
		ID:          "mem_auth",
		Title:       "Auth & session model",
		Path:        "mem_auth.md",
		Fingerprint: map[string]float64{"jwt": 0.5, "session": 0.5},
	})
	cfg := DefaultSurfaceConfig()
	cfg.Threshold = 0.1
	r := Surface(SurfaceInputs{
		PromptVec: vec(map[string]float64{"jwt": 1.0, "session": 1.0}),
		Vocab:     vocab,
		Index:     msi,
	}, cfg)
	if len(r.Selected) != 1 || r.Selected[0].Entry.ID != "mem_auth" {
		t.Errorf("expected mem_auth selected, got %#v", r.Selected)
	}
	if !strings.Contains(r.Block, "Memory") {
		t.Errorf("block missing header: %q", r.Block)
	}
	if !strings.Contains(r.Block, "mem_auth.md") {
		t.Errorf("block missing path: %q", r.Block)
	}
	if !strings.Contains(r.Block, "Auth & session model") {
		t.Errorf("block missing title: %q", r.Block)
	}
	if !strings.Contains(r.Block, "[personal]") {
		t.Errorf("block missing source provenance: %q", r.Block)
	}
}

func TestSurface_TopKEnforced(t *testing.T) {
	entries := make([]IndexEntry, 0, 4)
	for _, id := range []string{"mem_a", "mem_b", "mem_c", "mem_d"} {
		entries = append(entries, IndexEntry{
			ID:          id,
			Title:       id,
			Path:        id + ".md",
			Fingerprint: map[string]float64{"jwt": 0.5},
		})
	}
	msi, vocab := surfaceFixture(entries...)
	cfg := DefaultSurfaceConfig()
	cfg.TopK = 2
	cfg.Threshold = 0.0
	r := Surface(SurfaceInputs{
		PromptVec: vec(map[string]float64{"jwt": 1.0}),
		Vocab:     vocab,
		Index:     msi,
	}, cfg)
	if len(r.Selected) != 2 {
		t.Errorf("expected topK=2, got %d", len(r.Selected))
	}
}

func TestSurface_BudgetTruncates(t *testing.T) {
	entries := make([]IndexEntry, 0, 3)
	for _, id := range []string{"mem_a", "mem_b", "mem_c"} {
		entries = append(entries, IndexEntry{
			ID:          id,
			Title:       strings.Repeat("padding ", 10),
			Path:        id + ".md",
			Fingerprint: map[string]float64{"jwt": 0.5},
		})
	}
	msi, vocab := surfaceFixture(entries...)
	cfg := DefaultSurfaceConfig()
	cfg.TopK = 3
	cfg.Threshold = 0.0
	cfg.MaxBlockChars = 400 // tight budget — at least one entry but not all three
	r := Surface(SurfaceInputs{
		PromptVec: vec(map[string]float64{"jwt": 1.0}),
		Vocab:     vocab,
		Index:     msi,
	}, cfg)
	if r.Block == "" {
		t.Fatal("expected a partial block under tight budget")
	}
	if len(r.Block) > cfg.MaxBlockChars+300 {
		t.Errorf("block exceeded budget: len=%d, max=%d", len(r.Block), cfg.MaxBlockChars)
	}
}

func TestSurface_AssetMatchesEndpoint(t *testing.T) {
	// Memory has "POST /auth/refresh" in assets; prompt mentions the
	// same endpoint inline. Asset tier should fire even though the
	// prompt vector has no overlap with the fingerprint.
	msi, vocab := surfaceFixture(IndexEntry{
		ID:          "mem_endpoint",
		Title:       "Auth refresh",
		Path:        "mem_endpoint.md",
		Assets:      []string{"POST /auth/refresh"},
		Fingerprint: map[string]float64{"unrelated": 1.0},
	})
	cfg := DefaultSurfaceConfig()
	r := Surface(SurfaceInputs{
		PromptText: "Wire up POST /auth/refresh handler please",
		PromptVec:  vec(map[string]float64{"unrelated": 0.0001}),
		Vocab:      vocab,
		Index:      msi,
	}, cfg)
	if len(r.Selected) != 1 || r.Selected[0].Entry.ID != "mem_endpoint" {
		t.Fatalf("expected endpoint asset hit, got %v", r.Selected)
	}
	if r.Selected[0].Reasons[0].Tier != "asset" {
		t.Errorf("dominant reason = %q, want asset", r.Selected[0].Reasons[0].Tier)
	}
}

// FrequencyBonus must lift heavily-touched memories above newer ones
// at equivalent cosine without letting an irrelevant memory cross the
// threshold just because it has a high TouchedBy count.
func TestSurface_FrequencyBonusReordersButDoesntLiftSubThreshold(t *testing.T) {
	// Two memories with identical assets — same base cosine.
	msi, vocab := surfaceFixture(
		IndexEntry{
			ID:          "mem_fresh",
			Title:       "Fresh",
			Path:        "mem_fresh.md",
			Assets:      []string{"cmd/api/auth.go"},
			Fingerprint: map[string]float64{"auth": 1.0},
			TouchedBy:   0,
		},
		IndexEntry{
			ID:          "mem_touched",
			Title:       "Touched",
			Path:        "mem_touched.md",
			Assets:      []string{"cmd/api/auth.go"},
			Fingerprint: map[string]float64{"auth": 1.0},
			TouchedBy:   100, // ~33% boost at FrequencyBonus=0.05
		},
	)

	cfg := DefaultSurfaceConfig()
	cfg.TopK = 2
	r := Surface(SurfaceInputs{
		PromptText: "fix cmd/api/auth.go",
		PromptVec:  vec(map[string]float64{"auth": 1.0}),
		Vocab:      vocab,
		Index:      msi,
	}, cfg)
	if len(r.Selected) != 2 {
		t.Fatalf("expected 2 selected, got %d", len(r.Selected))
	}
	if r.Selected[0].Entry.ID != "mem_touched" {
		t.Errorf("frequency bonus should rank touched first, got %q", r.Selected[0].Entry.ID)
	}
	if r.Selected[0].Score <= r.Selected[1].Score {
		t.Errorf("touched score (%.4f) should exceed fresh (%.4f)",
			r.Selected[0].Score, r.Selected[1].Score)
	}

	// Now force a sub-threshold candidate: low cosine but high
	// TouchedBy. The bonus must not lift it past the threshold.
	subMsi, _ := surfaceFixture(IndexEntry{
		ID:          "mem_irrelevant_but_touched",
		Title:       "Touched but irrelevant",
		Path:        "mem_irr.md",
		Fingerprint: map[string]float64{"unrelated": 0.001},
		TouchedBy:   1000,
	})
	cfg.Threshold = 0.5
	r = Surface(SurfaceInputs{
		PromptText: "totally different topic",
		PromptVec:  vec(map[string]float64{"different": 1.0}),
		Vocab:      vocab,
		Index:      subMsi,
	}, cfg)
	if len(r.Selected) != 0 {
		t.Errorf("frequency bonus should not lift sub-threshold matches, got %d selected", len(r.Selected))
	}
}
