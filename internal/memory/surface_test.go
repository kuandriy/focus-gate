package memory

import (
	"strings"
	"testing"

	"github.com/kuandriy/focus-gate/internal/tfidf"
)

// vec is a tiny helper that builds a tfidf.Vector from a term→weight map.
// Mirrors NewVector but lives here so tests can be terse.
func vec(weights map[string]float64) tfidf.Vector {
	return tfidf.NewVector(weights)
}

func TestSurface_DisabledOrEmpty(t *testing.T) {
	mf := NewManifest()
	mf.Upsert(IndexEntry{ID: "mem_x", Title: "x", Fingerprint: map[string]float64{"a": 1.0}})
	cfg := DefaultSurfaceConfig()

	// Disabled.
	cfg.Enabled = false
	r := Surface(vec(map[string]float64{"a": 1.0}), mf, cfg)
	if r.Block != "" || len(r.Selected) != 0 {
		t.Error("disabled config should return empty SurfaceResult")
	}

	// Empty manifest.
	cfg = DefaultSurfaceConfig()
	r = Surface(vec(map[string]float64{"a": 1.0}), NewManifest(), cfg)
	if r.Block != "" || len(r.Selected) != 0 {
		t.Error("empty manifest should return empty SurfaceResult")
	}

	// Empty prompt vector.
	r = Surface(nil, mf, cfg)
	if r.Block != "" || len(r.Selected) != 0 {
		t.Error("nil prompt vector should return empty SurfaceResult")
	}
}

func TestSurface_BelowThreshold(t *testing.T) {
	mf := NewManifest()
	mf.Upsert(IndexEntry{
		ID:          "mem_x",
		Title:       "Auth",
		Fingerprint: map[string]float64{"unrelated": 1.0},
	})
	cfg := DefaultSurfaceConfig()
	r := Surface(vec(map[string]float64{"jwt": 1.0, "session": 1.0}), mf, cfg)
	if r.Block != "" {
		t.Errorf("orthogonal vectors should not surface, got %q", r.Block)
	}
}

func TestSurface_AboveThresholdRendersPointer(t *testing.T) {
	mf := NewManifest()
	mf.Upsert(IndexEntry{
		ID:          "mem_auth",
		Title:       "Auth & session model",
		Path:        "mem_auth.md",
		Fingerprint: map[string]float64{"jwt": 0.5, "session": 0.5},
	})
	cfg := DefaultSurfaceConfig()
	cfg.Threshold = 0.1
	r := Surface(vec(map[string]float64{"jwt": 1.0, "session": 1.0}), mf, cfg)
	if len(r.Selected) != 1 || r.Selected[0].ID != "mem_auth" {
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
}

func TestSurface_TopKEnforced(t *testing.T) {
	mf := NewManifest()
	for _, id := range []string{"mem_a", "mem_b", "mem_c", "mem_d"} {
		mf.Upsert(IndexEntry{
			ID:          id,
			Title:       id,
			Path:        id + ".md",
			Fingerprint: map[string]float64{"jwt": 0.5},
		})
	}
	cfg := DefaultSurfaceConfig()
	cfg.TopK = 2
	cfg.Threshold = 0.0
	r := Surface(vec(map[string]float64{"jwt": 1.0}), mf, cfg)
	if len(r.Selected) != 2 {
		t.Errorf("expected topK=2, got %d", len(r.Selected))
	}
}

func TestSurface_BudgetTruncates(t *testing.T) {
	mf := NewManifest()
	for _, id := range []string{"mem_a", "mem_b", "mem_c"} {
		mf.Upsert(IndexEntry{
			ID:          id,
			Title:       strings.Repeat("padding ", 10),
			Path:        id + ".md",
			Fingerprint: map[string]float64{"jwt": 0.5},
		})
	}
	cfg := DefaultSurfaceConfig()
	cfg.TopK = 3
	cfg.Threshold = 0.0
	cfg.MaxBlockChars = 200 // small budget — at least one entry but not all three
	r := Surface(vec(map[string]float64{"jwt": 1.0}), mf, cfg)
	if r.Block == "" {
		t.Fatal("expected a partial block under tight budget")
	}
	if len(r.Block) > cfg.MaxBlockChars+200 {
		// Generous slack — header is fixed, but we should not exceed by much.
		t.Errorf("block exceeded budget: len=%d, max=%d", len(r.Block), cfg.MaxBlockChars)
	}
}
