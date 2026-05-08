package memory

import (
	"testing"
	"time"
)

// buildIndexedMemory writes a v2 memory with the given metadata and
// returns the resulting IndexEntry. Saves the manifest so subsequent
// Load picks it up.
func buildIndexEntry(id, source, title string, assets, interests, topics []string) IndexEntry {
	wInterests := make([]WeightedEntry, len(interests))
	for i, n := range interests {
		wInterests[i] = WeightedEntry{Name: n, Weight: 1.0}
	}
	wTopics := make([]WeightedEntry, len(topics))
	for i, n := range topics {
		wTopics[i] = WeightedEntry{Name: n, Weight: 1.0}
	}
	return IndexEntry{
		ID:        id,
		Source:    source,
		Title:     title,
		Path:      id + ".md",
		Version:   1,
		Chapters:  1,
		Assets:    assets,
		Interests: wInterests,
		Topics:    wTopics,
		Updated:   time.Now().UTC().Truncate(time.Second),
	}
}

func TestRebuildInvertedIndexes_BuildsExpectedKeys(t *testing.T) {
	mf := NewManifest()
	mf.Entries = []IndexEntry{
		buildIndexEntry("mem_a", "personal", "Auth",
			[]string{"cmd/api/auth.go", "POST /auth/login"},
			[]string{"session lifecycle"},
			[]string{"JWT authentication"}),
		buildIndexEntry("mem_b", "personal", "Rate limit",
			[]string{"middleware/rate.go"},
			[]string{"rate limiting"},
			[]string{"middleware ordering"}),
	}
	mf.RebuildInvertedIndexes()

	if got := mf.ByAsset["cmd/api/auth.go"]; len(got) != 1 || got[0] != "mem_a" {
		t.Errorf("ByAsset[cmd/api/auth.go] = %v, want [mem_a]", got)
	}
	if got := mf.ByAsset["post /auth/login"]; len(got) != 1 || got[0] != "mem_a" {
		t.Errorf("ByAsset[post /auth/login] = %v, want [mem_a] (case-insensitive)", got)
	}
	if got := mf.ByInterest["rate limiting"]; len(got) != 1 || got[0] != "mem_b" {
		t.Errorf("ByInterest[rate limiting] = %v", got)
	}
	if got := mf.ByTopic["jwt authentication"]; len(got) != 1 || got[0] != "mem_a" {
		t.Errorf("ByTopic[jwt authentication] = %v", got)
	}
}

func TestRebuildInvertedIndexes_SharedKeyCollectsAllIDs(t *testing.T) {
	mf := NewManifest()
	mf.Entries = []IndexEntry{
		buildIndexEntry("mem_a", "personal", "A", []string{"shared.go"}, nil, nil),
		buildIndexEntry("mem_b", "personal", "B", []string{"shared.go"}, nil, nil),
	}
	mf.RebuildInvertedIndexes()
	got := mf.ByAsset["shared.go"]
	if len(got) != 2 || got[0] != "mem_a" || got[1] != "mem_b" {
		t.Errorf("expected sorted [mem_a, mem_b], got %v", got)
	}
}

func TestMultiSourceIndex_LookupAcrossSources(t *testing.T) {
	a := NewManifest()
	a.Source = "personal"
	a.Entries = []IndexEntry{buildIndexEntry("mem_a", "personal", "A", []string{"shared.go"}, nil, nil)}
	a.RebuildInvertedIndexes()

	b := NewManifest()
	b.Source = "team"
	b.Entries = []IndexEntry{buildIndexEntry("mem_b", "team", "B", []string{"shared.go", "other.go"}, nil, nil)}
	b.RebuildInvertedIndexes()

	msi := NewMultiSourceIndex(a, b)
	got := msi.LookupByAsset("shared.go")
	if len(got) != 2 {
		t.Errorf("expected 2 matches across sources, got %d", len(got))
	}
	got = msi.LookupByAsset("other.go")
	if len(got) != 1 || got[0].ID != "mem_b" {
		t.Errorf("expected only mem_b, got %v", got)
	}
}

func TestMultiSourceIndex_TouchPropagatesToOwningManifest(t *testing.T) {
	a := NewManifest()
	a.Source = "personal"
	a.Entries = []IndexEntry{buildIndexEntry("mem_a", "personal", "A", nil, nil, nil)}
	_ = a.Save(t.TempDir())
	a.dirty = false // Clear after Save so Touch is the only dirty trigger.

	msi := NewMultiSourceIndex(a)
	msi.Touch("mem_a")

	got, _ := a.Get("mem_a")
	if got.TouchedBy != 1 {
		t.Errorf("touchedBy = %d, want 1", got.TouchedBy)
	}
	if !a.Dirty() {
		t.Error("expected manifest to be dirty after Touch")
	}
}

func TestMultiSourceIndex_EmptyAndNilSafe(t *testing.T) {
	if !((*MultiSourceIndex)(nil)).Empty() {
		t.Error("nil MultiSourceIndex should be Empty")
	}
	msi := NewMultiSourceIndex(nil, NewManifest())
	if !msi.Empty() {
		t.Error("manifest with no entries should be Empty")
	}
	// Nil manifest should be filtered.
	if len(msi.Manifests) != 1 {
		t.Errorf("nil manifest should be filtered, got %d", len(msi.Manifests))
	}
}
