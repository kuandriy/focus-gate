package memory

import (
	"testing"
	"time"
)

func makeCandidate(treeID, tempID string) *Candidate {
	return &Candidate{
		SchemaVersion: SchemaVersion,
		TempID:        tempID,
		SourceTreeID:  treeID,
		Fingerprint:   map[string]float64{"x": 1.0, treeID: 0.5},
		CreatedAt:     time.Now().UTC().Truncate(time.Second),
	}
}

func TestLoadPending_MissingFileReturnsEmpty(t *testing.T) {
	dir := t.TempDir()
	q, err := LoadPending(dir, 0)
	if err != nil {
		t.Fatalf("missing file should not error, got %v", err)
	}
	if q == nil || len(q.Candidates) != 0 {
		t.Error("missing file should return empty queue")
	}
}

func TestPending_SaveRoundTrip(t *testing.T) {
	dir := t.TempDir()
	q := NewPendingQueue()
	added := q.AppendCandidates([]*Candidate{makeCandidate("t1", "a"), makeCandidate("t2", "b")}, 0.85)
	if added != 2 {
		t.Errorf("expected 2 added, got %d", added)
	}
	if err := q.Save(dir); err != nil {
		t.Fatal(err)
	}

	q2, err := LoadPending(dir, 0)
	if err != nil {
		t.Fatal(err)
	}
	if len(q2.Candidates) != 2 {
		t.Errorf("round-trip lost entries: got %d, want 2", len(q2.Candidates))
	}
	if _, ok := q2.Cooldowns["t1"]; !ok {
		t.Error("cooldown should have been recorded on append")
	}
}

func TestPending_AppendDedupsBySourceTree(t *testing.T) {
	q := NewPendingQueue()
	added := q.AppendCandidates([]*Candidate{
		makeCandidate("t1", "a"),
		makeCandidate("t1", "b"), // same source tree — should dedup
	}, 0.85)
	if added != 1 {
		t.Errorf("expected dedup by source tree, got added=%d", added)
	}
}

func TestPending_AgeOutDropsOldEntries(t *testing.T) {
	dir := t.TempDir()
	q := NewPendingQueue()
	fresh := makeCandidate("fresh", "fresh")
	old := makeCandidate("old", "old")
	old.CreatedAt = time.Now().Add(-48 * time.Hour).UTC().Truncate(time.Second)
	q.AppendCandidates([]*Candidate{fresh, old}, 0.85)
	if err := q.Save(dir); err != nil {
		t.Fatal(err)
	}

	q2, err := LoadPending(dir, 24*time.Hour)
	if err != nil {
		t.Fatal(err)
	}
	if len(q2.Candidates) != 1 || q2.Candidates[0].TempID != "fresh" {
		t.Errorf("age-out should drop 48h-old entry, got %+v", q2.Candidates)
	}
}

func TestPending_RemoveByTempID(t *testing.T) {
	q := NewPendingQueue()
	q.AppendCandidates([]*Candidate{makeCandidate("t1", "a"), makeCandidate("t2", "b")}, 0.85)
	if !q.Remove("a") {
		t.Error("Remove should return true for existing tempId")
	}
	if len(q.Candidates) != 1 {
		t.Errorf("expected 1 remaining candidate, got %d", len(q.Candidates))
	}
	if q.Remove("missing") {
		t.Error("Remove should return false for unknown tempId")
	}
}

func TestPending_ClearEmptiesQueue(t *testing.T) {
	q := NewPendingQueue()
	q.AppendCandidates([]*Candidate{makeCandidate("t1", "a"), makeCandidate("t2", "b")}, 0.85)
	n := q.Clear()
	if n != 2 {
		t.Errorf("Clear returned %d, want 2", n)
	}
	if len(q.Candidates) != 0 {
		t.Errorf("queue should be empty after Clear, got %d", len(q.Candidates))
	}
}
