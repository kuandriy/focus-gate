package memory

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"time"

	"github.com/kuandriy/focus-gate/internal/persist"
	"github.com/kuandriy/focus-gate/internal/tfidf"
)

// PendingFilename is the on-disk name of the candidate queue. Sits next
// to the memories/ subdirectory so it travels with the project data dir
// but isn't accidentally treated as a memory file.
const PendingFilename = "pending_memories.json"

// PendingQueue is the append-only list of candidates awaiting promotion.
// Cooldowns maps tree ID → last candidate time so SelectCandidate can
// respect the promotion cooldown across the runtime's lifetime.
type PendingQueue struct {
	SchemaVersion string               `json:"schemaVersion"`
	Candidates    []*Candidate         `json:"candidates"`
	Cooldowns     map[string]time.Time `json:"cooldowns,omitempty"`
	UpdatedAt     time.Time            `json:"updatedAt"`
}

// NewPendingQueue returns an empty queue stamped with the current schema.
func NewPendingQueue() *PendingQueue {
	return &PendingQueue{
		SchemaVersion: SchemaVersion,
		Cooldowns:     map[string]time.Time{},
	}
}

// PendingPath returns the on-disk path for the pending queue inside a
// project's data directory.
func PendingPath(dataDir string) string {
	return filepath.Join(dataDir, PendingFilename)
}

// LoadPending reads the queue from disk, dropping stale entries older
// than maxAge. A missing file is returned as an empty queue. A malformed
// file is reported via err — the caller may choose to start fresh.
//
// Passing maxAge <= 0 disables age-out.
func LoadPending(dataDir string, maxAge time.Duration) (*PendingQueue, error) {
	path := PendingPath(dataDir)
	data, err := os.ReadFile(path)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return NewPendingQueue(), nil
		}
		return nil, fmt.Errorf("read pending queue: %w", err)
	}
	q := NewPendingQueue()
	if err := json.Unmarshal(data, q); err != nil {
		return nil, fmt.Errorf("parse pending queue: %w", err)
	}
	if q.SchemaVersion != "" && q.SchemaVersion != SchemaVersion {
		return NewPendingQueue(), fmt.Errorf("pending queue schemaVersion %q, binary speaks %q — starting fresh",
			q.SchemaVersion, SchemaVersion)
	}
	if q.Cooldowns == nil {
		q.Cooldowns = map[string]time.Time{}
	}
	if maxAge > 0 {
		q.Candidates = filterByAge(q.Candidates, maxAge)
	}
	return q, nil
}

// filterByAge drops candidates whose CreatedAt is older than maxAge.
// Applied on load, not save — we keep whatever's on disk durable until
// a load actually happens, so crashes can't silently rewrite state.
func filterByAge(cands []*Candidate, maxAge time.Duration) []*Candidate {
	cutoff := time.Now().Add(-maxAge)
	out := cands[:0]
	for _, c := range cands {
		if c == nil {
			continue
		}
		if c.CreatedAt.Before(cutoff) {
			continue
		}
		out = append(out, c)
	}
	return out
}

// Save persists the queue atomically. Always stamps SchemaVersion +
// UpdatedAt so consumers can tell whether the queue has moved.
func (q *PendingQueue) Save(dataDir string) error {
	q.SchemaVersion = SchemaVersion
	q.UpdatedAt = time.Now().UTC().Truncate(time.Second)
	return persist.SaveAtomic(PendingPath(dataDir), q)
}

// AppendCandidates adds new candidates to the queue, deduplicating
// against what's already queued. Returns the number of new entries
// actually added (a caller-visible signal that anything was queued on
// this run — used to decide whether to emit the surface-time nudge).
//
// The dedup here complements DedupCandidates: that function handles
// the in-batch case; this one handles cross-run dedup when the user
// hasn't promoted for a while.
func (q *PendingQueue) AppendCandidates(newCands []*Candidate, dedupCosine float64) int {
	if dedupCosine <= 0 {
		dedupCosine = 0.85
	}
	added := 0
	for _, c := range newCands {
		if c == nil {
			continue
		}
		if q.alreadyQueued(c, dedupCosine) {
			continue
		}
		q.Candidates = append(q.Candidates, c)
		q.Cooldowns[c.SourceTreeID] = c.CreatedAt
		added++
	}
	return added
}

// alreadyQueued returns true if candidate c duplicates something already
// in the queue — either by exact TempID, exact SourceTreeID, or by
// fingerprint cosine above the threshold.
func (q *PendingQueue) alreadyQueued(c *Candidate, threshold float64) bool {
	cv := tfidf.NewVector(c.Fingerprint)
	for _, existing := range q.Candidates {
		if existing.TempID == c.TempID || existing.SourceTreeID == c.SourceTreeID {
			return true
		}
		ev := tfidf.NewVector(existing.Fingerprint)
		if tfidf.CosineSimilarity(cv, ev) >= threshold {
			return true
		}
	}
	return false
}

// Remove deletes a candidate by TempID. Returns true if removed. Used
// by `fg: memory commit` (per successful decision) and
// `fg: memory discard <id>` (manual cleanup).
func (q *PendingQueue) Remove(tempID string) bool {
	for i, c := range q.Candidates {
		if c.TempID == tempID {
			q.Candidates = append(q.Candidates[:i], q.Candidates[i+1:]...)
			return true
		}
	}
	return false
}

// Clear drops every pending entry. Invoked by `fg: memory discard all`
// when the user wants a clean slate.
func (q *PendingQueue) Clear() int {
	n := len(q.Candidates)
	q.Candidates = nil
	return n
}

// Sorted returns the candidates in deterministic order (newest first).
// Not stored on disk — just used for display paths like
// `fg: memory pending`.
func (q *PendingQueue) Sorted() []*Candidate {
	sorted := make([]*Candidate, len(q.Candidates))
	copy(sorted, q.Candidates)
	sort.Slice(sorted, func(i, j int) bool {
		return sorted[i].CreatedAt.After(sorted[j].CreatedAt)
	})
	return sorted
}
