package memory

import (
	"sort"
)

// MultiSourceIndex composes manifests from every enabled memory source
// into one logical view for surface-time lookup. Phase 5 wires this to
// real `memory.sources` config; before then, callers construct it with
// just the synthesized "personal" manifest.
//
// The index does not duplicate IndexEntry data — it iterates the
// underlying manifests directly, so the source manifests remain the
// authoritative store and Touch increments propagate back into the
// owning manifest for Save.
type MultiSourceIndex struct {
	Manifests []*Manifest
}

// NewMultiSourceIndex composes a MultiSourceIndex from the supplied
// manifests in attach order. Nil entries are skipped so callers can
// pass a slice that includes manifests for sources that failed to load.
func NewMultiSourceIndex(manifests ...*Manifest) *MultiSourceIndex {
	live := make([]*Manifest, 0, len(manifests))
	for _, m := range manifests {
		if m == nil {
			continue
		}
		live = append(live, m)
	}
	return &MultiSourceIndex{Manifests: live}
}

// Empty reports whether the index has no entries across any manifest.
// Useful guard for the Surface fast path.
func (msi *MultiSourceIndex) Empty() bool {
	if msi == nil {
		return true
	}
	for _, m := range msi.Manifests {
		if m != nil && len(m.Entries) > 0 {
			return false
		}
	}
	return true
}

// AllEntries returns every IndexEntry across every manifest, in
// source-attach order followed by per-source insertion order. Callers
// that need deterministic top-K behaviour should sort the returned
// slice on a tiebreak key before truncating.
func (msi *MultiSourceIndex) AllEntries() []IndexEntry {
	if msi == nil {
		return nil
	}
	total := 0
	for _, m := range msi.Manifests {
		total += len(m.Entries)
	}
	out := make([]IndexEntry, 0, total)
	for _, m := range msi.Manifests {
		out = append(out, m.Entries...)
	}
	return out
}

// LookupByAsset returns IDs of memories matching the given asset across
// every manifest. Lookup is case-insensitive (uses normalizeKey). The
// returned slice is sorted by ID for deterministic ordering, mirroring
// the per-manifest contract.
func (msi *MultiSourceIndex) LookupByAsset(asset string) []IndexEntry {
	return msi.lookup(asset, func(m *Manifest) map[string][]string { return m.ByAsset })
}

// LookupByInterest mirrors LookupByAsset for the interests inverted
// index.
func (msi *MultiSourceIndex) LookupByInterest(name string) []IndexEntry {
	return msi.lookup(name, func(m *Manifest) map[string][]string { return m.ByInterest })
}

// LookupByTopic mirrors LookupByAsset for the topics inverted index.
func (msi *MultiSourceIndex) LookupByTopic(name string) []IndexEntry {
	return msi.lookup(name, func(m *Manifest) map[string][]string { return m.ByTopic })
}

func (msi *MultiSourceIndex) lookup(rawKey string, sel func(*Manifest) map[string][]string) []IndexEntry {
	if msi == nil {
		return nil
	}
	key := normalizeKey(rawKey)
	if key == "" {
		return nil
	}
	var out []IndexEntry
	seen := map[string]bool{}
	for _, m := range msi.Manifests {
		idx := sel(m)
		if idx == nil {
			continue
		}
		ids := idx[key]
		for _, id := range ids {
			if seen[id] {
				continue
			}
			if entry, ok := m.Get(id); ok {
				out = append(out, *entry)
				seen[id] = true
			}
		}
	}
	sort.Slice(out, func(i, j int) bool { return out[i].ID < out[j].ID })
	return out
}

// Touch increments the touchedBy counter for an entry. Searches every
// manifest until the matching ID is found; missing IDs are silently
// ignored. Marks the owning manifest dirty so callers that batch-save
// at end-of-prompt pick up the change.
func (msi *MultiSourceIndex) Touch(id string) {
	if msi == nil {
		return
	}
	for _, m := range msi.Manifests {
		if _, ok := m.Get(id); !ok {
			continue
		}
		m.Touch(id)
		return
	}
}

// DirtyManifests returns the manifests whose dirty flag is set —
// callers iterate this list to issue per-source SaveIfDirty.
func (msi *MultiSourceIndex) DirtyManifests() []*Manifest {
	if msi == nil {
		return nil
	}
	var out []*Manifest
	for _, m := range msi.Manifests {
		if m != nil && m.Dirty() {
			out = append(out, m)
		}
	}
	return out
}
