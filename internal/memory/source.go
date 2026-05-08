package memory

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/kuandriy/focus-gate/internal/persist"
)

// SourcesFilename is the on-disk name of the source registry, stored
// next to the other state files in the project's data directory.
const SourcesFilename = "sources.json"

// Source identifies one attached memory directory. Multi-source attach
// lets the user point at a shared team-wide repo alongside their
// personal memories so a single Surface call can pull from both.
//
// Personal is the synthesized default — Detach refuses to remove it.
// Disabled sources stay in the registry but are skipped during Surface,
// which is how the "private review" workflow works (disable shared,
// promote → all writes land in personal).
type Source struct {
	Name     string `json:"name"`
	Path     string `json:"path"`
	Enabled  bool   `json:"enabled"`
	Writable bool   `json:"writable"`
}

// SourceRegistry is the registry persisted as `<dataDir>/sources.json`.
// Layered on top of any default-synthesized "personal" entry — callers
// load via LoadSources, which applies the fallback automatically.
type SourceRegistry struct {
	SchemaVersion string   `json:"schemaVersion"`
	Default       string   `json:"default"`
	Sources       []Source `json:"sources"`
}

// LoadSources reads the source registry from disk. If the file is
// missing it returns a synthesized registry containing the default
// "personal" source pointing at fallbackPath, and Default="personal".
//
// Schema-version drift causes a fresh registry rather than an error so
// the user's prompt isn't blocked by a malformed sources.json.
func LoadSources(dataDir, fallbackPath string) (*SourceRegistry, error) {
	r := defaultRegistry(fallbackPath)
	path := filepath.Join(dataDir, SourcesFilename)
	data, err := os.ReadFile(path)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return r, nil
		}
		return r, fmt.Errorf("read sources: %w", err)
	}
	var loaded SourceRegistry
	if err := json.Unmarshal(data, &loaded); err != nil {
		return r, fmt.Errorf("parse sources: %w", err)
	}
	if loaded.SchemaVersion != "" && loaded.SchemaVersion != SchemaVersion {
		return r, fmt.Errorf("sources schemaVersion %q, binary speaks %q — starting fresh",
			loaded.SchemaVersion, SchemaVersion)
	}
	// Make sure personal is always present so users can't accidentally
	// remove the path to their local memories by hand-editing the file.
	loaded.ensurePersonal(fallbackPath)
	if loaded.Default == "" {
		loaded.Default = DefaultSourceName
	}
	return &loaded, nil
}

func defaultRegistry(personalPath string) *SourceRegistry {
	return &SourceRegistry{
		SchemaVersion: SchemaVersion,
		Default:       DefaultSourceName,
		Sources: []Source{{
			Name:     DefaultSourceName,
			Path:     personalPath,
			Enabled:  true,
			Writable: true,
		}},
	}
}

// Save persists the registry atomically. Sorts sources by name (with
// personal pinned to the front) so on-disk diffs stay stable.
func (r *SourceRegistry) Save(dataDir string) error {
	if err := os.MkdirAll(dataDir, 0755); err != nil {
		return err
	}
	r.SchemaVersion = SchemaVersion
	r.sort()
	return persist.SaveAtomic(filepath.Join(dataDir, SourcesFilename), r)
}

func (r *SourceRegistry) sort() {
	sort.SliceStable(r.Sources, func(i, j int) bool {
		if r.Sources[i].Name == DefaultSourceName {
			return true
		}
		if r.Sources[j].Name == DefaultSourceName {
			return false
		}
		return r.Sources[i].Name < r.Sources[j].Name
	})
}

// Get returns a pointer to the named source and whether it was found.
// The pointer is into the underlying slice, so callers can mutate it
// in place; Save persists the result.
func (r *SourceRegistry) Get(name string) (*Source, bool) {
	for i := range r.Sources {
		if r.Sources[i].Name == name {
			return &r.Sources[i], true
		}
	}
	return nil, false
}

// Attach adds a new source. Returns an error if the name is empty,
// already attached, or the path is missing/unreadable.
func (r *SourceRegistry) Attach(name, path string, writable bool) error {
	name = strings.TrimSpace(name)
	if name == "" {
		return errors.New("source name required")
	}
	if _, ok := r.Get(name); ok {
		return fmt.Errorf("source %q already attached", name)
	}
	if path == "" {
		return errors.New("source path required")
	}
	if _, err := os.Stat(path); err != nil {
		return fmt.Errorf("source path %q: %w", path, err)
	}
	r.Sources = append(r.Sources, Source{
		Name:     name,
		Path:     path,
		Enabled:  true,
		Writable: writable,
	})
	return nil
}

// Detach removes a source by name. Refuses to remove "personal" — the
// user can disable it but losing the path to non-empty local memories
// would be too destructive. If the detached source was the default,
// the default is moved back to "personal".
func (r *SourceRegistry) Detach(name string) error {
	if name == DefaultSourceName {
		return fmt.Errorf("cannot detach %q — disable it instead", DefaultSourceName)
	}
	for i := range r.Sources {
		if r.Sources[i].Name == name {
			r.Sources = append(r.Sources[:i], r.Sources[i+1:]...)
			if r.Default == name {
				r.Default = DefaultSourceName
			}
			return nil
		}
	}
	return fmt.Errorf("source %q not attached", name)
}

// Enable / Disable flip the per-source flag. Enabled sources participate
// in Surface; disabled ones do not (writes still work to keep edit/
// archive flows simple).
func (r *SourceRegistry) Enable(name string) error  { return r.setEnabled(name, true) }
func (r *SourceRegistry) Disable(name string) error { return r.setEnabled(name, false) }

func (r *SourceRegistry) setEnabled(name string, v bool) error {
	s, ok := r.Get(name)
	if !ok {
		return fmt.Errorf("source %q not attached", name)
	}
	s.Enabled = v
	return nil
}

// SetDefault names the source where new memories go when a commit
// payload omits targetSource. Errors when the requested source isn't
// attached or isn't writable — non-writable sources cannot accept
// writes, defaulting to one would silently break the create path.
func (r *SourceRegistry) SetDefault(name string) error {
	s, ok := r.Get(name)
	if !ok {
		return fmt.Errorf("source %q not attached", name)
	}
	if !s.Writable {
		return fmt.Errorf("source %q is read-only — cannot be the default write target", name)
	}
	r.Default = name
	return nil
}

// EnabledSources returns the subset of sources flagged enabled, in the
// registry's stored order (personal first).
func (r *SourceRegistry) EnabledSources() []Source {
	out := make([]Source, 0, len(r.Sources))
	for _, s := range r.Sources {
		if s.Enabled {
			out = append(out, s)
		}
	}
	return out
}

// ensurePersonal guarantees a "personal" entry exists, synthesizing
// one with fallbackPath when the on-disk registry has somehow lost it.
func (r *SourceRegistry) ensurePersonal(fallbackPath string) {
	for i := range r.Sources {
		if r.Sources[i].Name == DefaultSourceName {
			if r.Sources[i].Path == "" {
				r.Sources[i].Path = fallbackPath
			}
			return
		}
	}
	r.Sources = append([]Source{{
		Name:     DefaultSourceName,
		Path:     fallbackPath,
		Enabled:  true,
		Writable: true,
	}}, r.Sources...)
}

// LoadEnabledManifests loads each enabled source's manifest, runs
// EnsureFresh against its directory, and tags every entry with its
// source name. Returned manifests are in registry order so callers
// can compose a MultiSourceIndex deterministically.
//
// Per-source load errors are returned alongside successful manifests
// so the hook can surface them without aborting.
func (r *SourceRegistry) LoadEnabledManifests(vocab VocabSnapshot) ([]*Manifest, []error) {
	var manifests []*Manifest
	var errs []error
	for _, s := range r.EnabledSources() {
		mf, perErrs := EnsureFresh(s.Path, vocab)
		// Stamp the configured source name onto manifest + entries.
		// EnsureFresh defaults to "personal" — we override here so a
		// shared directory's manifest reports its real attach name.
		mf.Source = s.Name
		for i := range mf.Entries {
			mf.Entries[i].Source = s.Name
		}
		manifests = append(manifests, mf)
		for _, e := range perErrs {
			errs = append(errs, fmt.Errorf("%s: %w", s.Name, e))
		}
	}
	return manifests, errs
}

// SourceDirs returns the {name → path} map for writable sources, ready
// to feed into CommitContext.SourceDirs. Read-only sources are
// filtered so the commit path can rely on the map's presence as a
// "source is writable" check.
func (r *SourceRegistry) SourceDirs() map[string]string {
	out := map[string]string{}
	for _, s := range r.Sources {
		if !s.Writable {
			continue
		}
		out[s.Name] = s.Path
	}
	return out
}
