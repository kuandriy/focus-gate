package main

import (
	"os"
	"path/filepath"
	"testing"
)

// samePath compares two filesystem paths after resolving symlinks. macOS
// puts t.TempDir under /var, which is a symlink to /private/var, so a
// direct string comparison with os.Getwd() fails. EvalSymlinks gives us a
// canonical form for both.
func samePath(t *testing.T, a, b string) bool {
	t.Helper()
	ra, errA := filepath.EvalSymlinks(a)
	rb, errB := filepath.EvalSymlinks(b)
	if errA != nil || errB != nil {
		return a == b
	}
	return ra == rb
}

// TestResolveConfigFilePrefersProjectLocal verifies that a .focus-gate.json in
// the current working directory wins over $FOCUS_GATE_CONFIG and the binary-
// adjacent global fallback.
func TestResolveConfigFilePrefersProjectLocal(t *testing.T) {
	dir := t.TempDir()
	projectCfg := filepath.Join(dir, ".focus-gate.json")
	if err := os.WriteFile(projectCfg, []byte("{}"), 0644); err != nil {
		t.Fatal(err)
	}

	// Point env at a separate path just to prove project-local wins.
	envCfg := filepath.Join(t.TempDir(), "env-cfg.json")
	if err := os.WriteFile(envCfg, []byte("{}"), 0644); err != nil {
		t.Fatal(err)
	}
	t.Setenv("FOCUS_GATE_CONFIG", envCfg)

	oldWd, _ := os.Getwd()
	if err := os.Chdir(dir); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = os.Chdir(oldWd) })

	got := resolveConfigFile()
	if !samePath(t, got, projectCfg) {
		t.Errorf("resolveConfigFile = %q, want project-local %q", got, projectCfg)
	}
}

// TestResolveConfigFileFallsBackToEnvVar verifies $FOCUS_GATE_CONFIG is used
// when no project-local .focus-gate.json exists.
func TestResolveConfigFileFallsBackToEnvVar(t *testing.T) {
	// cwd without a .focus-gate.json.
	dir := t.TempDir()
	oldWd, _ := os.Getwd()
	if err := os.Chdir(dir); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = os.Chdir(oldWd) })

	envCfg := filepath.Join(t.TempDir(), "env-cfg.json")
	if err := os.WriteFile(envCfg, []byte("{}"), 0644); err != nil {
		t.Fatal(err)
	}
	t.Setenv("FOCUS_GATE_CONFIG", envCfg)

	got := resolveConfigFile()
	if !samePath(t, got, envCfg) {
		t.Errorf("resolveConfigFile = %q, want env %q", got, envCfg)
	}
}

// TestResolveConfigFileFallsBackToBinaryDir verifies the global fallback kicks
// in when neither project-local nor env paths are usable.
func TestResolveConfigFileFallsBackToBinaryDir(t *testing.T) {
	dir := t.TempDir()
	oldWd, _ := os.Getwd()
	if err := os.Chdir(dir); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = os.Chdir(oldWd) })
	t.Setenv("FOCUS_GATE_CONFIG", "")

	got := resolveConfigFile()
	// The binary location comes from os.Executable(); under `go test` this
	// will be the test binary, not the real focus-gate binary. The only
	// property we can assert portably is "non-empty ending in config.json".
	if got == "" || filepath.Base(got) != "config.json" {
		t.Errorf("resolveConfigFile = %q, want a path ending in config.json", got)
	}
}
