package main

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/kuandriy/focus-gate/internal/memory"
	"github.com/kuandriy/focus-gate/internal/tfidf"
)

// writeTestMemory persists a v2 memory file to dir and returns the
// resulting Memory pointer (for ID and Path access). Used by the
// forget tests to set up a target file.
func writeTestMemory(t *testing.T, dir, title string, vocab memory.VocabSnapshot) *memory.Memory {
	t.Helper()
	m := &memory.Memory{Title: title}
	ch := memory.Chapter{
		Date:       time.Date(2026, 4, 1, 0, 0, 0, 0, time.UTC),
		Title:      "Initial",
		TimeMarker: "2026-04-01",
		What:       "did",
		Why:        "because",
	}
	if err := memory.AppendChapter(m, ch); err != nil {
		t.Fatal(err)
	}
	if err := memory.WriteFile(dir, m, vocab); err != nil {
		t.Fatal(err)
	}
	return m
}

func TestSlashMemoryForget_DryRunDoesNotDelete(t *testing.T) {
	dataDir := t.TempDir()
	memDir := filepath.Join(dataDir, "memories")
	if err := os.MkdirAll(memDir, 0755); err != nil {
		t.Fatal(err)
	}
	engine := tfidf.NewEngine()
	vocab := memory.NewVocabSnapshot(engine)
	m := writeTestMemory(t, memDir, "Auth", vocab)

	p := paths{dataDir: dataDir, memoryDir: memDir}
	var buf bytes.Buffer
	if err := slashMemoryForget(&buf, p, engine, m.ID); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(buf.String(), "Dry run") {
		t.Errorf("expected dry-run output, got %q", buf.String())
	}
	if _, err := os.Stat(m.Path(memDir)); err != nil {
		t.Errorf("dry run should not have removed the file: %v", err)
	}
}

func TestSlashMemoryForget_YesDeletesFileAndManifestEntry(t *testing.T) {
	dataDir := t.TempDir()
	memDir := filepath.Join(dataDir, "memories")
	if err := os.MkdirAll(memDir, 0755); err != nil {
		t.Fatal(err)
	}
	engine := tfidf.NewEngine()
	vocab := memory.NewVocabSnapshot(engine)
	m := writeTestMemory(t, memDir, "Auth", vocab)

	// Prime the manifest so we can verify it's pruned.
	mf, _ := memory.Load(memDir)
	_ = mf.Rebuild(memDir, vocab)
	_ = mf.Save(memDir)

	p := paths{dataDir: dataDir, memoryDir: memDir}
	var buf bytes.Buffer
	if err := slashMemoryForget(&buf, p, engine, m.ID+" --yes"); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(buf.String(), "removed") {
		t.Errorf("expected removed confirmation, got %q", buf.String())
	}
	if _, err := os.Stat(m.Path(memDir)); !os.IsNotExist(err) {
		t.Errorf("file still on disk: err=%v", err)
	}
	mf2, _ := memory.Load(memDir)
	if _, found := mf2.Get(m.ID); found {
		t.Error("manifest entry not pruned")
	}
}

func TestSlashMemoryForget_AmbiguousPrefixListsMatches(t *testing.T) {
	dataDir := t.TempDir()
	memDir := filepath.Join(dataDir, "memories")
	if err := os.MkdirAll(memDir, 0755); err != nil {
		t.Fatal(err)
	}
	engine := tfidf.NewEngine()
	vocab := memory.NewVocabSnapshot(engine)
	writeTestMemory(t, memDir, "First", vocab)
	writeTestMemory(t, memDir, "Second", vocab)

	p := paths{dataDir: dataDir, memoryDir: memDir}
	var buf bytes.Buffer
	// "mem_" matches both files but isn't a full ID.
	if err := slashMemoryForget(&buf, p, engine, "mem_"); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(buf.String(), "matches 2 memories") {
		t.Errorf("expected ambiguity message, got %q", buf.String())
	}
}

func TestSlashMemoryForget_MissingIDReportsCleanly(t *testing.T) {
	dataDir := t.TempDir()
	memDir := filepath.Join(dataDir, "memories")
	_ = os.MkdirAll(memDir, 0755)
	engine := tfidf.NewEngine()

	p := paths{dataDir: dataDir, memoryDir: memDir}
	var buf bytes.Buffer
	if err := slashMemoryForget(&buf, p, engine, "mem_does_not_exist"); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(buf.String(), "No memory matches") {
		t.Errorf("expected no-match message, got %q", buf.String())
	}
}
