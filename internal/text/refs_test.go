package text

import (
	"os"
	"path/filepath"
	"testing"
)

func TestExtractFilePaths_RelativePaths(t *testing.T) {
	tests := []struct {
		name  string
		input string
		want  []string
	}{
		{
			name:  "simple relative path",
			input: "fix the bug in src/auth/middleware.go",
			want:  []string{"src/auth/middleware.go"},
		},
		{
			name:  "multiple paths",
			input: "compare src/auth/jwt.go and src/auth/session.go",
			want:  []string{"src/auth/jwt.go", "src/auth/session.go"},
		},
		{
			name:  "dot-slash prefix",
			input: "look at ./cmd/main.go",
			want:  []string{"./cmd/main.go"},
		},
		{
			name:  "dotdot-slash prefix",
			input: "look at ../utils/helper.ts",
			want:  []string{"../utils/helper.ts"},
		},
		{
			name:  "internal Go path",
			input: "edit internal/gate/gate.go please",
			want:  []string{"internal/gate/gate.go"},
		},
		{
			name:  "deep nesting",
			input: "check pkg/api/v2/handlers/auth.go",
			want:  []string{"pkg/api/v2/handlers/auth.go"},
		},
		{
			name:  "various extensions",
			input: "update api/routes.ts and styles/main.css and db/schema.sql",
			want:  []string{"api/routes.ts", "styles/main.css", "db/schema.sql"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := ExtractFilePaths(tt.input, 10)
			if len(got) != len(tt.want) {
				t.Fatalf("got %v, want %v", got, tt.want)
			}
			for i := range got {
				if got[i] != tt.want[i] {
					t.Errorf("got[%d] = %q, want %q", i, got[i], tt.want[i])
				}
			}
		})
	}
}

func TestExtractFilePaths_BacktickWrapped(t *testing.T) {
	got := ExtractFilePaths("update `middleware.go` and `schema.sql`", 10)
	if len(got) != 2 {
		t.Fatalf("expected 2 refs, got %v", got)
	}
	if got[0] != "middleware.go" {
		t.Errorf("got[0] = %q, want middleware.go", got[0])
	}
	if got[1] != "schema.sql" {
		t.Errorf("got[1] = %q, want schema.sql", got[1])
	}
}

func TestExtractFilePaths_FilterURLs(t *testing.T) {
	// Go module imports should be filtered
	got := ExtractFilePaths("import github.com/user/repo/pkg.go", 10)
	for _, ref := range got {
		if ref == "github.com/user/repo/pkg.go" {
			t.Errorf("should filter Go module path, got %q", ref)
		}
	}
}

func TestExtractFilePaths_FilterDomainPaths(t *testing.T) {
	tests := []string{
		"see golang.org/x/tools/go.go",
		"from example.io/api/handler.go",
		"use pkg.dev/lib/util.go",
	}
	for _, input := range tests {
		got := ExtractFilePaths(input, 10)
		if len(got) > 0 {
			t.Errorf("input %q should produce no refs, got %v", input, got)
		}
	}
}

func TestExtractFilePaths_Deduplication(t *testing.T) {
	got := ExtractFilePaths("check src/auth/jwt.go and also src/auth/jwt.go again", 10)
	if len(got) != 1 {
		t.Errorf("expected 1 deduplicated ref, got %v", got)
	}
}

func TestExtractFilePaths_MaxRefs(t *testing.T) {
	input := "files: a/b.go c/d.go e/f.go g/h.go i/j.go k/l.go"
	got := ExtractFilePaths(input, 3)
	if len(got) > 3 {
		t.Errorf("expected max 3 refs, got %d", len(got))
	}
}

func TestExtractFilePaths_NoMatch(t *testing.T) {
	tests := []string{
		"add authentication to the API",
		"fix the database migration",
		"",
		"just a plain sentence with no paths",
	}
	for _, input := range tests {
		got := ExtractFilePaths(input, 10)
		if len(got) != 0 {
			t.Errorf("input %q should produce no refs, got %v", input, got)
		}
	}
}

func TestExtractFilePaths_UnknownExtension(t *testing.T) {
	got := ExtractFilePaths("see path/to/file.xyz123", 10)
	if len(got) != 0 {
		t.Errorf("unknown extension should be filtered, got %v", got)
	}
}

func TestExtractFilePaths_QuotedPaths(t *testing.T) {
	got := ExtractFilePaths(`edit "src/auth/middleware.go" please`, 10)
	if len(got) != 1 || got[0] != "src/auth/middleware.go" {
		t.Errorf("expected src/auth/middleware.go, got %v", got)
	}
}

func TestExtractFilePaths_ZeroMaxRefs(t *testing.T) {
	got := ExtractFilePaths("src/auth/jwt.go", 0)
	if got != nil {
		t.Errorf("maxRefs=0 should return nil, got %v", got)
	}
}

func TestFilterExistingPaths_EmptyBaseDirIsNoOp(t *testing.T) {
	in := []string{"a.go", "b/c.go"}
	got := FilterExistingPaths("", in)
	if len(got) != len(in) {
		t.Errorf("empty baseDir should pass paths through, got %v", got)
	}
}

func TestFilterExistingPaths_DropsMissingFiles(t *testing.T) {
	dir := t.TempDir()
	// Create one real file; the other is rhetorical.
	if err := os.MkdirAll(filepath.Join(dir, "src"), 0755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "src", "real.go"), []byte(""), 0644); err != nil {
		t.Fatal(err)
	}

	paths := []string{"src/real.go", "var/log/phantom.json"}
	got := FilterExistingPaths(dir, paths)

	if len(got) != 1 || got[0] != "src/real.go" {
		t.Errorf("expected only src/real.go to survive, got %v", got)
	}
}

func TestFilterExistingPaths_AbsolutePathsHandledIndependentlyOfBaseDir(t *testing.T) {
	// An absolute path should be checked as-is, ignoring baseDir.
	dir := t.TempDir()
	real := filepath.Join(dir, "real.go")
	if err := os.WriteFile(real, []byte(""), 0644); err != nil {
		t.Fatal(err)
	}

	// baseDir is unrelated (TempDir ensures uniqueness). Absolute paths
	// should still resolve correctly.
	unrelatedBase := t.TempDir()
	missing := filepath.Join(dir, "phantom.go")

	got := FilterExistingPaths(unrelatedBase, []string{real, missing})
	if len(got) != 1 || got[0] != real {
		t.Errorf("expected only the real absolute path to survive, got %v", got)
	}
}
