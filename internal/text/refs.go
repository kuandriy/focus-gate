package text

import (
	"os"
	"path/filepath"
	"regexp"
	"strings"
)

// Known code/config file extensions. Kept broad enough to cover common
// developer workflows without matching prose fragments.
var codeExts = map[string]bool{
	"go": true, "ts": true, "js": true, "tsx": true, "jsx": true,
	"py": true, "java": true, "rs": true, "rb": true, "cpp": true,
	"c": true, "h": true, "hpp": true, "cs": true, "swift": true,
	"kt": true, "scala": true, "php": true, "vue": true, "svelte": true,
	"yaml": true, "yml": true, "json": true, "toml": true, "xml": true,
	"sql": true, "sh": true, "bash": true, "zsh": true,
	"css": true, "scss": true, "less": true, "html": true,
	"md": true, "proto": true, "graphql": true, "tf": true,
	"mod": true, "sum": true, "lock": true, "env": true, "cfg": true,
}

// pathPattern matches file-path-like strings that contain at least one /
// and end with .ext. Captures the path portion.
var pathPattern = regexp.MustCompile(`(?:^|[\s` + "`" + `"'(])` +
	`((?:\.{0,2}/)?` + // optional ./ or ../ prefix
	`(?:[\w.@-]+/)+` + // one or more directory components
	`[\w.@-]+` + // filename
	`\.([a-zA-Z0-9]{1,10}))` + // .extension
	`(?:$|[\s` + "`" + `"'),;:\]])`)

// backtickFilePattern matches backtick-wrapped filenames (no path required).
// E.g. `middleware.go`, `schema.sql`
var backtickFilePattern = regexp.MustCompile("`" + `([\w.-]+\.([a-zA-Z0-9]{1,10}))` + "`")

// domainSegments filters out URL-like and Go module paths.
var domainSuffixes = []string{".com/", ".org/", ".io/", ".net/", ".dev/", ".co/"}

// ExtractFilePaths extracts file path references from prompt text.
// Returns deduplicated paths, capped at maxRefs. Filters out URLs and
// Go module import paths.
func ExtractFilePaths(text string, maxRefs int) []string {
	if text == "" || maxRefs <= 0 {
		return nil
	}

	seen := make(map[string]bool)
	var refs []string

	addRef := func(path, ext string) {
		if !codeExts[strings.ToLower(ext)] {
			return
		}
		// Filter URLs
		if strings.Contains(path, "://") {
			return
		}
		// Filter domain-like paths (Go imports, URLs without protocol)
		lower := strings.ToLower(path)
		for _, suffix := range domainSuffixes {
			if strings.Contains(lower, suffix) {
				return
			}
		}
		if !seen[path] {
			seen[path] = true
			refs = append(refs, path)
		}
	}

	// Match paths with directory separators
	for _, match := range pathPattern.FindAllStringSubmatch(text, -1) {
		if len(match) >= 3 {
			addRef(match[1], match[2])
		}
	}

	// Match backtick-wrapped filenames (no path separator required)
	for _, match := range backtickFilePattern.FindAllStringSubmatch(text, -1) {
		if len(match) >= 3 {
			addRef(match[1], match[2])
		}
	}

	if len(refs) > maxRefs {
		refs = refs[:maxRefs]
	}
	return refs
}

// FilterExistingPaths returns only those paths that exist on disk relative to
// baseDir. An empty baseDir disables validation and the input is returned
// unchanged — callers that can't supply a working directory (tests, CI dry
// runs) still get the full set. Relative paths are resolved against baseDir;
// absolute paths are used as-is.
//
// This is the cheap antidote to rhetorical mentions ("store it in /var/log")
// accumulating as phantom refs. The cost is one os.Stat per path at
// classification time, which is negligible compared to tokenize+vectorize.
func FilterExistingPaths(baseDir string, paths []string) []string {
	if baseDir == "" || len(paths) == 0 {
		return paths
	}
	out := make([]string, 0, len(paths))
	for _, p := range paths {
		resolved := p
		if !filepath.IsAbs(p) {
			resolved = filepath.Join(baseDir, p)
		}
		if _, err := os.Stat(resolved); err == nil {
			out = append(out, p)
		}
	}
	return out
}
