package text

// TruncateRunes returns s shortened to at most n runes, appending "…"
// when truncation actually happened. Callers that prefer a "..." ASCII
// suffix should use TruncateRunesWithSuffix.
//
// Unlike `s[:n]`, this routine never splits a multi-byte UTF-8 sequence
// — useful for any path that emits user-visible text (slash command
// output, log lines, surface blocks). n ≤ 0 returns "".
func TruncateRunes(s string, n int) string {
	return TruncateRunesWithSuffix(s, n, "…")
}

// TruncateRunesWithSuffix is the configurable form. The suffix counts
// against n: a 5-rune limit with a 1-rune ellipsis yields up to 4 runes
// of payload + 1 of suffix when truncation fires. If n is too small to
// fit the suffix, the suffix is dropped and the payload is hard-cut.
func TruncateRunesWithSuffix(s string, n int, suffix string) string {
	if n <= 0 {
		return ""
	}
	runes := []rune(s)
	if len(runes) <= n {
		return s
	}
	suffixLen := len([]rune(suffix))
	if suffixLen >= n {
		return string(runes[:n])
	}
	return string(runes[:n-suffixLen]) + suffix
}
