package text

import "testing"

func TestTruncateRunes_NoOpWhenWithinLimit(t *testing.T) {
	if got := TruncateRunes("hello", 10); got != "hello" {
		t.Errorf("got %q, want hello", got)
	}
}

func TestTruncateRunes_ShortensAndAppendsEllipsis(t *testing.T) {
	got := TruncateRunes("hello world", 7)
	if len([]rune(got)) != 7 {
		t.Errorf("got %q (%d runes), want 7", got, len([]rune(got)))
	}
	if got[len(got)-len("…"):] != "…" {
		t.Errorf("got %q, expected to end with ellipsis", got)
	}
}

func TestTruncateRunes_DoesNotSplitMultibyte(t *testing.T) {
	// String with combining accents: "naïve résumé" — multi-byte runes.
	in := "naïve résumé"
	got := TruncateRunes(in, 6)
	// Result must be valid UTF-8.
	for _, r := range got {
		if r == '�' {
			t.Errorf("invalid UTF-8 produced: %q", got)
		}
	}
	if len([]rune(got)) > 6 {
		t.Errorf("rune limit exceeded: %d > 6", len([]rune(got)))
	}
}

func TestTruncateRunes_ZeroOrNegativeReturnsEmpty(t *testing.T) {
	for _, n := range []int{0, -1, -100} {
		if got := TruncateRunes("hello", n); got != "" {
			t.Errorf("n=%d: got %q, want empty", n, got)
		}
	}
}

func TestTruncateRunesWithSuffix_HardCutWhenSuffixTooLong(t *testing.T) {
	// 3-rune limit but a 4-rune suffix → suffix dropped, payload cut.
	got := TruncateRunesWithSuffix("hello world", 3, "....")
	if got != "hel" {
		t.Errorf("got %q, want hel", got)
	}
}

func TestTruncateRunesWithSuffix_ASCIIDots(t *testing.T) {
	got := TruncateRunesWithSuffix("hello world", 8, "...")
	// 5 payload runes + 3 suffix = 8 total
	if got != "hello..." {
		t.Errorf("got %q, want hello...", got)
	}
}
