package text

import "testing"

func TestLevenshteinCutoff_ExactMatch(t *testing.T) {
	if d := LevenshteinCutoff("environ", "environ", 2); d != 0 {
		t.Errorf("equal strings: got %d, want 0", d)
	}
}

func TestLevenshteinCutoff_SingleSubstitution(t *testing.T) {
	if d := LevenshteinCutoff("environ", "envoron", 2); d != 1 {
		t.Errorf("single substitution: got %d, want 1", d)
	}
}

func TestLevenshteinCutoff_SingleInsertion(t *testing.T) {
	if d := LevenshteinCutoff("environ", "environs", 2); d != 1 {
		t.Errorf("single insertion: got %d, want 1", d)
	}
}

func TestLevenshteinCutoff_SingleDeletion(t *testing.T) {
	if d := LevenshteinCutoff("environ", "enviro", 2); d != 1 {
		t.Errorf("single deletion: got %d, want 1", d)
	}
}

func TestLevenshteinCutoff_AboveCutoffReturnsCutoffPlusOne(t *testing.T) {
	// Distance is 3; cutoff is 1 → must return > cutoff (i.e. at least 2).
	d := LevenshteinCutoff("abcdef", "xyzdef", 1)
	if d <= 1 {
		t.Errorf("expected > cutoff for far strings, got %d", d)
	}
}

func TestLevenshteinCutoff_LengthDiffPrefilter(t *testing.T) {
	// Length differs by 5 → prefilter should short-circuit with cutoff+1.
	if d := LevenshteinCutoff("abc", "abcdefgh", 2); d != 3 {
		t.Errorf("length prefilter: got %d, want 3 (cutoff+1)", d)
	}
}

func TestLevenshteinCutoff_EmptyStrings(t *testing.T) {
	if d := LevenshteinCutoff("", "", 1); d != 0 {
		t.Errorf("both empty: got %d, want 0", d)
	}
	if d := LevenshteinCutoff("", "abc", 5); d != 3 {
		t.Errorf("one empty: got %d, want 3", d)
	}
	if d := LevenshteinCutoff("abc", "", 5); d != 3 {
		t.Errorf("other empty: got %d, want 3", d)
	}
}

func TestLevenshteinCutoff_UnicodeRunes(t *testing.T) {
	// Each emoji is a single rune; substitution should count as 1 edit.
	if d := LevenshteinCutoff("aébc", "aíbc", 2); d != 1 {
		t.Errorf("unicode substitution: got %d, want 1", d)
	}
}

func TestLevenshteinCutoff_RealTypos(t *testing.T) {
	// Sanity cases drawn from user-observed spelling drift.
	cases := []struct {
		a, b string
		want int
	}{
		{"environ", "envaeron", 2}, // insert + substitute: needs maxDistance=2
		{"difficult", "dificult", 1},
		{"authentication", "authentcation", 1},
		{"migrate", "migratee", 1},
	}
	for _, c := range cases {
		d := LevenshteinCutoff(c.a, c.b, 3)
		if d != c.want {
			t.Errorf("LevenshteinCutoff(%q,%q) = %d, want %d", c.a, c.b, d, c.want)
		}
	}
}
