package text

import "strings"

// Derivational suffixes ordered longest first for single-pass matching.
// "er" is intentionally excluded — too many English root words end in "er"
// (container, server, computer, docker) causing false conflation.
var derivational = []string{
	"ization", "ising", "izing", "ional",
	"ment", "ness", "less",
	"able", "ible", "tion", "sion", "ling", "ally",
	"ful", "ous", "ive", "ing", "ed", "ly",
}

// Stem applies a lightweight two-pass suffix stemmer.
//
// Pass 1 strips plurals (s/es/ies).
// Pass 2 strips one derivational suffix (longest match first).
//
// This produces consistent stems: "containerization" and "containers" both → "container".
func Stem(word string) string {
	if len(word) < 4 {
		return word
	}

	// Pass 1: remove plurals
	if len(word) > 4 && strings.HasSuffix(word, "ies") {
		word = word[:len(word)-3] + "y"
	} else if len(word) > 4 && strings.HasSuffix(word, "es") && word[len(word)-3] != 's' {
		word = word[:len(word)-2]
	} else if len(word) > 3 && word[len(word)-1] == 's' && word[len(word)-2] != 's' {
		word = word[:len(word)-1]
	}

	// Pass 2: remove one derivational suffix (longest match, single pass)
	for _, suf := range derivational {
		if len(word) > len(suf)+2 && strings.HasSuffix(word, suf) {
			return word[:len(word)-len(suf)]
		}
	}
	return word
}
