package text

import (
	"regexp"
	"strings"
	"unicode"
)

var stopWords = map[string]bool{
	"a": true, "an": true, "the": true, "and": true, "or": true, "but": true,
	"in": true, "on": true, "at": true, "to": true, "for": true, "of": true,
	"with": true, "by": true, "from": true, "is": true, "it": true, "as": true,
	"be": true, "was": true, "are": true, "been": true, "being": true,
	"have": true, "has": true, "had": true, "do": true, "does": true, "did": true,
	"will": true, "would": true, "could": true, "should": true, "may": true,
	"might": true, "can": true, "shall": true, "must": true, "this": true,
	"that": true, "these": true, "those": true, "i": true, "me": true, "my": true,
	"we": true, "our": true, "you": true, "your": true, "he": true, "she": true,
	"his": true, "her": true, "they": true, "them": true, "their": true,
	"what": true, "which": true, "who": true, "when": true, "where": true,
	"how": true, "why": true, "not": true, "no": true, "so": true, "if": true,
	"then": true, "than": true, "too": true, "very": true, "just": true,
	"about": true, "also": true, "into": true, "each": true, "all": true,
	"any": true, "some": true, "more": true, "most": true, "other": true,
	"up": true, "out": true, "its": true, "only": true, "own": true, "same": true,
	"there": true, "here": true, "am": true, "were": true, "while": true,
	"during": true, "before": true, "after": true, "above": true, "below": true,
	"between": true, "through": true, "again": true, "further": true, "once": true,
	"both": true, "such": true, "don": true, "didn": true, "doesn": true,
	"won": true, "isn": true, "aren": true, "wasn": true, "weren": true,
	"let": true, "need": true, "want": true, "like": true, "make": true,
	"think": true, "know": true, "see": true, "get": true, "got": true,
	"go": true, "going": true, "one": true, "two": true, "first": true,
	"new": true, "well": true, "now": true, "way": true, "even": true,
	"back": true, "much": true, "because": true, "thing": true, "things": true,
	"still": true, "us": true, "really": true, "right": true, "re": true,
	"ve": true, "ll": true, "said": true, "say": true, "use": true, "used": true,
}

// tagPattern matches XML-style tags from IDE context injection.
var tagPattern = regexp.MustCompile(`<[a-z_-]+>[\s\S]*?</[a-z_-]+>`)

// Tokenize converts raw text into stemmed, filtered tokens.
// It lowercases, strips non-alphanumeric characters, stems each token,
// and removes stop words and single-character tokens.
func Tokenize(text string) []string {
	if text == "" {
		return nil
	}

	lower := strings.ToLower(text)

	// Split on boundaries, keeping hyphens and underscores within tokens.
	// This prevents compound-word fragments from false-stemming
	// (e.g. "session-expiry" stays whole instead of "session" → "ses" via -sion).
	raw := strings.FieldsFunc(lower, func(r rune) bool {
		return !unicode.IsLetter(r) && !unicode.IsDigit(r) && r != '-' && r != '_'
	})

	var tokens []string
	for _, t := range raw {
		t = Stem(t)
		if len(t) > 1 && !stopWords[t] {
			tokens = append(tokens, t)
		}
	}
	if len(tokens) == 0 {
		return nil
	}
	return tokens
}

// CleanPrompt strips IDE and system tags from raw prompt text.
func CleanPrompt(raw string) string {
	return strings.TrimSpace(tagPattern.ReplaceAllString(raw, ""))
}

// TermFrequency computes normalized term frequencies for a token list.
func TermFrequency(tokens []string) map[string]float64 {
	tf := make(map[string]float64, len(tokens))
	for _, t := range tokens {
		tf[t]++
	}
	n := float64(len(tokens))
	if n == 0 {
		n = 1
	}
	for k := range tf {
		tf[k] /= n
	}
	return tf
}
