package text

import (
	"reflect"
	"testing"
)

func TestTokenizeWithCorpus_DisabledIsPassthrough(t *testing.T) {
	corpus := map[string]int{"environ": 10}
	got := TokenizeWithCorpus("environ envaeron", corpus, CanonicalizeOpts{Enabled: false, MaxDistance: 2, MinWordLen: 5, MinEstablishedDF: 3})
	want := Tokenize("environ envaeron")
	if !reflect.DeepEqual(got, want) {
		t.Errorf("disabled should match Tokenize: got %v, want %v", got, want)
	}
}

func TestTokenizeWithCorpus_EmptyCorpusIsPassthrough(t *testing.T) {
	got := TokenizeWithCorpus("environ envaeron", nil, CanonicalizeOpts{Enabled: true, MaxDistance: 2, MinWordLen: 5, MinEstablishedDF: 3})
	want := Tokenize("environ envaeron")
	if !reflect.DeepEqual(got, want) {
		t.Errorf("nil corpus should match Tokenize: got %v, want %v", got, want)
	}
}

func TestTokenizeWithCorpus_EstablishedRewritesTypo(t *testing.T) {
	// "environ" has DF 5 — well above minEstablishedDF. "envaeron" is within
	// 2 edits. Expect the typo to canonicalise to "environ".
	corpus := map[string]int{"environ": 5}
	got := TokenizeWithCorpus("envaeron", corpus, CanonicalizeOpts{Enabled: true, MaxDistance: 2, MinWordLen: 5, MinEstablishedDF: 3})
	if len(got) != 1 || got[0] != "environ" {
		t.Errorf("expected typo rewritten to 'environ', got %v", got)
	}
}

func TestTokenizeWithCorpus_UnestablishedDoesNotRewrite(t *testing.T) {
	// "environ" appears only once (below minEstablishedDF=3) so the typo
	// must pass through as its own novel token.
	corpus := map[string]int{"environ": 1}
	got := TokenizeWithCorpus("envaeron", corpus, CanonicalizeOpts{Enabled: true, MaxDistance: 2, MinWordLen: 5, MinEstablishedDF: 3})
	if len(got) != 1 || got[0] == "environ" {
		t.Errorf("expected typo NOT rewritten (term below minEstablishedDF), got %v", got)
	}
}

func TestTokenizeWithCorpus_ShortWordsNeverRewritten(t *testing.T) {
	// Short token — would otherwise be at distance 2 from a corpus term,
	// but minWordLen=5 gates it out.
	corpus := map[string]int{"each": 10}
	got := TokenizeWithCorpus("auth", corpus, CanonicalizeOpts{Enabled: true, MaxDistance: 2, MinWordLen: 5, MinEstablishedDF: 3})
	if len(got) != 1 || got[0] != "auth" {
		t.Errorf("expected short token to pass through, got %v", got)
	}
}

func TestTokenizeWithCorpus_KnownTokenNoRewrite(t *testing.T) {
	// Token exactly matches an existing term; no scan needed, result is identical.
	corpus := map[string]int{"migrat": 10, "migrant": 2}
	got := TokenizeWithCorpus("migrat", corpus, CanonicalizeOpts{Enabled: true, MaxDistance: 2, MinWordLen: 5, MinEstablishedDF: 3})
	if len(got) != 1 || got[0] != "migrat" {
		t.Errorf("expected known token to pass through as-is, got %v", got)
	}
}

func TestTokenizeWithCorpus_TieBreaksAlphabetically(t *testing.T) {
	// Two established terms both one edit away from the typo. Expect the
	// alphabetically-first one to win so the behaviour is deterministic.
	corpus := map[string]int{"abcdef": 10, "abcdex": 10}
	got := TokenizeWithCorpus("abcdez", corpus, CanonicalizeOpts{Enabled: true, MaxDistance: 1, MinWordLen: 5, MinEstablishedDF: 3})
	if len(got) != 1 || got[0] != "abcdef" {
		t.Errorf("expected 'abcdef' (alphabetically first tie), got %v", got)
	}
}

func TestTokenizeWithCorpus_FurtherThanMaxDistanceNotRewritten(t *testing.T) {
	// Corpus has "environ" (DF=5). Input is 4 edits away — outside maxDistance=2.
	corpus := map[string]int{"environ": 5}
	got := TokenizeWithCorpus("quantum", corpus, CanonicalizeOpts{Enabled: true, MaxDistance: 2, MinWordLen: 5, MinEstablishedDF: 3})
	if len(got) != 1 || got[0] == "environ" {
		t.Errorf("expected unrelated word NOT rewritten, got %v", got)
	}
}
