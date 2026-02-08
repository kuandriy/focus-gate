package text

import "testing"

func TestStem(t *testing.T) {
	tests := []struct {
		input string
		want  string
	}{
		// Short words unchanged
		{"the", "the"},
		{"go", "go"},
		{"api", "api"},

		// Plurals: -ies → -y
		{"dependencies", "dependency"},
		{"queries", "query"},

		// Plurals: -es → strip (but not when preceded by 's')
		{"matches", "match"},
		{"processes", "processe"}, // preceded by 's', -es skipped, -s stripped
		{"classes", "classe"},     // preceded by 's', -es skipped, -s stripped

		// Plurals: -s → strip (not -ss)
		{"containers", "container"},
		{"tokens", "token"},
		{"less", "less"}, // -ss: don't strip

		// Derivational: -ization
		{"containerization", "container"},
		{"optimization", "optim"}, // -tion

		// Derivational: -ment
		{"deployment", "deploy"},
		{"management", "manage"},

		// Derivational: -ness
		{"readiness", "readi"},

		// Derivational: -ing
		{"processing", "process"},
		{"running", "runn"},

		// Derivational: -able/-ible
		{"configurable", "configur"},
		{"accessible", "access"},

		// Derivational: -ly
		{"quickly", "quick"},

		// "er" NOT stripped — root preservation
		{"server", "server"},
		{"container", "container"},
		{"computer", "computer"},
		{"docker", "docker"},
		{"water", "water"},

		// Both passes: plurals then derivational
		{"authentications", "authentica"}, // -s then -tion
	}

	for _, tt := range tests {
		got := Stem(tt.input)
		if got != tt.want {
			t.Errorf("Stem(%q) = %q, want %q", tt.input, got, tt.want)
		}
	}
}
