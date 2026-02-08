package text

import (
	"reflect"
	"testing"
)

func TestTokenize(t *testing.T) {
	tests := []struct {
		name  string
		input string
		want  []string
	}{
		{
			name:  "empty",
			input: "",
			want:  nil,
		},
		{
			name:  "stop words only",
			input: "the and or but in on at",
			want:  nil,
		},
		{
			name:  "basic tokens",
			input: "add JWT authentication to the API",
			want:  []string{"add", "jwt", "authentica", "api"},
		},
		{
			name:  "punctuation stripped, hyphens kept",
			input: "fix: the session-expiry bug!",
			want:  []string{"fix", "session-expiry", "bug"},
		},
		{
			name:  "mixed case",
			input: "Create UserProfile Component",
			want:  []string{"create", "userprofile", "component"},
		},
		{
			name:  "numbers preserved",
			input: "add base64 encoding to v2 api",
			want:  []string{"add", "base64", "encod", "v2", "api"},
		},
		{
			name:  "single chars filtered",
			input: "a b c real token",
			want:  []string{"real", "token"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := Tokenize(tt.input)
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("Tokenize(%q)\n  got  %v\n  want %v", tt.input, got, tt.want)
			}
		})
	}
}

func TestCleanPrompt(t *testing.T) {
	tests := []struct {
		name  string
		input string
		want  string
	}{
		{
			name:  "no tags",
			input: "plain prompt text",
			want:  "plain prompt text",
		},
		{
			name:  "ide selection stripped",
			input: "fix this <ide_selection>some code here</ide_selection> please",
			want:  "fix this  please",
		},
		{
			name:  "system reminder stripped",
			input: "<system-reminder>hook output</system-reminder>actual prompt",
			want:  "actual prompt",
		},
		{
			name:  "multiple tags stripped",
			input: "<ide_opened_file>foo.js</ide_opened_file>fix bug<system-reminder>x</system-reminder>",
			want:  "fix bug",
		},
		{
			name:  "whitespace trimmed",
			input: "  hello world  ",
			want:  "hello world",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := CleanPrompt(tt.input)
			if got != tt.want {
				t.Errorf("CleanPrompt(%q) = %q, want %q", tt.input, got, tt.want)
			}
		})
	}
}

func TestTermFrequency(t *testing.T) {
	tokens := []string{"auth", "token", "auth", "jwt"}
	tf := TermFrequency(tokens)

	if tf["auth"] != 0.5 {
		t.Errorf("tf[auth] = %f, want 0.5", tf["auth"])
	}
	if tf["token"] != 0.25 {
		t.Errorf("tf[token] = %f, want 0.25", tf["token"])
	}
	if tf["jwt"] != 0.25 {
		t.Errorf("tf[jwt] = %f, want 0.25", tf["jwt"])
	}
}

func TestTermFrequencyEmpty(t *testing.T) {
	tf := TermFrequency(nil)
	if len(tf) != 0 {
		t.Errorf("TermFrequency(nil) should be empty, got %v", tf)
	}
}
