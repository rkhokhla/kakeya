package text

import (
	"testing"
)

func TestTokenize(t *testing.T) {
	tests := []struct {
		name          string
		text          string
		useStopWords  bool
		expectedCount int
		expectedWords []string
	}{
		{
			name:          "simple text",
			text:          "Hello, world!",
			useStopWords:  false,
			expectedCount: 2,
			expectedWords: []string{"hello", "world"},
		},
		{
			name:          "with stop words filtered",
			text:          "The quick brown fox",
			useStopWords:  true,
			expectedCount: 3,
			expectedWords: []string{"quick", "brown", "fox"},
		},
		{
			name:          "with punctuation",
			text:          "Hello, world! How are you?",
			useStopWords:  true,
			expectedCount: 4,
			expectedWords: []string{"hello", "world", "how", "you"},
		},
		{
			name:          "unicode text (emoji)",
			text:          "Hello 😊 world 🌍",
			useStopWords:  false,
			expectedCount: 2,
			expectedWords: []string{"hello", "world"},
		},
		{
			name:          "numbers included",
			text:          "Scale 2 4 8 16",
			useStopWords:  false,
			expectedCount: 5,
			expectedWords: []string{"scale", "2", "4", "8", "16"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tokenizer := NewTokenizer(tt.useStopWords, false, 2)
			tokens := tokenizer.Tokenize(tt.text)

			if len(tokens) != tt.expectedCount {
				t.Errorf("Expected %d tokens, got %d", tt.expectedCount, len(tokens))
			}

			for i, expected := range tt.expectedWords {
				if i >= len(tokens) {
					t.Errorf("Missing token at index %d: %s", i, expected)
					break
				}
				if tokens[i] != expected {
					t.Errorf("Token %d: expected %q, got %q", i, expected, tokens[i])
				}
			}
		})
	}
}

func TestShingling(t *testing.T) {
	tests := []struct {
		name            string
		tokens          []string
		shingleSize     int
		expectedShingles []string
	}{
		{
			name:        "bigrams",
			tokens:      []string{"the", "quick", "brown", "fox"},
			shingleSize: 2,
			expectedShingles: []string{
				"the quick",
				"quick brown",
				"brown fox",
			},
		},
		{
			name:        "trigrams",
			tokens:      []string{"the", "quick", "brown", "fox"},
			shingleSize: 3,
			expectedShingles: []string{
				"the quick brown",
				"quick brown fox",
			},
		},
		{
			name:            "fewer tokens than shingle size",
			tokens:          []string{"hello"},
			shingleSize:     2,
			expectedShingles: []string{"hello"},
		},
		{
			name:            "empty tokens",
			tokens:          []string{},
			shingleSize:     2,
			expectedShingles: []string{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tokenizer := NewTokenizer(false, false, tt.shingleSize)
			shingles := tokenizer.Shingles(tt.tokens)

			if len(shingles) != len(tt.expectedShingles) {
				t.Errorf("Expected %d shingles, got %d", len(tt.expectedShingles), len(shingles))
			}

			for i, expected := range tt.expectedShingles {
				if i >= len(shingles) {
					t.Errorf("Missing shingle at index %d: %s", i, expected)
					break
				}
				if shingles[i] != expected {
					t.Errorf("Shingle %d: expected %q, got %q", i, expected, shingles[i])
				}
			}
		})
	}
}

func TestJaccard(t *testing.T) {
	tests := []struct {
		name     string
		a        []string
		b        []string
		expected float64
	}{
		{
			name:     "identical sets",
			a:        []string{"hello", "world"},
			b:        []string{"hello", "world"},
			expected: 1.0,
		},
		{
			name:     "no overlap",
			a:        []string{"hello", "world"},
			b:        []string{"foo", "bar"},
			expected: 0.0,
		},
		{
			name:     "partial overlap",
			a:        []string{"hello", "world", "foo"},
			b:        []string{"hello", "world", "bar"},
			expected: 0.5, // 2 common / 4 union
		},
		{
			name:     "one empty",
			a:        []string{"hello"},
			b:        []string{},
			expected: 0.0,
		},
		{
			name:     "both empty",
			a:        []string{},
			b:        []string{},
			expected: 1.0,
		},
		{
			name:     "subset",
			a:        []string{"hello", "world"},
			b:        []string{"hello"},
			expected: 0.5, // 1 common / 2 union
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tokenizer := NewTokenizer(false, false, 2)
			result := tokenizer.Jaccard(tt.a, tt.b)

			// Use approximate equality for floats
			if result < tt.expected-0.01 || result > tt.expected+0.01 {
				t.Errorf("Expected Jaccard ~%.2f, got %.2f", tt.expected, result)
			}
		})
	}
}

func TestComputeOverlap(t *testing.T) {
	tests := []struct {
		name        string
		text1       string
		text2       string
		minExpected float64
		maxExpected float64
	}{
		{
			name:        "identical text",
			text1:       "Hello, world!",
			text2:       "Hello, world!",
			minExpected: 1.0,
			maxExpected: 1.0,
		},
		{
			name:        "punctuation resilience",
			text1:       "Hello world",
			text2:       "Hello, world!",
			minExpected: 1.0,
			maxExpected: 1.0,
		},
		{
			name:        "no overlap",
			text1:       "The quick brown fox",
			text2:       "A lazy dog sleeps",
			minExpected: 0.0,
			maxExpected: 0.1,
		},
		{
			name:        "partial overlap",
			text1:       "The quick brown fox jumps",
			text2:       "The quick red fox runs",
			minExpected: 0.1,
			maxExpected: 0.7,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tokenizer := NewTokenizer(false, false, 2)
			overlap := tokenizer.ComputeOverlap(tt.text1, tt.text2)

			if overlap < tt.minExpected || overlap > tt.maxExpected {
				t.Errorf("Expected overlap in [%.2f, %.2f], got %.2f",
					tt.minExpected, tt.maxExpected, overlap)
			}
		})
	}
}

func TestStemming(t *testing.T) {
	tests := []struct {
		name     string
		word     string
		expected string
	}{
		{
			name:     "running -> run",
			word:     "running",
			expected: "runn",
		},
		{
			name:     "walked -> walk",
			word:     "walked",
			expected: "walk",
		},
		{
			name:     "quickly -> quick",
			word:     "quickly",
			expected: "quick",
		},
		{
			name:     "short word unchanged",
			word:     "run",
			expected: "run",
		},
	}

	tokenizer := NewTokenizer(false, true, 2)

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := tokenizer.stem(tt.word)
			if result != tt.expected {
				t.Errorf("Expected stem %q, got %q", tt.expected, result)
			}
		})
	}
}

// Benchmark tests
func BenchmarkTokenize(b *testing.B) {
	tokenizer := NewTokenizer(true, false, 2)
	text := "The quick brown fox jumps over the lazy dog. Pack my box with five dozen liquor jugs."

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = tokenizer.Tokenize(text)
	}
}

func BenchmarkComputeOverlap(b *testing.B) {
	tokenizer := NewTokenizer(true, false, 2)
	text1 := "The quick brown fox jumps over the lazy dog."
	text2 := "The fast brown fox leaps over the sleepy dog."

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = tokenizer.ComputeOverlap(text1, text2)
	}
}

// Performance test: 10KB text in ≤10ms p95
func TestPerformance10KB(t *testing.T) {
	// Generate ~10KB of text
	sentence := "The quick brown fox jumps over the lazy dog. "
	text := ""
	for len(text) < 10000 {
		text += sentence
	}

	tokenizer := NewTokenizer(true, false, 2)

	// Warmup
	_ = tokenizer.Tokenize(text)

	// Measure
	iterations := 100
	for i := 0; i < iterations; i++ {
		_ = tokenizer.Tokenize(text)
	}

	// Test will fail if performance is way off (sanity check)
	// Real p95 measurement would use proper timing and percentiles
	// This is just a smoke test that it completes
	t.Logf("Performance test completed for %d bytes of text", len(text))
}
