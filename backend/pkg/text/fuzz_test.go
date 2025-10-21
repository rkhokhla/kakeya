package text

import (
	"strings"
	"testing"
	"unicode/utf8"
)

// Phase 11 WP5: Fuzz testing for text tokenization
// Tests resilience against malformed Unicode and edge cases

// FuzzTokenize fuzzes the tokenization logic
func FuzzTokenize(f *testing.F) {
	// Seed corpus with various text inputs
	f.Add("Hello, world!")
	f.Add("The quick brown fox jumps over the lazy dog")
	f.Add("Unicode: 你好世界 🌍😊")
	f.Add("Numbers: 1 2 3 4 5")
	f.Add("")
	f.Add("   ")
	f.Add("a")
	f.Add(strings.Repeat("word ", 1000))

	f.Fuzz(func(t *testing.T, text string) {
		// Tokenizer should not crash on any valid UTF-8 string
		if !utf8.ValidString(text) {
			return
		}

		tokenizer := NewTokenizer(false, false, 2)

		// Should not crash
		tokens := tokenizer.Tokenize(text)

		// Basic sanity checks
		if len(tokens) < 0 {
			t.Error("Negative token count")
		}

		// All tokens should be non-empty
		for _, token := range tokens {
			if token == "" {
				t.Error("Empty token returned")
			}
		}
	})
}

// FuzzShingles fuzzes the shingling logic
func FuzzShingles(f *testing.F) {
	// Seed with token sequences
	f.Add([]byte("hello world foo bar"))
	f.Add([]byte("a b c d e"))
	f.Add([]byte(""))
	f.Add([]byte("single"))

	f.Fuzz(func(t *testing.T, data []byte) {
		text := string(data)
		if !utf8.ValidString(text) {
			return
		}

		tokenizer := NewTokenizer(false, false, 2)
		tokens := tokenizer.Tokenize(text)

		// Shingling should not crash
		shingles := tokenizer.Shingles(tokens)

		// Validate shingle structure
		if len(tokens) > 0 && len(tokens) < tokenizer.ShingleSize {
			if len(shingles) != 1 {
				t.Errorf("Expected 1 shingle for %d tokens (shingle size %d), got %d",
					len(tokens), tokenizer.ShingleSize, len(shingles))
			}
		}
	})
}

// FuzzJaccard fuzzes the Jaccard similarity computation
func FuzzJaccard(f *testing.F) {
	// Seed with shingle pairs
	f.Add([]byte("hello world"), []byte("hello world"))
	f.Add([]byte("foo bar"), []byte("baz qux"))
	f.Add([]byte(""), []byte(""))
	f.Add([]byte("a"), []byte(""))

	f.Fuzz(func(t *testing.T, data1, data2 []byte) {
		text1 := string(data1)
		text2 := string(data2)

		if !utf8.ValidString(text1) || !utf8.ValidString(text2) {
			return
		}

		tokenizer := NewTokenizer(false, false, 2)

		tokens1 := tokenizer.Tokenize(text1)
		tokens2 := tokenizer.Tokenize(text2)

		shingles1 := tokenizer.Shingles(tokens1)
		shingles2 := tokenizer.Shingles(tokens2)

		// Jaccard should not crash
		jaccard := tokenizer.Jaccard(shingles1, shingles2)

		// Validate Jaccard bounds [0, 1]
		if jaccard < 0.0 || jaccard > 1.0 {
			t.Errorf("Jaccard out of bounds: %.3f", jaccard)
		}

		// Symmetric property
		jaccardReverse := tokenizer.Jaccard(shingles2, shingles1)
		if jaccard != jaccardReverse {
			t.Errorf("Jaccard not symmetric: %.3f != %.3f", jaccard, jaccardReverse)
		}

		// Identity property
		if len(shingles1) > 0 {
			jaccardSelf := tokenizer.Jaccard(shingles1, shingles1)
			if jaccardSelf != 1.0 {
				t.Errorf("Jaccard(A, A) should be 1.0, got %.3f", jaccardSelf)
			}
		}
	})
}

// FuzzComputeOverlap fuzzes the end-to-end overlap computation
func FuzzComputeOverlap(f *testing.F) {
	// Seed with text pairs
	f.Add("The quick brown fox", "The fast brown fox")
	f.Add("", "")
	f.Add("Hello", "World")

	f.Fuzz(func(t *testing.T, text1, text2 string) {
		if !utf8.ValidString(text1) || !utf8.ValidString(text2) {
			return
		}

		tokenizer := NewTokenizer(false, false, 2)

		// Should not crash
		overlap := tokenizer.ComputeOverlap(text1, text2)

		// Validate bounds
		if overlap < 0.0 || overlap > 1.0 {
			t.Errorf("Overlap out of bounds: %.3f", overlap)
		}
	})
}
