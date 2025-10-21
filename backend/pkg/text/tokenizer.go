package text

import (
	"strings"
	"unicode"
)

// Phase 11 WP2: Tokenized RAG Overlap
// Word-level Jaccard similarity with n-gram shingles for robust grounding
// vs punctuation/formatting drift

// Tokenizer handles text tokenization, shingling, and Jaccard similarity
type Tokenizer struct {
	StopWords   map[string]bool
	Stemming    bool
	ShingleSize int // Default: 2 (bigrams)
}

// DefaultStopWords returns common English stop words
func DefaultStopWords() map[string]bool {
	words := []string{
		"a", "an", "and", "are", "as", "at", "be", "by", "for",
		"from", "has", "he", "in", "is", "it", "its", "of", "on",
		"that", "the", "to", "was", "will", "with",
	}
	stopWords := make(map[string]bool, len(words))
	for _, w := range words {
		stopWords[w] = true
	}
	return stopWords
}

// NewTokenizer creates a new tokenizer with specified configuration
func NewTokenizer(useStopWords, useStemming bool, shingleSize int) *Tokenizer {
	var stopWords map[string]bool
	if useStopWords {
		stopWords = DefaultStopWords()
	}

	if shingleSize <= 0 {
		shingleSize = 2 // Default to bigrams
	}

	return &Tokenizer{
		StopWords:   stopWords,
		Stemming:    useStemming,
		ShingleSize: shingleSize,
	}
}

// Tokenize splits text into lowercase words (Unicode-aware)
func (t *Tokenizer) Tokenize(text string) []string {
	var tokens []string
	var currentWord strings.Builder

	for _, r := range text {
		if unicode.IsLetter(r) || unicode.IsNumber(r) {
			// Keep letters and numbers
			currentWord.WriteRune(unicode.ToLower(r))
		} else {
			// Delimiter found
			if currentWord.Len() > 0 {
				word := currentWord.String()

				// Apply stemming if enabled
				if t.Stemming {
					word = t.stem(word)
				}

				// Filter stop words if enabled
				if t.StopWords == nil || !t.StopWords[word] {
					tokens = append(tokens, word)
				}

				currentWord.Reset()
			}
		}
	}

	// Add last word if present
	if currentWord.Len() > 0 {
		word := currentWord.String()
		if t.Stemming {
			word = t.stem(word)
		}
		if t.StopWords == nil || !t.StopWords[word] {
			tokens = append(tokens, word)
		}
	}

	return tokens
}

// Shingles creates n-gram shingles from tokens
func (t *Tokenizer) Shingles(tokens []string) []string {
	if len(tokens) < t.ShingleSize {
		// If fewer tokens than shingle size, return single shingle
		if len(tokens) == 0 {
			return []string{}
		}
		return []string{strings.Join(tokens, " ")}
	}

	shingles := make([]string, 0, len(tokens)-t.ShingleSize+1)
	for i := 0; i <= len(tokens)-t.ShingleSize; i++ {
		shingle := strings.Join(tokens[i:i+t.ShingleSize], " ")
		shingles = append(shingles, shingle)
	}

	return shingles
}

// Jaccard computes Jaccard similarity between two shingle sets
// Returns value in [0, 1] where 1 = identical, 0 = no overlap
func (t *Tokenizer) Jaccard(a, b []string) float64 {
	if len(a) == 0 && len(b) == 0 {
		return 1.0 // Both empty = perfect match
	}
	if len(a) == 0 || len(b) == 0 {
		return 0.0 // One empty, one not = no overlap
	}

	// Build set for a
	setA := make(map[string]bool, len(a))
	for _, shingle := range a {
		setA[shingle] = true
	}

	// Count intersection and build union
	intersection := 0
	setUnion := make(map[string]bool, len(a)+len(b))
	for shingle := range setA {
		setUnion[shingle] = true
	}

	for _, shingle := range b {
		if setA[shingle] {
			intersection++
		}
		setUnion[shingle] = true
	}

	// Jaccard = |intersection| / |union|
	return float64(intersection) / float64(len(setUnion))
}

// stem applies simple suffix-stripping stemming (Porter-like)
// This is a simplified version; production would use snowball stemmer
func (t *Tokenizer) stem(word string) string {
	if len(word) < 4 {
		return word // Don't stem very short words
	}

	// Simple suffix rules (English)
	suffixes := []string{"ing", "ed", "ly", "es", "s"}
	for _, suffix := range suffixes {
		if strings.HasSuffix(word, suffix) {
			stemmed := word[:len(word)-len(suffix)]
			if len(stemmed) >= 2 {
				return stemmed
			}
		}
	}

	return word
}

// ComputeOverlap is a convenience method that tokenizes, shingles, and computes Jaccard
func (t *Tokenizer) ComputeOverlap(text1, text2 string) float64 {
	tokens1 := t.Tokenize(text1)
	tokens2 := t.Tokenize(text2)

	shingles1 := t.Shingles(tokens1)
	shingles2 := t.Shingles(tokens2)

	return t.Jaccard(shingles1, shingles2)
}
