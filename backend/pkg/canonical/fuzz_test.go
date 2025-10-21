package canonical

import (
	"encoding/json"
	"testing"
)

// Phase 11 WP5: Fuzz testing for canonical JSON and signing
// Tests resilience against malformed inputs

// FuzzCanonicalJSON fuzzes canonical JSON generation
func FuzzCanonicalJSON(f *testing.F) {
	// Seed corpus with various inputs
	f.Add(float64(1.234567890123))
	f.Add(float64(0.0))
	f.Add(float64(-999.999999999))
	f.Add(float64(1e10))

	f.Fuzz(func(t *testing.T, value float64) {
		// Should not crash on any float64 value
		_ = F9(value)

		// Round9 should be stable
		rounded := Round9(value)
		roundedTwice := Round9(rounded)

		// Idempotency check
		if rounded != roundedTwice {
			t.Errorf("Round9 not idempotent: %.9f != %.9f", rounded, roundedTwice)
		}
	})
}

// FuzzJSONMarshaling fuzzes JSON marshaling/unmarshaling
func FuzzJSONMarshaling(f *testing.F) {
	// Seed with various JSON structures
	f.Add([]byte(`{"a":1,"b":2}`))
	f.Add([]byte(`{"nested":{"key":"value"}}`))
	f.Add([]byte(`[]`))
	f.Add([]byte(`null`))

	f.Fuzz(func(t *testing.T, data []byte) {
		var obj interface{}

		// Should not crash on any JSON input
		err := json.Unmarshal(data, &obj)
		if err != nil {
			return
		}

		// Re-marshaling should not crash
		_, _ = json.Marshal(obj)
	})
}

// FuzzSigningSubset fuzzes signature subset extraction
func FuzzSigningSubset(f *testing.F) {
	// Seed with various map structures
	f.Add(`{"pcs_id":"test","D_hat":1.5,"coh_star":0.7}`)
	f.Add(`{"invalid_field":"value"}`)
	f.Add(`{}`)

	f.Fuzz(func(t *testing.T, jsonStr string) {
		var obj map[string]interface{}

		// Should not crash on any valid JSON map
		err := json.Unmarshal([]byte(jsonStr), &obj)
		if err != nil {
			return
		}

		// Extract subset (should not crash)
		subset := make(map[string]interface{})
		sigFields := []string{"pcs_id", "merkle_root", "epoch", "shard_id", "D_hat", "coh_star", "r", "budget"}
		for _, field := range sigFields {
			if val, ok := obj[field]; ok {
				subset[field] = val
			}
		}

		// Marshal subset (should not crash)
		_, _ = json.Marshal(subset)
	})
}
