package cache

import (
	"testing"
	"time"
)

func TestLRUWithTTL_BasicOperations(t *testing.T) {
	cache, err := NewLRUWithTTL[string, int](3, 0) // no TTL
	if err != nil {
		t.Fatalf("failed to create cache: %v", err)
	}
	defer cache.Close()

	// Test Set and Get
	cache.Set("key1", 42)
	if val, ok := cache.Get("key1"); !ok || val != 42 {
		t.Errorf("Get(key1) = (%v, %v), want (42, true)", val, ok)
	}

	// Test miss
	if _, ok := cache.Get("nonexistent"); ok {
		t.Error("Get(nonexistent) should return false")
	}

	// Test LRU eviction
	cache.Set("key2", 100)
	cache.Set("key3", 200)
	cache.Set("key4", 300) // should evict key1 (LRU)

	if _, ok := cache.Get("key1"); ok {
		t.Error("key1 should have been evicted")
	}

	if val, ok := cache.Get("key4"); !ok || val != 300 {
		t.Errorf("Get(key4) = (%v, %v), want (300, true)", val, ok)
	}
}

func TestLRUWithTTL_Expiration(t *testing.T) {
	cache, err := NewLRUWithTTL[string, string](10, 50*time.Millisecond)
	if err != nil {
		t.Fatalf("failed to create cache: %v", err)
	}
	defer cache.Close()

	cache.Set("key1", "value1")

	// Should be present immediately
	if _, ok := cache.Get("key1"); !ok {
		t.Error("key1 should be present before expiration")
	}

	// Wait for expiration
	time.Sleep(100 * time.Millisecond)

	// Should be expired
	if _, ok := cache.Get("key1"); ok {
		t.Error("key1 should have expired")
	}
}

func TestLRUWithTTL_Stats(t *testing.T) {
	cache, err := NewLRUWithTTL[string, int](5, 0)
	if err != nil {
		t.Fatalf("failed to create cache: %v", err)
	}
	defer cache.Close()

	cache.Set("key1", 1)
	cache.Set("key2", 2)

	cache.Get("key1")   // hit
	cache.Get("key1")   // hit
	cache.Get("missing") // miss

	stats := cache.Stats()
	if stats.Hits != 2 {
		t.Errorf("Stats.Hits = %d, want 2", stats.Hits)
	}
	if stats.Misses != 1 {
		t.Errorf("Stats.Misses = %d, want 1", stats.Misses)
	}
	if stats.Size != 2 {
		t.Errorf("Stats.Size = %d, want 2", stats.Size)
	}

	expectedHitRate := 2.0 / 3.0
	if stats.HitRate < expectedHitRate-0.01 || stats.HitRate > expectedHitRate+0.01 {
		t.Errorf("Stats.HitRate = %f, want ~%f", stats.HitRate, expectedHitRate)
	}
}

func TestLRUWithTTL_Delete(t *testing.T) {
	cache, err := NewLRUWithTTL[string, int](5, 0)
	if err != nil {
		t.Fatalf("failed to create cache: %v", err)
	}
	defer cache.Close()

	cache.Set("key1", 42)
	cache.Delete("key1")

	if _, ok := cache.Get("key1"); ok {
		t.Error("key1 should have been deleted")
	}
}

func TestLRUWithTTL_Clear(t *testing.T) {
	cache, err := NewLRUWithTTL[string, int](5, 0)
	if err != nil {
		t.Fatalf("failed to create cache: %v", err)
	}
	defer cache.Close()

	cache.Set("key1", 1)
	cache.Set("key2", 2)
	cache.Set("key3", 3)

	cache.Clear()

	if cache.Len() != 0 {
		t.Errorf("Len() = %d after Clear(), want 0", cache.Len())
	}
}

func TestLRUWithTTL_CleanupExpired(t *testing.T) {
	cache, err := NewLRUWithTTL[string, int](10, 50*time.Millisecond)
	if err != nil {
		t.Fatalf("failed to create cache: %v", err)
	}
	defer cache.Close()

	cache.Set("key1", 1)
	cache.Set("key2", 2)
	cache.Set("key3", 3)

	// Wait for expiration
	time.Sleep(100 * time.Millisecond)

	removed := cache.CleanupExpired()
	if removed != 3 {
		t.Errorf("CleanupExpired() = %d, want 3", removed)
	}

	if cache.Len() != 0 {
		t.Errorf("Len() = %d after cleanup, want 0", cache.Len())
	}
}
