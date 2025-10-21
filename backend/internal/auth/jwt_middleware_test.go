package auth

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestJWTMiddleware_Disabled(t *testing.T) {
	config := &JWTConfig{Enabled: false}
	middleware := JWTMiddleware(config)

	handler := middleware(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	}))

	req := httptest.NewRequest("GET", "/v1/pcs/submit", nil)
	w := httptest.NewRecorder()

	handler.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("Expected status 200, got %d", w.Code)
	}
}

func TestJWTMiddleware_MissingVerified(t *testing.T) {
	config := DefaultJWTConfig()
	middleware := JWTMiddleware(config)

	handler := middleware(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	}))

	req := httptest.NewRequest("GET", "/v1/pcs/submit", nil)
	w := httptest.NewRecorder()

	handler.ServeHTTP(w, req)

	if w.Code != http.StatusUnauthorized {
		t.Errorf("Expected status 401, got %d", w.Code)
	}
}

func TestJWTMiddleware_MissingTenantID(t *testing.T) {
	config := DefaultJWTConfig()
	middleware := JWTMiddleware(config)

	handler := middleware(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	}))

	req := httptest.NewRequest("GET", "/v1/pcs/submit", nil)
	req.Header.Set("X-Auth-Verified", "true")
	w := httptest.NewRecorder()

	handler.ServeHTTP(w, req)

	if w.Code != http.StatusUnauthorized {
		t.Errorf("Expected status 401, got %d", w.Code)
	}
}

func TestJWTMiddleware_ValidHeaders(t *testing.T) {
	config := DefaultJWTConfig()
	middleware := JWTMiddleware(config)

	var capturedTenantID string
	var capturedUserID string
	var capturedScopes []string

	handler := middleware(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		tenantID, ok := GetTenantID(r.Context())
		if !ok {
			t.Error("TenantID not found in context")
		}
		capturedTenantID = tenantID

		userID, ok := GetUserID(r.Context())
		if ok {
			capturedUserID = userID
		}

		scopes, ok := GetScopes(r.Context())
		if ok {
			capturedScopes = scopes
		}

		w.WriteHeader(http.StatusOK)
	}))

	req := httptest.NewRequest("GET", "/v1/pcs/submit", nil)
	req.Header.Set("X-Auth-Verified", "true")
	req.Header.Set("X-Tenant-ID", "tenant-123")
	req.Header.Set("X-User-ID", "user-456")
	req.Header.Set("X-Scopes", `["read", "write"]`)
	w := httptest.NewRecorder()

	handler.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("Expected status 200, got %d", w.Code)
	}

	if capturedTenantID != "tenant-123" {
		t.Errorf("Expected tenant_id 'tenant-123', got '%s'", capturedTenantID)
	}

	if capturedUserID != "user-456" {
		t.Errorf("Expected user_id 'user-456', got '%s'", capturedUserID)
	}

	if len(capturedScopes) != 2 || capturedScopes[0] != "read" || capturedScopes[1] != "write" {
		t.Errorf("Expected scopes ['read', 'write'], got %v", capturedScopes)
	}
}

func TestJWTMiddleware_BypassHealth(t *testing.T) {
	config := DefaultJWTConfig()
	config.BypassForHealth = true
	middleware := JWTMiddleware(config)

	handler := middleware(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	}))

	req := httptest.NewRequest("GET", "/health", nil)
	w := httptest.NewRecorder()

	handler.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("Expected status 200 for /health bypass, got %d", w.Code)
	}
}

func TestJWTMiddleware_BypassMetrics(t *testing.T) {
	config := DefaultJWTConfig()
	config.BypassForMetrics = true
	middleware := JWTMiddleware(config)

	handler := middleware(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	}))

	req := httptest.NewRequest("GET", "/metrics", nil)
	w := httptest.NewRecorder()

	handler.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("Expected status 200 for /metrics bypass, got %d", w.Code)
	}
}

func TestGetTenantID(t *testing.T) {
	ctx := context.Background()
	ctx = context.WithValue(ctx, TenantIDKey, "tenant-789")

	tenantID, ok := GetTenantID(ctx)
	if !ok {
		t.Error("Expected tenant_id to be present")
	}
	if tenantID != "tenant-789" {
		t.Errorf("Expected 'tenant-789', got '%s'", tenantID)
	}

	// Test missing value
	ctx2 := context.Background()
	_, ok = GetTenantID(ctx2)
	if ok {
		t.Error("Expected tenant_id to be absent")
	}
}

func TestRequireScope(t *testing.T) {
	tests := []struct {
		name          string
		scopes        []string
		requiredScope string
		expected      bool
	}{
		{
			name:          "scope present",
			scopes:        []string{"read", "write", "admin"},
			requiredScope: "write",
			expected:      true,
		},
		{
			name:          "scope absent",
			scopes:        []string{"read", "write"},
			requiredScope: "admin",
			expected:      false,
		},
		{
			name:          "empty scopes",
			scopes:        []string{},
			requiredScope: "read",
			expected:      false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ctx := context.Background()
			if len(tt.scopes) > 0 {
				ctx = context.WithValue(ctx, ScopesKey, tt.scopes)
			}

			result := RequireScope(ctx, tt.requiredScope)
			if result != tt.expected {
				t.Errorf("Expected %v, got %v", tt.expected, result)
			}
		})
	}
}

func TestScopesCSVFallback(t *testing.T) {
	config := DefaultJWTConfig()
	middleware := JWTMiddleware(config)

	var capturedScopes []string

	handler := middleware(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		scopes, ok := GetScopes(r.Context())
		if ok {
			capturedScopes = scopes
		}
		w.WriteHeader(http.StatusOK)
	}))

	req := httptest.NewRequest("GET", "/v1/pcs/submit", nil)
	req.Header.Set("X-Auth-Verified", "true")
	req.Header.Set("X-Tenant-ID", "tenant-123")
	req.Header.Set("X-Scopes", "read, write, admin") // CSV format
	w := httptest.NewRecorder()

	handler.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("Expected status 200, got %d", w.Code)
	}

	if len(capturedScopes) != 3 || capturedScopes[0] != "read" || capturedScopes[1] != "write" || capturedScopes[2] != "admin" {
		t.Errorf("Expected scopes ['read', 'write', 'admin'], got %v", capturedScopes)
	}
}
