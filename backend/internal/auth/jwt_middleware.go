package auth

import (
	"context"
	"encoding/json"
	"net/http"
	"strings"
)

// Phase 11 WP3: Gateway JWT Authentication
// Backend middleware that validates gateway-verified JWT headers
// Prevents tenant_id spoofing by requiring gateway-level verification

type contextKey string

const (
	// Context keys for request-scoped data
	TenantIDKey contextKey = "tenant_id"
	UserIDKey   contextKey = "user_id"
	ScopesKey   contextKey = "scopes"
)

// JWTConfig holds JWT middleware configuration
type JWTConfig struct {
	Enabled          bool
	RequireVerified  bool   // Require X-Auth-Verified header
	TenantIDHeader   string // Default: "X-Tenant-ID"
	UserIDHeader     string // Default: "X-User-ID"
	ScopesHeader     string // Default: "X-Scopes"
	VerifiedHeader   string // Default: "X-Auth-Verified"
	BypassForHealth  bool   // Allow /health without JWT
	BypassForMetrics bool   // Allow /metrics without JWT
}

// DefaultJWTConfig returns production defaults
func DefaultJWTConfig() *JWTConfig {
	return &JWTConfig{
		Enabled:          true,
		RequireVerified:  true,
		TenantIDHeader:   "X-Tenant-ID",
		UserIDHeader:     "X-User-ID",
		ScopesHeader:     "X-Scopes",
		VerifiedHeader:   "X-Auth-Verified",
		BypassForHealth:  true,
		BypassForMetrics: true,
	}
}

// JWTMiddleware validates JWT headers set by gateway (Envoy/NGINX)
func JWTMiddleware(config *JWTConfig) func(http.Handler) http.Handler {
	if config == nil {
		config = DefaultJWTConfig()
	}

	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			// Skip JWT check if disabled
			if !config.Enabled {
				next.ServeHTTP(w, r)
				return
			}

			// Bypass for health/metrics if configured
			if config.BypassForHealth && r.URL.Path == "/health" {
				next.ServeHTTP(w, r)
				return
			}
			if config.BypassForMetrics && r.URL.Path == "/metrics" {
				next.ServeHTTP(w, r)
				return
			}

			// Require gateway-verified JWT
			if config.RequireVerified {
				verified := r.Header.Get(config.VerifiedHeader)
				if verified != "true" {
					sendError(w, http.StatusUnauthorized, "Unauthorized: JWT verification required at gateway")
					return
				}
			}

			// Parse tenant_id from header (set by gateway from JWT claims)
			tenantID := r.Header.Get(config.TenantIDHeader)
			if tenantID == "" {
				sendError(w, http.StatusUnauthorized, "Unauthorized: Missing tenant_id claim in JWT")
				return
			}

			// Parse optional user_id
			userID := r.Header.Get(config.UserIDHeader)

			// Parse optional scopes (JSON array)
			var scopes []string
			scopesRaw := r.Header.Get(config.ScopesHeader)
			if scopesRaw != "" {
				if err := json.Unmarshal([]byte(scopesRaw), &scopes); err != nil {
					// Try comma-separated format as fallback
					scopes = strings.Split(scopesRaw, ",")
					for i := range scopes {
						scopes[i] = strings.TrimSpace(scopes[i])
					}
				}
			}

			// Bind to request context
			ctx := r.Context()
			ctx = context.WithValue(ctx, TenantIDKey, tenantID)
			if userID != "" {
				ctx = context.WithValue(ctx, UserIDKey, userID)
			}
			if len(scopes) > 0 {
				ctx = context.WithValue(ctx, ScopesKey, scopes)
			}

			// Continue with enriched context
			next.ServeHTTP(w, r.WithContext(ctx))
		})
	}
}

// GetTenantID extracts tenant_id from request context
func GetTenantID(ctx context.Context) (string, bool) {
	tenantID, ok := ctx.Value(TenantIDKey).(string)
	return tenantID, ok
}

// GetUserID extracts user_id from request context
func GetUserID(ctx context.Context) (string, bool) {
	userID, ok := ctx.Value(UserIDKey).(string)
	return userID, ok
}

// GetScopes extracts scopes from request context
func GetScopes(ctx context.Context) ([]string, bool) {
	scopes, ok := ctx.Value(ScopesKey).([]string)
	return scopes, ok
}

// RequireScope checks if request has required scope
func RequireScope(ctx context.Context, requiredScope string) bool {
	scopes, ok := GetScopes(ctx)
	if !ok {
		return false
	}

	for _, scope := range scopes {
		if scope == requiredScope {
			return true
		}
	}
	return false
}

// sendError writes JSON error response
func sendError(w http.ResponseWriter, statusCode int, message string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(statusCode)
	json.NewEncoder(w).Encode(map[string]interface{}{
		"error":  true,
		"status": statusCode,
		"message": message,
	})
}
