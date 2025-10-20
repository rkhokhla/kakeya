package main

import (
	"context"
	"encoding/json"
	"io"
	"log"
	"net/http"
	"os"
	"os/signal"
	"strconv"
	"strings"
	"syscall"
	"time"

	"github.com/fractal-lba/kakeya/internal/api"
	"github.com/fractal-lba/kakeya/internal/dedup"
	"github.com/fractal-lba/kakeya/internal/metrics"
	"github.com/fractal-lba/kakeya/internal/signing"
	"github.com/fractal-lba/kakeya/internal/tenant"
	"github.com/fractal-lba/kakeya/internal/verify"
	"github.com/fractal-lba/kakeya/internal/wal"
	"github.com/prometheus/client_golang/prometheus/promhttp"
	"golang.org/x/time/rate"
)

type Server struct {
	verifier     *verify.Engine
	dedupStore   dedup.Store
	inboxWAL     *wal.InboxWAL
	sigVerifier  *signing.MultiTenantVerifier
	metrics      *metrics.Metrics
	limiter      *rate.Limiter // Global rate limiter (backward compat)
	tenantMgr    *tenant.Manager
	multiTenant  bool // Phase 3 multi-tenant mode enabled
	metricsAuth struct {
		enabled  bool
		user     string
		password string
	}
}

func main() {
	// Initialize components
	verifyParams := api.DefaultVerifyParams()
	verifier := verify.NewEngine(verifyParams)

	// Setup dedup store
	dedupBackend := getEnv("DEDUP_BACKEND", "memory")
	var dedupStore dedup.Store
	var err error

	switch dedupBackend {
	case "memory":
		snapshotPath := getEnv("DEDUP_SNAPSHOT", "data/dedup.json")
		dedupStore = dedup.NewMemoryStore(snapshotPath)
	case "redis":
		redisAddr := getEnv("REDIS_ADDR", "localhost:6379")
		dedupStore, err = dedup.NewRedisStore(redisAddr)
		if err != nil {
			log.Fatalf("Failed to create Redis store: %v", err)
		}
	case "postgres":
		connStr := getEnv("POSTGRES_CONN", "")
		dedupStore, err = dedup.NewPostgresStore(connStr)
		if err != nil {
			log.Fatalf("Failed to create Postgres store: %v", err)
		}
	default:
		log.Fatalf("Unknown DEDUP_BACKEND: %s", dedupBackend)
	}

	// Setup WAL
	walDir := getEnv("WAL_DIR", "data/wal")
	inboxWAL, err := wal.NewInboxWAL(walDir)
	if err != nil {
		log.Fatalf("Failed to create inbox WAL: %v", err)
	}

	// Setup signature verification
	sigAlg := getEnv("PCS_SIGN_ALG", "none")
	var fallbackVerifier signing.Verifier

	switch sigAlg {
	case "hmac":
		hmacKey := getEnv("PCS_HMAC_KEY", "")
		if hmacKey == "" {
			log.Fatal("PCS_HMAC_KEY is required when PCS_SIGN_ALG=hmac")
		}
		fallbackVerifier = signing.NewHMACVerifier(hmacKey)
	case "ed25519":
		pubKeyB64 := getEnv("PCS_ED25519_PUB_B64", "")
		if pubKeyB64 == "" {
			log.Fatal("PCS_ED25519_PUB_B64 is required when PCS_SIGN_ALG=ed25519")
		}
		fallbackVerifier, err = signing.NewEd25519Verifier(pubKeyB64)
		if err != nil {
			log.Fatalf("Failed to create Ed25519 verifier: %v", err)
		}
	case "none":
		fallbackVerifier = &signing.NoOpVerifier{}
	default:
		log.Fatalf("Unknown PCS_SIGN_ALG: %s", sigAlg)
	}

	// Multi-tenant signature verifier (Phase 3)
	mtVerifier := signing.NewMultiTenantVerifier(fallbackVerifier)

	// Setup metrics
	m := metrics.New()

	// Rate limiter (global, for backward compatibility)
	tokenRate := getEnvInt("TOKEN_RATE", 100)
	limiter := rate.NewLimiter(rate.Limit(tokenRate), tokenRate*2)

	// Multi-tenant mode (Phase 3)
	multiTenantEnabled := getEnv("MULTI_TENANT", "false") == "true"
	tenantMgr := tenant.NewManager()

	// Register default tenant for backward compatibility
	if !multiTenantEnabled {
		defaultTenant := tenant.DefaultTenant()
		if err := tenantMgr.RegisterTenant(defaultTenant); err != nil {
			log.Fatalf("Failed to register default tenant: %v", err)
		}
		log.Println("Running in single-tenant mode (backward compatible)")
	} else {
		log.Println("Running in multi-tenant mode (Phase 3)")
		// Load tenants from configuration
		loadTenants(tenantMgr, mtVerifier)
	}

	// Create server
	srv := &Server{
		verifier:    verifier,
		dedupStore:  dedupStore,
		inboxWAL:    inboxWAL,
		sigVerifier: mtVerifier,
		metrics:     m,
		limiter:     limiter,
		tenantMgr:   tenantMgr,
		multiTenant: multiTenantEnabled,
	}

	// Metrics auth
	srv.metricsAuth.enabled = getEnv("METRICS_USER", "") != ""
	srv.metricsAuth.user = getEnv("METRICS_USER", "")
	srv.metricsAuth.password = getEnv("METRICS_PASS", "")

	// Setup HTTP routes
	mux := http.NewServeMux()
	mux.HandleFunc("/v1/pcs/submit", srv.handleSubmit)
	mux.Handle("/metrics", srv.metricsHandler())
	mux.HandleFunc("/health", handleHealth)

	// HTTP server
	port := getEnv("PORT", "8080")
	httpServer := &http.Server{
		Addr:         ":" + port,
		Handler:      mux,
		ReadTimeout:  10 * time.Second,
		WriteTimeout: 10 * time.Second,
		IdleTimeout:  60 * time.Second,
	}

	// Graceful shutdown
	shutdown := make(chan os.Signal, 1)
	signal.Notify(shutdown, os.Interrupt, syscall.SIGTERM)

	go func() {
		log.Printf("Starting server on port %s", port)
		if err := httpServer.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Fatalf("Server error: %v", err)
		}
	}()

	<-shutdown
	log.Println("Shutting down server...")

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	if err := httpServer.Shutdown(ctx); err != nil {
		log.Printf("Server shutdown error: %v", err)
	}

	// Close resources
	if err := inboxWAL.Close(); err != nil {
		log.Printf("Error closing WAL: %v", err)
	}
	if err := dedupStore.Close(); err != nil {
		log.Printf("Error closing dedup store: %v", err)
	}

	log.Println("Server stopped")
}

func (s *Server) handleSubmit(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	// Extract tenant ID from header (Phase 3)
	tenantID := r.Header.Get("X-Tenant-Id")
	if tenantID == "" {
		tenantID = "default" // Backward compatibility
	}

	// Tenant-aware rate limiting (Phase 3)
	if s.multiTenant {
		ctx := r.Context()
		if err := s.tenantMgr.Allow(ctx, tenantID); err != nil {
			if err == tenant.ErrQuotaExceeded {
				s.metrics.QuotaExceededByTenant.WithLabelValues(tenantID).Inc()
				w.Header().Set("Retry-After", "10")
				http.Error(w, "Tenant quota exceeded", http.StatusTooManyRequests)
				return
			}
			log.Printf("Tenant check failed for %s: %v", tenantID, err)
			http.Error(w, "Tenant error", http.StatusBadRequest)
			return
		}
	} else {
		// Global rate limiting (backward compatibility)
		if !s.limiter.Allow() {
			w.Header().Set("Retry-After", "10")
			http.Error(w, "Too many requests", http.StatusTooManyRequests)
			return
		}
	}

	// Update metrics (both global and per-tenant)
	s.metrics.IngestTotal.Inc()
	if s.multiTenant {
		s.metrics.IngestTotalByTenant.WithLabelValues(tenantID).Inc()
	}

	// Read body
	body, err := io.ReadAll(io.LimitReader(r.Body, 1<<20)) // 1MB limit
	if err != nil {
		http.Error(w, "Failed to read body", http.StatusBadRequest)
		return
	}

	// Append to WAL BEFORE parsing (fault tolerance)
	if err := s.inboxWAL.Append(body); err != nil {
		log.Printf("WAL append error: %v", err)
		s.metrics.WALErrors.Inc()
		http.Error(w, "Internal server error", http.StatusInternalServerError)
		return
	}

	// Parse PCS
	var pcs api.PCS
	if err := json.Unmarshal(body, &pcs); err != nil {
		http.Error(w, "Invalid JSON", http.StatusBadRequest)
		return
	}

	// Verify signature BEFORE dedup check (per CLAUDE_PHASE1.md)
	// This ensures we don't cache results for unsigned/invalid signatures
	// Use tenant-specific verifier in multi-tenant mode (Phase 3)
	if s.multiTenant {
		if err := s.sigVerifier.VerifyForTenant(tenantID, &pcs); err != nil {
			log.Printf("Signature verification failed for tenant %s, PCS %s: %v", tenantID, pcs.PCSID, err)
			s.metrics.SignatureErr.Inc()
			s.metrics.SignatureErrByTenant.WithLabelValues(tenantID).Inc()
			http.Error(w, "Signature verification failed", http.StatusUnauthorized)
			return
		}
	} else {
		if err := s.sigVerifier.Verify(&pcs); err != nil {
			log.Printf("Signature verification failed for %s: %v", pcs.PCSID, err)
			s.metrics.SignatureErr.Inc()
			http.Error(w, "Signature verification failed", http.StatusUnauthorized)
			return
		}
	}

	// Idempotent dedup check
	ctx := r.Context()
	existingResult, err := s.dedupStore.Get(ctx, pcs.PCSID)
	if err != nil {
		log.Printf("Dedup store error: %v", err)
		http.Error(w, "Internal server error", http.StatusInternalServerError)
		return
	}

	if existingResult != nil {
		// Duplicate - return cached result
		s.metrics.DedupHits.Inc()
		if s.multiTenant {
			s.metrics.DedupHitsByTenant.WithLabelValues(tenantID).Inc()
		}
		respondWithResult(w, existingResult)
		return
	}

	// Verify PCS
	result, err := s.verifier.Verify(&pcs)
	if err != nil {
		log.Printf("Verification error for %s: %v", pcs.PCSID, err)
		http.Error(w, "Verification failed", http.StatusInternalServerError)
		return
	}

	// Store result (first-write wins due to dedup contract)
	ttl := s.verifier.Params().DedupTTL
	if err := s.dedupStore.Set(ctx, pcs.PCSID, result, ttl); err != nil {
		log.Printf("Failed to store dedup result: %v", err)
		// Continue anyway - this is not fatal
	}

	// Update metrics (global and per-tenant)
	if result.Accepted && !result.Escalated {
		s.metrics.Accepted.Inc()
		if s.multiTenant {
			s.metrics.AcceptedByTenant.WithLabelValues(tenantID).Inc()
		}
	}
	if result.Escalated {
		s.metrics.Escalated.Inc()
		if s.multiTenant {
			s.metrics.EscalatedByTenant.WithLabelValues(tenantID).Inc()
		}
	}

	respondWithResult(w, result)
}

func (s *Server) metricsHandler() http.Handler {
	handler := promhttp.Handler()

	if !s.metricsAuth.enabled {
		return handler
	}

	// Wrap with Basic Auth
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		user, pass, ok := r.BasicAuth()
		if !ok || user != s.metricsAuth.user || pass != s.metricsAuth.password {
			w.Header().Set("WWW-Authenticate", `Basic realm="Metrics"`)
			http.Error(w, "Unauthorized", http.StatusUnauthorized)
			return
		}
		handler.ServeHTTP(w, r)
	})
}

func handleHealth(w http.ResponseWriter, r *http.Request) {
	w.WriteHeader(http.StatusOK)
	w.Write([]byte("OK"))
}

func respondWithResult(w http.ResponseWriter, result *api.VerifyResult) {
	status := http.StatusOK
	if result.Escalated {
		status = http.StatusAccepted
	}
	if !result.Accepted {
		status = http.StatusAccepted
	}

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(result)
}

func loadConfig() map[string]string {
	return map[string]string{
		"DEDUP_BACKEND": getEnv("DEDUP_BACKEND", "memory"),
		"PORT":          getEnv("PORT", "8080"),
	}
}

func getEnv(key, defaultValue string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return defaultValue
}

func getEnvInt(key string, defaultValue int) int {
	if value := os.Getenv(key); value != "" {
		if i, err := strconv.Atoi(value); err == nil {
			return i
		}
	}
	return defaultValue
}

// loadTenants loads tenant configuration from environment variables (Phase 3)
// Format: TENANTS=tenant1:hmac:key1,tenant2:ed25519:pubkey2
func loadTenants(mgr *tenant.Manager, verifier *signing.MultiTenantVerifier) {
	tenantsConfig := getEnv("TENANTS", "")
	if tenantsConfig == "" {
		log.Println("No tenants configured. Use TENANTS env var to configure multi-tenant mode")
		// Register a default tenant for testing
		defaultTenant := tenant.DefaultTenant()
		if err := mgr.RegisterTenant(defaultTenant); err != nil {
			log.Printf("Warning: failed to register default tenant: %v", err)
		}
		return
	}

	tenantSpecs := strings.Split(tenantsConfig, ",")
	for _, spec := range tenantSpecs {
		parts := strings.Split(spec, ":")
		if len(parts) < 3 {
			log.Printf("Invalid tenant spec: %s (expected format: tenant_id:alg:key)", spec)
			continue
		}

		tenantID := strings.TrimSpace(parts[0])
		alg := strings.TrimSpace(parts[1])
		key := strings.TrimSpace(parts[2])

		// Register tenant
		t := &tenant.Tenant{
			ID:           tenantID,
			DisplayName:  tenantID,
			SigningKey:   key,
			SigningAlg:   alg,
			TokenRate:    100,  // Default rate
			BurstRate:    200,  // Default burst
			DailyQuota:   0,    // Unlimited by default
			CustomParams: false,
			CreatedAt:    time.Now(),
			Active:       true,
			Metadata:     make(map[string]string),
		}

		if err := mgr.RegisterTenant(t); err != nil {
			log.Printf("Failed to register tenant %s: %v", tenantID, err)
			continue
		}

		// Register verifier for tenant
		if err := verifier.RegisterTenant(tenantID, alg, key); err != nil {
			log.Printf("Failed to register verifier for tenant %s: %v", tenantID, err)
			continue
		}

		log.Printf("Registered tenant: %s (alg=%s)", tenantID, alg)
	}
}
