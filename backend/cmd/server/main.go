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
	"syscall"
	"time"

	"github.com/fractal-lba/kakeya/internal/api"
	"github.com/fractal-lba/kakeya/internal/dedup"
	"github.com/fractal-lba/kakeya/internal/metrics"
	"github.com/fractal-lba/kakeya/internal/signing"
	"github.com/fractal-lba/kakeya/internal/verify"
	"github.com/fractal-lba/kakeya/internal/wal"
	"github.com/prometheus/client_golang/prometheus/promhttp"
	"golang.org/x/time/rate"
)

type Server struct {
	verifier  *verify.Engine
	dedupStore dedup.Store
	inboxWAL  *wal.InboxWAL
	sigVerifier signing.Verifier
	metrics   *metrics.Metrics
	limiter   *rate.Limiter
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
	var sigVerifier signing.Verifier

	switch sigAlg {
	case "hmac":
		hmacKey := getEnv("PCS_HMAC_KEY", "")
		if hmacKey == "" {
			log.Fatal("PCS_HMAC_KEY is required when PCS_SIGN_ALG=hmac")
		}
		sigVerifier = signing.NewHMACVerifier(hmacKey)
	case "ed25519":
		pubKeyB64 := getEnv("PCS_ED25519_PUB_B64", "")
		if pubKeyB64 == "" {
			log.Fatal("PCS_ED25519_PUB_B64 is required when PCS_SIGN_ALG=ed25519")
		}
		sigVerifier, err = signing.NewEd25519Verifier(pubKeyB64)
		if err != nil {
			log.Fatalf("Failed to create Ed25519 verifier: %v", err)
		}
	case "none":
		sigVerifier = &signing.NoOpVerifier{}
	default:
		log.Fatalf("Unknown PCS_SIGN_ALG: %s", sigAlg)
	}

	// Setup metrics
	m := metrics.New()

	// Rate limiter
	tokenRate := getEnvInt("TOKEN_RATE", 100)
	limiter := rate.NewLimiter(rate.Limit(tokenRate), tokenRate*2)

	// Create server
	srv := &Server{
		verifier:    verifier,
		dedupStore:  dedupStore,
		inboxWAL:    inboxWAL,
		sigVerifier: sigVerifier,
		metrics:     m,
		limiter:     limiter,
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

	// Rate limiting
	if !s.limiter.Allow() {
		w.Header().Set("Retry-After", "10")
		http.Error(w, "Too many requests", http.StatusTooManyRequests)
		return
	}

	s.metrics.IngestTotal.Inc()

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
		respondWithResult(w, existingResult)
		return
	}

	// Verify signature
	if err := s.sigVerifier.Verify(&pcs); err != nil {
		log.Printf("Signature verification failed for %s: %v", pcs.PCSID, err)
		s.metrics.SignatureErr.Inc()
		http.Error(w, "Signature verification failed", http.StatusUnauthorized)
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

	// Update metrics
	if result.Accepted && !result.Escalated {
		s.metrics.Accepted.Inc()
	}
	if result.Escalated {
		s.metrics.Escalated.Inc()
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
