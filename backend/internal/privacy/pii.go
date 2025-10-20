package privacy

import (
	"fmt"
	"regexp"
	"strings"
)

// PIIType represents types of personally identifiable information
type PIIType string

const (
	PIITypeEmail       PIIType = "email"
	PIITypePhone       PIIType = "phone"
	PIITypeSSN         PIIType = "ssn"
	PIITypeCreditCard  PIIType = "credit_card"
	PIITypeIPAddress   PIIType = "ip_address"
	PIITypeGeneric     PIIType = "generic"
)

// PIIDetection represents a detected PII instance
type PIIDetection struct {
	Type     PIIType
	Value    string
	Field    string
	Position int
	Confidence float64 // 0.0-1.0
}

// PIIScanner scans for personally identifiable information (Phase 3 WP5)
type PIIScanner struct {
	patterns map[PIIType]*regexp.Regexp
	mode     ScanMode
}

// ScanMode determines scanner behavior
type ScanMode int

const (
	ScanModeDetect ScanMode = iota // Report only
	ScanModeBlock                   // Reject on detection
	ScanModeRedact                  // Replace with [REDACTED]
)

// NewPIIScanner creates a new PII scanner with common patterns
func NewPIIScanner(mode ScanMode) *PIIScanner {
	return &PIIScanner{
		mode: mode,
		patterns: map[PIIType]*regexp.Regexp{
			// Email: basic pattern (RFC 5322 simplified)
			PIITypeEmail: regexp.MustCompile(`\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b`),

			// Phone: US/international formats
			PIITypePhone: regexp.MustCompile(`\b(?:\+?1[-.\s]?)?(?:\([0-9]{3}\)|[0-9]{3})[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b`),

			// SSN: XXX-XX-XXXX format
			PIITypeSSN: regexp.MustCompile(`\b[0-9]{3}-[0-9]{2}-[0-9]{4}\b`),

			// Credit card: simplified pattern (Luhn algorithm not checked)
			PIITypeCreditCard: regexp.MustCompile(`\b[0-9]{4}[-\s]?[0-9]{4}[-\s]?[0-9]{4}[-\s]?[0-9]{4}\b`),

			// IP address: IPv4
			PIITypeIPAddress: regexp.MustCompile(`\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b`),
		},
	}
}

// Scan scans text for PII patterns
func (s *PIIScanner) Scan(text string, field string) ([]PIIDetection, error) {
	var detections []PIIDetection

	for piiType, pattern := range s.patterns {
		matches := pattern.FindAllStringIndex(text, -1)
		for _, match := range matches {
			value := text[match[0]:match[1]]

			// Skip false positives
			if s.isFalsePositive(piiType, value) {
				continue
			}

			detection := PIIDetection{
				Type:       piiType,
				Value:      value,
				Field:      field,
				Position:   match[0],
				Confidence: s.computeConfidence(piiType, value),
			}

			detections = append(detections, detection)
		}
	}

	return detections, nil
}

// ScanFields scans multiple fields for PII
func (s *PIIScanner) ScanFields(fields map[string]string) ([]PIIDetection, error) {
	var allDetections []PIIDetection

	for field, value := range fields {
		detections, err := s.Scan(value, field)
		if err != nil {
			return nil, fmt.Errorf("failed to scan field %s: %w", field, err)
		}
		allDetections = append(allDetections, detections...)
	}

	return allDetections, nil
}

// Redact redacts PII from text based on scanner mode
func (s *PIIScanner) Redact(text string) (string, []PIIDetection, error) {
	detections, err := s.Scan(text, "")
	if err != nil {
		return "", nil, err
	}

	if len(detections) == 0 {
		return text, detections, nil
	}

	// Redact in reverse order to preserve positions
	redacted := text
	for i := len(detections) - 1; i >= 0; i-- {
		d := detections[i]
		replacement := fmt.Sprintf("[REDACTED_%s]", strings.ToUpper(string(d.Type)))
		redacted = redacted[:d.Position] + replacement + redacted[d.Position+len(d.Value):]
	}

	return redacted, detections, nil
}

// isFalsePositive applies heuristics to reduce false positives
func (s *PIIScanner) isFalsePositive(piiType PIIType, value string) bool {
	switch piiType {
	case PIITypeEmail:
		// Skip common test/example emails
		lower := strings.ToLower(value)
		if strings.Contains(lower, "example.com") ||
			strings.Contains(lower, "test.com") ||
			strings.Contains(lower, "localhost") {
			return true
		}

	case PIITypeIPAddress:
		// Skip common non-PII IPs (localhost, private ranges more lenient)
		if strings.HasPrefix(value, "127.") ||
			strings.HasPrefix(value, "0.") ||
			value == "255.255.255.255" {
			return true
		}

	case PIITypeCreditCard:
		// Skip common test card numbers (simplified)
		if strings.HasPrefix(value, "0000") ||
			strings.HasPrefix(value, "1111") ||
			strings.HasPrefix(value, "9999") {
			return true
		}
	}

	return false
}

// computeConfidence assigns confidence score to detection
func (s *PIIScanner) computeConfidence(piiType PIIType, value string) float64 {
	// Simple heuristic (production would use more sophisticated scoring)
	switch piiType {
	case PIITypeEmail:
		if strings.Count(value, "@") == 1 && strings.Contains(value, ".") {
			return 0.9
		}
		return 0.6

	case PIITypePhone:
		// Higher confidence if formatted with dashes/parens
		if strings.Contains(value, "-") || strings.Contains(value, "(") {
			return 0.85
		}
		return 0.7

	case PIITypeSSN:
		return 0.95 // High confidence for XXX-XX-XXXX pattern

	case PIITypeCreditCard:
		// Would apply Luhn algorithm in production
		return 0.75

	case PIITypeIPAddress:
		return 0.6 // Lower confidence (many false positives)

	default:
		return 0.5
	}
}

// Mode returns the scanner's mode
func (s *PIIScanner) Mode() ScanMode {
	return s.mode
}

// SetMode sets the scanner's mode
func (s *PIIScanner) SetMode(mode ScanMode) {
	s.mode = mode
}

// ShouldBlock returns true if scanner should block requests with PII
func (s *PIIScanner) ShouldBlock() bool {
	return s.mode == ScanModeBlock
}

// ShouldRedact returns true if scanner should redact PII
func (s *PIIScanner) ShouldRedact() bool {
	return s.mode == ScanModeRedact
}

// PIIPolicy represents PII handling policy for a tenant (Phase 3 WP5)
type PIIPolicy struct {
	TenantID       string
	Mode           ScanMode
	EnabledTypes   map[PIIType]bool
	CustomPatterns map[string]*regexp.Regexp
	ReportOnly     bool // Log but don't block (staged rollout)
}

// NewDefaultPIIPolicy returns a default PII policy
func NewDefaultPIIPolicy(tenantID string) *PIIPolicy {
	return &PIIPolicy{
		TenantID: tenantID,
		Mode:     ScanModeDetect, // Report-only by default (Phase 3 safe rollout)
		EnabledTypes: map[PIIType]bool{
			PIITypeEmail:      true,
			PIITypePhone:      true,
			PIITypeSSN:        true,
			PIITypeCreditCard: true,
			PIITypeIPAddress:  false, // Too many false positives
		},
		CustomPatterns: make(map[string]*regexp.Regexp),
		ReportOnly:     true,
	}
}
