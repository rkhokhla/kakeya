/**
 * Canonical JSON utilities for FLK/PCS signature generation.
 * Phase 10 WP1: Ensures cross-language signature compatibility (Python/Go/TS).
 *
 * Key requirements:
 * - Floats formatted to exactly 9 decimal places
 * - Stable field ordering for signature subset
 * - No whitespace in JSON output
 */

/**
 * Signature subset: 8 fields used for signing (Phase 1 specification)
 */
export const SIGNATURE_FIELDS = [
  "pcs_id",
  "merkle_root",
  "epoch",
  "shard_id",
  "D_hat",
  "coh_star",
  "r",
  "budget",
] as const;

/**
 * Signature subset interface
 */
export interface SignatureSubset {
  pcs_id: string;
  merkle_root: string;
  epoch: number;
  shard_id: string;
  D_hat: number;
  coh_star: number;
  r: number;
  budget: number;
}

/**
 * Format float to exactly 9 decimal places.
 *
 * This ensures stable signatures across languages and prevents
 * floating-point drift.
 *
 * @param value - Float to format
 * @returns String representation with 9 decimal places (e.g., "1.234567890")
 *
 * @example
 * ```typescript
 * formatFloat9dp(1.23456789012345); // returns "1.234567890"
 * formatFloat9dp(0.5);              // returns "0.500000000"
 * ```
 */
export function formatFloat9dp(value: number): string {
  return value.toFixed(9);
}

/**
 * Round float to 9 decimal places.
 *
 * Used for normalizing floats before formatting to ensure
 * consistent behavior across operations.
 *
 * @param value - Float to round
 * @returns Float rounded to 9 decimal places
 */
export function round9(value: number): number {
  const factor = 1e9;
  return Math.round(value * factor) / factor;
}

/**
 * Extract the 8-field subset used for signing.
 *
 * @param pcs - Full PCS object
 * @returns Object with only signature fields
 * @throws Error if required signature field is missing
 */
export function extractSignatureSubset(pcs: Record<string, any>): SignatureSubset {
  const subset: Partial<SignatureSubset> = {};

  for (const field of SIGNATURE_FIELDS) {
    if (!(field in pcs)) {
      throw new Error(`Missing required signature field: ${field}`);
    }
    subset[field as keyof SignatureSubset] = pcs[field];
  }

  return subset as SignatureSubset;
}

/**
 * Generate canonical JSON bytes for signing.
 *
 * Rules:
 * - Signature fields only (8 fields)
 * - Floats formatted to 9 decimal places
 * - Keys sorted alphabetically
 * - No whitespace (compact JSON)
 * - UTF-8 encoded
 *
 * @param pcsSubset - PCS object (should contain signature fields only)
 * @returns Canonical JSON as Uint8Array, ready for signing
 *
 * @example
 * ```typescript
 * const pcs = {
 *   pcs_id: "abc123",
 *   D_hat: 1.23456789,
 *   coh_star: 0.75,
 *   r: 0.5,
 *   budget: 0.35,
 *   merkle_root: "def456",
 *   epoch: 1,
 *   shard_id: "shard-001"
 * };
 * const payload = canonicalJSONBytes(pcs);
 * ```
 */
export function canonicalJSONBytes(pcsSubset: SignatureSubset): Uint8Array {
  // Normalize floats to 9 decimal places
  const normalized: Record<string, any> = {};

  for (const [key, value] of Object.entries(pcsSubset)) {
    if (typeof value === "number" && !Number.isInteger(value)) {
      // Round to 9dp, then parse back to number to preserve precision
      normalized[key] = parseFloat(formatFloat9dp(round9(value)));
    } else {
      normalized[key] = value;
    }
  }

  // Sort keys alphabetically
  const sortedKeys = Object.keys(normalized).sort();
  const sortedObj: Record<string, any> = {};
  for (const key of sortedKeys) {
    sortedObj[key] = normalized[key];
  }

  // Generate compact JSON (no whitespace)
  const jsonStr = JSON.stringify(sortedObj);

  // Convert to UTF-8 bytes
  const encoder = new TextEncoder();
  return encoder.encode(jsonStr);
}

/**
 * Generate signature payload from full PCS.
 *
 * Convenience function that combines subset extraction and
 * canonical JSON generation.
 *
 * @param pcs - Full PCS object
 * @returns Canonical payload bytes for signing
 *
 * @example
 * ```typescript
 * const pcs = {
 *   pcs_id: "test",
 *   D_hat: 1.0,
 *   coh_star: 0.75,
 *   r: 0.5,
 *   budget: 0.35,
 *   merkle_root: "abc",
 *   epoch: 1,
 *   shard_id: "s1",
 *   extra_field: "ignored"
 * };
 * const payload = signaturePayload(pcs);
 * ```
 */
export function signaturePayload(pcs: Record<string, any>): Uint8Array {
  const subset = extractSignatureSubset(pcs);
  return canonicalJSONBytes(subset);
}

/**
 * Backwards compatibility: Phase 1-9 used this function name.
 * @deprecated Use signaturePayload() instead.
 */
export function SigningPayload(pcs: Record<string, any>): Uint8Array {
  return signaturePayload(pcs);
}
