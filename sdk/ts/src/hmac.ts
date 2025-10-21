/**
 * HMAC-SHA256 signing utilities for PCS.
 * Phase 10 WP2: Simplified HMAC (sign payload directly, no pre-hash).
 */

import { createHmac } from 'crypto';
import { signaturePayload } from './canonical.js';

/**
 * Sign a PCS using HMAC-SHA256.
 *
 * Process (WP2 simplified):
 * 1. Generate canonical signature payload
 * 2. HMAC-SHA256 the payload directly with provided key
 * 3. Return base64-encoded signature
 *
 * Pre-hashing is unnecessary since HMAC provides cryptographic security.
 *
 * @param pcs - Complete PCS object
 * @param key - HMAC secret key (string or Buffer)
 * @returns Base64-encoded HMAC signature
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
 *   shard_id: "s1"
 * };
 * const signature = signHMAC(pcs, "my-secret-key");
 * ```
 */
export function signHMAC(pcs: Record<string, any>, key: string | Buffer): string {
  const payload = signaturePayload(pcs);
  const hmac = createHmac('sha256', key);
  hmac.update(payload);
  return hmac.digest('base64');
}

/**
 * Verify HMAC-SHA256 signature on a PCS.
 *
 * Process (WP2 simplified):
 * 1. Extract signature from pcs.sig
 * 2. Generate canonical signature payload (excluding sig field)
 * 3. HMAC-SHA256 the payload directly with provided key
 * 4. Compare with extracted signature
 *
 * @param pcs - Complete PCS object with "sig" field
 * @param key - HMAC secret key (string or Buffer)
 * @returns true if signature is valid, false otherwise
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
 *   sig: "base64-signature"
 * };
 * const isValid = verifyHMAC(pcs, "my-secret-key");
 * ```
 */
export function verifyHMAC(pcs: Record<string, any>, key: string | Buffer): boolean {
  try {
    // Extract signature
    const sigB64 = pcs.sig;
    if (!sigB64) {
      return false;
    }

    // Compute signature over PCS (excluding sig field)
    const pcsWithoutSig = { ...pcs };
    delete pcsWithoutSig.sig;

    const payload = signaturePayload(pcsWithoutSig);
    const hmac = createHmac('sha256', key);
    hmac.update(payload);
    const computed = hmac.digest('base64');

    // Simple string comparison (Node.js crypto.timingSafeEqual would be better)
    return computed === sigB64;
  } catch (error) {
    // Any error in verification should return false
    return false;
  }
}
