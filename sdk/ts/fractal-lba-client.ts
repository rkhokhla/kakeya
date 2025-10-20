// TypeScript SDK for Fractal LBA + Kakeya FT Stack API (Phase 4 WP5)
import * as crypto from 'crypto';

/**
 * Proof-of-Computation Summary structure
 */
export interface PCS {
  pcs_id: string;
  schema: string;
  version: string;
  shard_id: string;
  epoch: number;
  attempt: number;
  sent_at: string;
  seed: number;
  scales: number[];
  N_j: Record<string, number>;
  coh_star: number;
  v_star: number[];
  D_hat: number;
  r: number;
  regime: string;
  budget: number;
  merkle_root: string;
  sig?: string;
  ft: FaultToleranceInfo;
}

/**
 * Fault tolerance metadata
 */
export interface FaultToleranceInfo {
  outbox_seq: number;
  degraded: boolean;
  fallbacks: string[];
  clock_skew_ms: number;
}

/**
 * Verification result from backend
 */
export interface VerifyResult {
  accepted: boolean;
  recomputed_D_hat?: number;
  recomputed_budget?: number;
  reason?: string;
  escalated: boolean;
}

/**
 * Client configuration options
 */
export interface ClientOptions {
  baseURL: string;
  tenantID?: string;
  signingKey?: string;
  signingAlg?: 'hmac' | 'none';
  timeout?: number;
  maxRetries?: number;
}

/**
 * Custom error types
 */
export class FractalLBAError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'FractalLBAError';
  }
}

export class SignatureError extends FractalLBAError {
  constructor(message: string) {
    super(message);
    this.name = 'SignatureError';
  }
}

export class ValidationError extends FractalLBAError {
  constructor(message: string) {
    super(message);
    this.name = 'ValidationError';
  }
}

export class APIError extends FractalLBAError {
  constructor(
    message: string,
    public statusCode: number,
    public responseBody?: string
  ) {
    super(message);
    this.name = 'APIError';
  }
}

/**
 * Fractal LBA API Client (Phase 4 WP5)
 *
 * Implements Phase 1 canonical signing with automatic HMAC-SHA256 signing,
 * multi-tenant support, retry logic, and validation.
 *
 * @example
 * const client = new FractalLBAClient({
 *   baseURL: 'https://api.example.com',
 *   tenantID: 'tenant1',
 *   signingKey: 'supersecret',
 *   signingAlg: 'hmac'
 * });
 *
 * const result = await client.submitPCS(pcs);
 * console.log('Accepted:', result.accepted);
 */
export class FractalLBAClient {
  private baseURL: string;
  private tenantID: string;
  private signingKey: string;
  private signingAlg: 'hmac' | 'none';
  private timeout: number;
  private maxRetries: number;

  constructor(options: ClientOptions) {
    this.baseURL = options.baseURL;
    this.tenantID = options.tenantID || '';
    this.signingKey = options.signingKey || '';
    this.signingAlg = options.signingAlg || 'none';
    this.timeout = options.timeout || 30000;
    this.maxRetries = options.maxRetries || 3;
  }

  /**
   * Submit a Proof-of-Computation Summary
   */
  async submitPCS(pcs: PCS): Promise<VerifyResult> {
    // Validate PCS
    this.validatePCS(pcs);

    // Sign PCS if signing is enabled
    if (this.signingAlg !== 'none') {
      this.signPCS(pcs);
    }

    // Send request with retries
    return this.retryRequest(async () => {
      const response = await fetch(`${this.baseURL}/v1/pcs/submit`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'User-Agent': 'fractal-lba-ts-sdk/0.4.0',
          ...(this.tenantID && { 'X-Tenant-Id': this.tenantID })
        },
        body: JSON.stringify(pcs),
        signal: AbortSignal.timeout(this.timeout)
      });

      // Read response body
      const responseBody = await response.text();

      // Handle response status
      switch (response.status) {
        case 200:
        case 202:
          return JSON.parse(responseBody) as VerifyResult;
        case 401:
          throw new SignatureError('Signature verification failed (401)');
        case 429:
          throw new APIError('Rate limit exceeded (429)', 429, responseBody);
        default:
          throw new APIError(
            `API error (${response.status})`,
            response.status,
            responseBody
          );
      }
    });
  }

  /**
   * Health check
   */
  async healthCheck(): Promise<void> {
    const response = await fetch(`${this.baseURL}/health`, {
      method: 'GET',
      signal: AbortSignal.timeout(this.timeout)
    });

    if (response.status !== 200) {
      throw new APIError(
        `Health check failed with status: ${response.status}`,
        response.status
      );
    }
  }

  /**
   * Validate PCS structure
   */
  private validatePCS(pcs: PCS): void {
    if (!pcs.pcs_id) {
      throw new ValidationError('pcs_id is required');
    }
    if (pcs.schema !== 'fractal-lba-kakeya') {
      throw new ValidationError(`Invalid schema: ${pcs.schema}`);
    }
    if (pcs.coh_star < 0 || pcs.coh_star > 1.05) {
      throw new ValidationError(`coh_star out of bounds: ${pcs.coh_star}`);
    }
    if (pcs.r < 0 || pcs.r > 1) {
      throw new ValidationError(`r out of bounds: ${pcs.r}`);
    }
    if (pcs.budget < 0 || pcs.budget > 1) {
      throw new ValidationError(`budget out of bounds: ${pcs.budget}`);
    }
  }

  /**
   * Sign PCS using Phase 1 canonical signing (8-field subset, 9-decimal rounding)
   */
  private signPCS(pcs: PCS): void {
    if (this.signingAlg === 'hmac') {
      if (!this.signingKey) {
        throw new SignatureError('HMAC key not configured');
      }

      // Create signature subset (Phase 1 spec: 8 fields, 9-decimal rounding)
      const subset = {
        budget: round9(pcs.budget),
        coh_star: round9(pcs.coh_star),
        D_hat: round9(pcs.D_hat),
        epoch: pcs.epoch,
        merkle_root: pcs.merkle_root,
        pcs_id: pcs.pcs_id,
        r: round9(pcs.r),
        shard_id: pcs.shard_id
      };

      // Canonical JSON (sorted keys, no spaces)
      const canonicalJSON = JSON.stringify(subset, Object.keys(subset).sort());

      // SHA-256 digest
      const digest = crypto.createHash('sha256').update(canonicalJSON).digest();

      // HMAC-SHA256
      const hmac = crypto.createHmac('sha256', this.signingKey);
      hmac.update(digest);
      const signature = hmac.digest();

      // Base64 encode
      pcs.sig = signature.toString('base64');
    } else {
      throw new SignatureError(`Unsupported signing algorithm: ${this.signingAlg}`);
    }
  }

  /**
   * Retry request with exponential backoff and jitter
   */
  private async retryRequest<T>(
    fn: () => Promise<T>,
    attempt: number = 0
  ): Promise<T> {
    try {
      return await fn();
    } catch (error) {
      // Don't retry validation or signature errors
      if (error instanceof ValidationError || error instanceof SignatureError) {
        throw error;
      }

      // Don't retry if max retries reached
      if (attempt >= this.maxRetries) {
        throw error;
      }

      // Exponential backoff with jitter: base_delay * 2^attempt + jitter
      const baseDelay = 1000; // 1 second
      const maxJitter = 1000; // 1 second
      const delay = baseDelay * Math.pow(2, attempt) + Math.random() * maxJitter;

      // Wait before retrying
      await new Promise(resolve => setTimeout(resolve, delay));

      // Retry
      return this.retryRequest(fn, attempt + 1);
    }
  }
}

/**
 * Round a number to 9 decimal places (Phase 1 spec)
 */
function round9(x: number): number {
  return Math.round(x * 1e9) / 1e9;
}
