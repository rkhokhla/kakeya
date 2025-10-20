# Fractal LBA TypeScript SDK

TypeScript SDK for the Fractal LBA + Kakeya FT Stack API (Phase 4 WP5).

## Features

- ✅ **Phase 1 Canonical Signing**: 8-field subset with 9-decimal rounding, HMAC-SHA256
- ✅ **Multi-Tenant Support**: X-Tenant-Id header for multi-tenant deployments
- ✅ **Automatic Retries**: Exponential backoff with jitter (default: 3 retries)
- ✅ **Type Safety**: Full TypeScript type definitions
- ✅ **Validation**: Client-side PCS validation before submission
- ✅ **Error Handling**: Custom error types for different failure scenarios

## Installation

```bash
npm install @fractal-lba/client
```

## Quick Start

```typescript
import { FractalLBAClient, PCS } from '@fractal-lba/client';

// Create client with HMAC signing
const client = new FractalLBAClient({
  baseURL: 'https://api.example.com',
  tenantID: 'tenant1',
  signingKey: 'supersecret',
  signingAlg: 'hmac'
});

// Prepare PCS
const pcs: PCS = {
  pcs_id: '...',
  schema: 'fractal-lba-kakeya',
  version: '0.1',
  shard_id: 'shard-001',
  epoch: 1,
  attempt: 1,
  sent_at: new Date().toISOString(),
  seed: 42,
  scales: [2, 4, 8, 16, 32],
  N_j: { '2': 3, '4': 5, '8': 9, '16': 17, '32': 31 },
  coh_star: 0.73,
  v_star: [0.12, 0.98, -0.05],
  D_hat: 1.41,
  r: 0.87,
  regime: 'sticky',
  budget: 0.42,
  merkle_root: 'a'.repeat(64),
  ft: {
    outbox_seq: 1,
    degraded: false,
    fallbacks: [],
    clock_skew_ms: 0
  }
};

// Submit PCS
try {
  const result = await client.submitPCS(pcs);

  if (result.accepted) {
    console.log('✅ PCS accepted');
    console.log('Recomputed D̂:', result.recomputed_D_hat);
    console.log('Recomputed budget:', result.recomputed_budget);
  } else {
    console.log('⚠️ PCS escalated');
    console.log('Reason:', result.reason);
  }
} catch (error) {
  console.error('❌ Submission failed:', error.message);
}
```

## API Reference

### `FractalLBAClient`

#### Constructor Options

```typescript
interface ClientOptions {
  baseURL: string;       // API base URL (e.g., 'https://api.example.com')
  tenantID?: string;     // Tenant ID for multi-tenant deployments
  signingKey?: string;   // HMAC key for signature generation
  signingAlg?: 'hmac' | 'none'; // Signing algorithm (default: 'none')
  timeout?: number;      // Request timeout in ms (default: 30000)
  maxRetries?: number;   // Max retry attempts (default: 3)
}
```

#### Methods

**`submitPCS(pcs: PCS): Promise<VerifyResult>`**

Submit a Proof-of-Computation Summary.

- **Parameters**: `pcs` - The PCS to submit
- **Returns**: `VerifyResult` - Verification outcome
- **Throws**:
  - `ValidationError` - Invalid PCS structure
  - `SignatureError` - Signature generation/verification failed
  - `APIError` - API request failed

**`healthCheck(): Promise<void>`**

Check if the API is healthy.

- **Throws**: `APIError` - Health check failed

### Error Types

```typescript
FractalLBAError        // Base error class
├── ValidationError    // Invalid PCS structure
├── SignatureError     // Signature-related errors
└── APIError           // API request errors (includes statusCode, responseBody)
```

## Canonical Signing (Phase 1)

The SDK implements Phase 1 canonical signing automatically when `signingAlg: 'hmac'` is configured:

1. **8-field signature subset**: `pcs_id`, `merkle_root`, `epoch`, `shard_id`, `D_hat`, `coh_star`, `r`, `budget`
2. **9-decimal rounding**: All float fields rounded to 9 decimal places
3. **Canonical JSON**: Sorted keys, no spaces
4. **SHA-256 digest**: Hash of canonical JSON
5. **HMAC-SHA256**: HMAC of digest with signing key
6. **Base64 encoding**: Final signature encoded in base64

## Multi-Tenant Support

```typescript
const client = new FractalLBAClient({
  baseURL: 'https://api.example.com',
  tenantID: 'tenant-abc',  // X-Tenant-Id header
  signingKey: 'tenant-abc-secret',
  signingAlg: 'hmac'
});
```

## Retry Logic

The SDK automatically retries failed requests with exponential backoff and jitter:

- **Base delay**: 1 second
- **Backoff**: `base_delay * 2^attempt + random(0, 1000ms)`
- **Max retries**: 3 (configurable)
- **No retry**: Validation errors, signature errors

## Examples

### Without Signing (Development)

```typescript
const client = new FractalLBAClient({
  baseURL: 'http://localhost:8080',
  signingAlg: 'none'  // No signature
});

const result = await client.submitPCS(pcs);
```

### With Custom Timeout and Retries

```typescript
const client = new FractalLBAClient({
  baseURL: 'https://api.example.com',
  timeout: 60000,      // 60 seconds
  maxRetries: 5,       // 5 retry attempts
  signingKey: 'key',
  signingAlg: 'hmac'
});
```

### Error Handling

```typescript
try {
  const result = await client.submitPCS(pcs);
} catch (error) {
  if (error instanceof ValidationError) {
    console.error('Invalid PCS:', error.message);
  } else if (error instanceof SignatureError) {
    console.error('Signature error:', error.message);
  } else if (error instanceof APIError) {
    console.error(`API error (${error.statusCode}):`, error.responseBody);
  }
}
```

## Building

```bash
npm install
npm run build
```

## Testing

```bash
npm test
```

## License

MIT
