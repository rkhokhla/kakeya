# Fractal LBA WASM SDK

WebAssembly SDK for running Fractal LBA agents in the browser (Phase 6).

## Features

- ✅ **Browser-native**: Runs directly in modern browsers via WebAssembly
- ✅ **Zero dependencies**: Pure WASM with no external JS dependencies
- ✅ **Async/await**: Uses browser fetch API with JavaScript Promises
- ✅ **Automatic signing**: HMAC-SHA256 with Phase 1 canonical signing
- ✅ **Type-safe**: Full TypeScript bindings generated from Rust
- ✅ **Lightweight**: ~50KB gzipped WASM binary

## Installation

### Via npm

```bash
npm install fractal-lba-wasm
```

### Build from Source

```bash
# Install wasm-pack
curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh

# Build WASM module
cd sdk/wasm
wasm-pack build --target web --release

# Output in pkg/ directory
```

## Usage

### Basic Example (Vanilla JS)

```html
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Fractal LBA WASM Agent</title>
</head>
<body>
    <h1>Fractal LBA Agent (Browser)</h1>
    <button id="submit">Submit PCS</button>
    <pre id="output"></pre>

    <script type="module">
        import init, { FractalLBAClient } from './pkg/fractal_lba_wasm.js';

        async function main() {
            // Initialize WASM module
            await init();

            // Create client
            const client = new FractalLBAClient(
                "https://api.example.com/v1/pcs/submit",
                "tenant-001", // tenant_id (optional)
                "supersecret" // signing key
            );

            // Create PCS
            const pcs = {
                pcs_id: "a".repeat(64),
                schema: "fractal-lba-kakeya",
                version: "0.6",
                shard_id: "shard-001",
                epoch: 1,
                attempt: 1,
                sent_at: new Date().toISOString(),
                seed: 42,
                scales: [2, 4, 8],
                n_j: { "2": 3, "4": 5, "8": 9 },
                coh_star: 0.73,
                v_star: [0.12, 0.98, -0.05],
                d_hat: 1.41,
                r: 0.87,
                regime: "sticky",
                budget: 0.42,
                merkle_root: "b".repeat(64),
                ft: {
                    outbox_seq: 123,
                    degraded: false,
                    fallbacks: [],
                    clock_skew_ms: 0
                }
            };

            // Submit (automatic signing + submission)
            document.getElementById('submit').addEventListener('click', async () => {
                try {
                    const result = await client.submit_pcs(pcs);
                    document.getElementById('output').textContent = JSON.stringify(result, null, 2);
                    console.log("PCS accepted:", result.accepted);
                } catch (err) {
                    console.error("Submission failed:", err);
                    document.getElementById('output').textContent = "Error: " + err;
                }
            });
        }

        main();
    </script>
</body>
</html>
```

### TypeScript Example

```typescript
import init, { FractalLBAClient, PCS } from 'fractal-lba-wasm';

async function submitPCS() {
    // Initialize WASM
    await init();

    // Create client
    const client = new FractalLBAClient(
        "https://api.example.com/v1/pcs/submit",
        "tenant-001",
        "supersecret"
    );

    // Create PCS
    const pcs: PCS = {
        pcs_id: "a".repeat(64),
        schema: "fractal-lba-kakeya",
        version: "0.6",
        shard_id: "shard-001",
        epoch: 1,
        attempt: 1,
        sent_at: new Date().toISOString(),
        seed: 42,
        scales: [2, 4, 8],
        n_j: { "2": 3, "4": 5, "8": 9 },
        coh_star: 0.73,
        v_star: [0.12, 0.98, -0.05],
        d_hat: 1.41,
        r: 0.87,
        regime: "sticky",
        budget: 0.42,
        merkle_root: "b".repeat(64),
        ft: {
            outbox_seq: 123,
            degraded: false,
            fallbacks: [],
            clock_skew_ms: 0
        }
    };

    // Submit
    try {
        const result = await client.submit_pcs(pcs);
        console.log(`Accepted: ${result.accepted}, Budget: ${result.budget_computed}`);
    } catch (err) {
        console.error(`Submission failed: ${err}`);
    }
}
```

### Health Check

```javascript
const healthy = await client.health_check();
console.log("Backend healthy:", healthy);
```

### Utility Functions

```javascript
import { sha256, compute_merkle_root } from 'fractal-lba-wasm';

// SHA-256 hash
const hash = sha256("Hello, World!");
console.log(hash); // "dffd6021bb2bd5b0af676290809ec3a53191dd81c7f70a4b28688a362182986f"

// Merkle root
const leaves = ["hash1", "hash2", "hash3", "hash4"];
const root = compute_merkle_root(leaves);
console.log(root);
```

## Browser Compatibility

- ✅ Chrome 57+
- ✅ Firefox 52+
- ✅ Safari 11+
- ✅ Edge 16+

Requires:
- WebAssembly support
- Fetch API
- Promises

## Performance

- **WASM module size**: ~50KB gzipped
- **Initialization time**: ~10ms
- **Signing latency**: ~1ms (browser-native crypto)
- **Memory overhead**: ~500KB (WASM linear memory)

## Use Cases

### 1. Browser-Based Agents

Run PCS computation directly in the browser for client-side workloads:

```javascript
// Compute signals from browser event stream
const events = captureUserInteractions();
const signals = computeSignals(events);
const pcs = buildPCS(signals);

// Submit to backend
const result = await client.submit_pcs(pcs);
```

### 2. Edge Workers (Cloudflare, Vercel)

Deploy agents at the edge:

```javascript
// Cloudflare Worker
export default {
    async fetch(request, env) {
        await init();
        const client = new FractalLBAClient(env.API_ENDPOINT, null, env.SIGNING_KEY);
        // ... compute and submit PCS
    }
}
```

### 3. Service Workers

Background PCS computation:

```javascript
// service-worker.js
self.addEventListener('message', async (event) => {
    const pcs = event.data.pcs;
    const result = await client.submit_pcs(pcs);
    event.ports[0].postMessage(result);
});
```

## Security Considerations

### CORS

The backend must allow CORS requests:

```http
Access-Control-Allow-Origin: https://yourdomain.com
Access-Control-Allow-Methods: POST, OPTIONS
Access-Control-Allow-Headers: Content-Type, X-Tenant-Id
```

### Secret Management

**Never hardcode signing keys in client-side code.** Use environment variables or secure key management:

```javascript
// Good: fetch key from secure backend
const signingKey = await fetchSigningKeyFromBackend(userId);
const client = new FractalLBAClient(endpoint, tenantId, signingKey);

// Bad: hardcoded key (visible in browser)
const client = new FractalLBAClient(endpoint, tenantId, "supersecret"); // ❌
```

### Content Security Policy

Add to your HTML:

```html
<meta http-equiv="Content-Security-Policy" content="
    default-src 'self';
    connect-src https://api.example.com;
    script-src 'self' 'wasm-unsafe-eval';
">
```

## Testing

```bash
# Install wasm-pack and test dependencies
cargo install wasm-pack

# Run tests in headless browser
wasm-pack test --headless --firefox
wasm-pack test --headless --chrome
```

## Build Optimization

For production, enable optimizations:

```toml
# Cargo.toml
[profile.release]
opt-level = "z"    # Optimize for size
lto = true         # Link-time optimization
codegen-units = 1  # Single codegen unit for smaller binary
```

Build command:

```bash
wasm-pack build --target web --release --out-dir pkg
```

## License

MIT
