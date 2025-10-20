#!/usr/bin/env python3
"""
Ed25519 keypair generator for PCS signing (CLAUDE_PHASE2 WP2).

Generates Ed25519 keypair and outputs base64-encoded keys suitable for:
- Agent: Private key for signing PCS
- Backend: Public key for verification

Usage:
    python3 scripts/ed25519-keygen.py

Output:
    - Private key (base64): For agent Secret
    - Public key (base64): For backend ConfigMap/Secret

Security:
    - Private key must be stored in Kubernetes Secret
    - Public key can be in ConfigMap (not sensitive)
    - Never commit private keys to version control
"""

import sys
import base64

try:
    from cryptography.hazmat.primitives.asymmetric import ed25519
    from cryptography.hazmat.primitives import serialization
except ImportError:
    print("Error: cryptography library not installed", file=sys.stderr)
    print("Install with: pip install cryptography", file=sys.stderr)
    sys.exit(1)


def generate_keypair():
    """Generate Ed25519 keypair and return base64-encoded strings."""
    # Generate private key
    private_key = ed25519.Ed25519PrivateKey.generate()

    # Derive public key
    public_key = private_key.public_key()

    # Serialize to raw bytes
    private_bytes = private_key.private_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PrivateFormat.Raw,
        encryption_algorithm=serialization.NoEncryption()
    )

    public_bytes = public_key.public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw
    )

    # Encode as base64
    private_b64 = base64.b64encode(private_bytes).decode('utf-8')
    public_b64 = base64.b64encode(public_bytes).decode('utf-8')

    return private_b64, public_b64


def print_kubernetes_manifests(private_b64, public_b64):
    """Print Kubernetes Secret and ConfigMap manifests."""
    print("\n" + "=" * 70)
    print("Agent Secret (Private Key)")
    print("=" * 70)
    print(f"""
apiVersion: v1
kind: Secret
metadata:
  name: pcs-ed25519-agent-secret
  namespace: fractal-lba
type: Opaque
stringData:
  PCS_ED25519_PRIV_B64: "{private_b64}"
""")

    print("\n" + "=" * 70)
    print("Backend ConfigMap (Public Key)")
    print("=" * 70)
    print(f"""
apiVersion: v1
kind: ConfigMap
metadata:
  name: pcs-ed25519-config
  namespace: fractal-lba
data:
  PCS_ED25519_PUB_B64: "{public_b64}"
""")


def print_helm_values(public_b64):
    """Print Helm values snippet."""
    print("\n" + "=" * 70)
    print("Helm Values Snippet")
    print("=" * 70)
    print(f"""
backend:
  env:
    PCS_SIGN_ALG: ed25519
    PCS_ED25519_PUB_B64: "{public_b64}"
""")


def print_docker_compose_env(private_b64, public_b64):
    """Print Docker Compose environment snippet."""
    print("\n" + "=" * 70)
    print("Docker Compose Environment")
    print("=" * 70)
    print(f"""
# Backend
PCS_SIGN_ALG=ed25519
PCS_ED25519_PUB_B64={public_b64}

# Agent
PCS_SIGN_ALG=ed25519
PCS_ED25519_PRIV_B64={private_b64}
""")


def main():
    print("Generating Ed25519 keypair...")
    private_b64, public_b64 = generate_keypair()

    print("\n" + "=" * 70)
    print("✓ Ed25519 Keypair Generated")
    print("=" * 70)
    print(f"\nPrivate Key (base64, 44 chars): {private_b64}")
    print(f"Public Key  (base64, 44 chars): {public_b64}")

    print("\n" + "=" * 70)
    print("SECURITY WARNING")
    print("=" * 70)
    print("""
⚠️  The private key above must be kept SECRET!

✅ DO:
   - Store in Kubernetes Secret
   - Use SOPS/age encryption for GitOps
   - Rotate keys periodically (90 days recommended)
   - Use different keys per environment

❌ DON'T:
   - Commit to version control
   - Log in plaintext
   - Share via Slack/email
   - Reuse across environments
""")

    # Print Kubernetes manifests
    print_kubernetes_manifests(private_b64, public_b64)

    # Print Helm values
    print_helm_values(public_b64)

    # Print Docker Compose env
    print_docker_compose_env(private_b64, public_b64)

    print("\n" + "=" * 70)
    print("Next Steps")
    print("=" * 70)
    print("""
1. Save private key to Secret (agent):
   kubectl create secret generic pcs-ed25519-agent-secret \\
     --from-literal=PCS_ED25519_PRIV_B64="{private}" \\
     --namespace fractal-lba

2. Save public key to ConfigMap (backend):
   kubectl create configmap pcs-ed25519-config \\
     --from-literal=PCS_ED25519_PUB_B64="{public}" \\
     --namespace fractal-lba

3. Update Helm values or deployment to use Ed25519

4. Test signing:
   python3 -c "from agent.src.utils import signing; print('Ed25519 ready!')"

5. Verify backend accepts signed PCS

For key rotation, see: docs/operations/key-rotation.md
""".format(private=private_b64, public=public_b64))


if __name__ == '__main__':
    main()
