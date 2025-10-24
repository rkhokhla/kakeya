#!/usr/bin/env python3
"""Comprehensive tests for signal computer module."""

import sys
import json
import logging
import numpy as np
from pathlib import Path

# Add agent src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "agent" / "src"))

from signal_computer import SignalComputer, SignalComputationError, create_signal_computer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_intrinsic_dimensionality():
    """Test D̂ computation with various embedding configurations."""
    logger.info("\n=== Testing Intrinsic Dimensionality (D̂) ===")

    computer = create_signal_computer()

    # Test 1: Random embeddings (high dimensional)
    logger.info("Test 1: Random embeddings (should have high D̂)")
    random_embeddings = np.random.randn(100, 768)
    D_hat, valid = computer._compute_intrinsic_dimensionality(random_embeddings)
    logger.info(f"  D̂={D_hat:.2f}, valid={valid}")
    assert D_hat > 10, f"Random data should have high D̂, got {D_hat}"
    assert valid, "Should be valid for 100 samples"

    # Test 2: Low-rank embeddings (low dimensional)
    logger.info("Test 2: Low-rank embeddings (should have low D̂)")
    # Create embeddings that lie on a 3D plane
    basis = np.random.randn(3, 768)
    coeffs = np.random.randn(100, 3)
    low_rank_embeddings = coeffs @ basis
    D_hat, valid = computer._compute_intrinsic_dimensionality(low_rank_embeddings)
    logger.info(f"  D̂={D_hat:.2f}, valid={valid}")
    assert D_hat <= 10, f"Low-rank data should have low D̂, got {D_hat}"
    assert valid, "Should be valid"

    # Test 3: Single sample (edge case)
    logger.info("Test 3: Single sample (edge case)")
    single_embedding = np.random.randn(1, 768)
    D_hat, valid = computer._compute_intrinsic_dimensionality(single_embedding)
    logger.info(f"  D̂={D_hat:.2f}, valid={valid}")
    assert not valid, "Single sample should be invalid"

    # Test 4: Identical embeddings (zero variance)
    logger.info("Test 4: Identical embeddings (should detect zero variance)")
    identical_embeddings = np.ones((10, 768))
    D_hat, valid = computer._compute_intrinsic_dimensionality(identical_embeddings)
    logger.info(f"  D̂={D_hat:.2f}, valid={valid}")
    assert D_hat == 0.0, "Identical embeddings should have D̂=0"
    assert not valid, "Zero variance should be invalid"

    logger.info("✅ All D̂ tests passed!\n")


def test_coherence():
    """Test coh★ computation with various embedding patterns."""
    logger.info("=== Testing Coherence (coh★) ===")

    computer = create_signal_computer()

    # Test 1: Identical embeddings (perfect coherence)
    logger.info("Test 1: Identical embeddings (should have coh★≈1.0)")
    identical = np.ones((10, 768))
    coh_star, valid = computer._compute_coherence(identical)
    logger.info(f"  coh★={coh_star:.4f}, valid={valid}")
    assert coh_star > 0.99, f"Identical should have coh★≈1.0, got {coh_star}"

    # Test 2: Orthogonal embeddings (low coherence)
    logger.info("Test 2: Orthogonal embeddings (should have low coh★)")
    # Create orthonormal vectors
    orthogonal = np.eye(10, 768)
    coh_star, valid = computer._compute_coherence(orthogonal)
    logger.info(f"  coh★={coh_star:.4f}, valid={valid}")
    assert coh_star < 0.1, f"Orthogonal should have low coh★, got {coh_star}"
    assert valid, "Should be valid"

    # Test 3: Similar but not identical (moderate coherence)
    logger.info("Test 3: Similar embeddings (moderate coh★)")
    base = np.random.randn(768)
    similar = base[np.newaxis, :] + np.random.randn(10, 768) * 0.1
    # Normalize to unit length
    similar = similar / np.linalg.norm(similar, axis=1, keepdims=True)
    coh_star, valid = computer._compute_coherence(similar)
    logger.info(f"  coh★={coh_star:.4f}, valid={valid}")
    assert 0.5 < coh_star < 1.0, f"Similar should have moderate coh★, got {coh_star}"
    assert valid, "Should be valid"

    # Test 4: Single sample (edge case)
    logger.info("Test 4: Single sample (edge case)")
    single = np.random.randn(1, 768)
    coh_star, valid = computer._compute_coherence(single)
    logger.info(f"  coh★={coh_star:.4f}, valid={valid}")
    assert not valid, "Single sample should be invalid"

    # Test 5: Random embeddings (low to moderate coherence)
    logger.info("Test 5: Random embeddings (should have low-moderate coh★)")
    random = np.random.randn(20, 768)
    # Normalize
    random = random / np.linalg.norm(random, axis=1, keepdims=True)
    coh_star, valid = computer._compute_coherence(random)
    logger.info(f"  coh★={coh_star:.4f}, valid={valid}")
    assert 0.0 <= coh_star <= 0.3, f"Random should have low coh★, got {coh_star}"
    assert valid, "Should be valid"

    logger.info("✅ All coh★ tests passed!\n")


def test_lz_complexity():
    """Test r_LZ computation with various text patterns."""
    logger.info("=== Testing Lempel-Ziv Complexity (r_LZ) ===")

    computer = create_signal_computer()

    # Test 1: Highly repetitive text (low r_LZ)
    logger.info("Test 1: Repetitive text (should have low r_LZ)")
    repetitive = "hello " * 1000
    r_LZ, valid = computer._compute_lz_complexity(repetitive)
    logger.info(f"  r_LZ={r_LZ:.4f}, valid={valid}")
    assert r_LZ < 0.1, f"Repetitive text should have low r_LZ, got {r_LZ}"
    assert valid, "Should be valid"

    # Test 2: Random text (high r_LZ)
    logger.info("Test 2: Random text (should have high r_LZ)")
    random_text = ''.join(np.random.choice(list('abcdefghijklmnopqrstuvwxyz '), 1000))
    r_LZ, valid = computer._compute_lz_complexity(random_text)
    logger.info(f"  r_LZ={r_LZ:.4f}, valid={valid}")
    assert r_LZ > 0.5, f"Random text should have high r_LZ, got {r_LZ}"
    assert valid, "Should be valid"

    # Test 3: Natural text (moderate r_LZ)
    logger.info("Test 3: Natural text (should have moderate r_LZ)")
    # Use varied natural text, not repetition
    natural = """
    Artificial intelligence has made significant progress in recent years,
    particularly in the field of natural language processing. Large language
    models demonstrate remarkable capabilities in understanding and generating
    human-like text. However, these systems still face challenges with
    factual accuracy and can produce hallucinations. Researchers are actively
    working on methods to detect and mitigate these issues. The evaluation
    of AI systems requires careful consideration of multiple factors including
    coherence, relevance, and truthfulness of generated outputs.
    """ * 3  # Repeat a few times to get enough length
    r_LZ, valid = computer._compute_lz_complexity(natural)
    logger.info(f"  r_LZ={r_LZ:.4f}, valid={valid}")
    assert 0.05 < r_LZ < 0.9, f"Natural text should have moderate r_LZ, got {r_LZ}"
    assert valid, "Should be valid"

    # Test 4: Short text (edge case)
    logger.info("Test 4: Short text (should be invalid)")
    short = "hi"
    r_LZ, valid = computer._compute_lz_complexity(short)
    logger.info(f"  r_LZ={r_LZ:.4f}, valid={valid}")
    assert not valid, "Short text should be invalid"

    # Test 5: Unicode text
    logger.info("Test 5: Unicode text (should handle properly)")
    unicode_text = "Hello 世界! " * 100
    r_LZ, valid = computer._compute_lz_complexity(unicode_text)
    logger.info(f"  r_LZ={r_LZ:.4f}, valid={valid}")
    assert 0.0 <= r_LZ <= 1.0, f"Unicode text should have valid r_LZ, got {r_LZ}"
    assert valid, "Should be valid"

    logger.info("✅ All r_LZ tests passed!\n")


def test_compute_all_signals():
    """Test computing all signals together."""
    logger.info("=== Testing compute_all_signals() ===")

    computer = create_signal_computer()

    # Test with realistic data
    logger.info("Test 1: Realistic embeddings and text")
    embeddings = np.random.randn(10, 768)
    # Normalize to unit length (like real embeddings)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    text = "This is a realistic sentence with normal English text. " * 20

    result = computer.compute_all_signals(embeddings, text)

    logger.info(f"  D̂={result.D_hat:.4f} (valid={result.D_hat_valid})")
    logger.info(f"  coh★={result.coh_star:.4f} (valid={result.coh_star_valid})")
    logger.info(f"  r_LZ={result.r_LZ:.4f} (valid={result.r_LZ_valid})")
    logger.info(f"  Overall valid: {result.is_valid()}")
    logger.info(f"  Warnings: {result.warnings}")

    # Basic sanity checks
    assert not np.isnan(result.D_hat), "D̂ should not be NaN"
    assert not np.isnan(result.coh_star), "coh★ should not be NaN"
    assert not np.isnan(result.r_LZ), "r_LZ should not be NaN"
    assert 0.0 <= result.coh_star <= 1.0, f"coh★ out of range: {result.coh_star}"
    assert 0.0 <= result.r_LZ <= 1.0, f"r_LZ out of range: {result.r_LZ}"
    assert result.D_hat > 0, f"D̂ should be positive: {result.D_hat}"

    # Test dictionary conversion
    result_dict = result.to_dict()
    assert 'D_hat' in result_dict
    assert 'coh_star' in result_dict
    assert 'r_LZ' in result_dict
    assert 'valid' in result_dict

    logger.info("✅ All compute_all_signals tests passed!\n")


def test_error_handling():
    """Test error handling for invalid inputs."""
    logger.info("=== Testing Error Handling ===")

    computer = create_signal_computer()

    # Test 1: NaN embeddings
    logger.info("Test 1: NaN embeddings (should raise error)")
    try:
        nan_embeddings = np.array([[1.0, 2.0, np.nan]])
        computer.compute_all_signals(nan_embeddings, "test")
        assert False, "Should have raised SignalComputationError"
    except SignalComputationError as e:
        logger.info(f"  ✅ Caught expected error: {e}")

    # Test 2: Inf embeddings
    logger.info("Test 2: Inf embeddings (should raise error)")
    try:
        inf_embeddings = np.array([[1.0, 2.0, np.inf]])
        computer.compute_all_signals(inf_embeddings, "test")
        assert False, "Should have raised SignalComputationError"
    except SignalComputationError as e:
        logger.info(f"  ✅ Caught expected error: {e}")

    # Test 3: Empty embeddings
    logger.info("Test 3: Empty embeddings (should raise error)")
    try:
        empty_embeddings = np.array([])
        computer.compute_all_signals(empty_embeddings, "test")
        assert False, "Should have raised SignalComputationError"
    except SignalComputationError as e:
        logger.info(f"  ✅ Caught expected error: {e}")

    # Test 4: Empty text
    logger.info("Test 4: Empty text (should raise error)")
    try:
        valid_embeddings = np.random.randn(10, 768)
        computer.compute_all_signals(valid_embeddings, "")
        assert False, "Should have raised SignalComputationError"
    except SignalComputationError as e:
        logger.info(f"  ✅ Caught expected error: {e}")

    # Test 5: Wrong type
    logger.info("Test 5: Wrong type (should raise error)")
    try:
        computer.compute_all_signals([1, 2, 3], "test")  # List instead of array
        assert False, "Should have raised SignalComputationError"
    except SignalComputationError as e:
        logger.info(f"  ✅ Caught expected error: {e}")

    logger.info("✅ All error handling tests passed!\n")


def test_with_real_embedding():
    """Test with actual embedding from extraction phase."""
    logger.info("=== Testing with Real Embedding ===")

    # Load a real embedding
    import glob
    npz_files = glob.glob("data/embeddings/gpt2/truthfulqa/*.npz")
    if not npz_files:
        logger.warning("No real embeddings found, skipping test")
        return

    # Load first embedding
    embedding_file = npz_files[0]
    data = np.load(embedding_file)
    embedding = data['embedding']

    logger.info(f"Loaded embedding from: {Path(embedding_file).name}")
    logger.info(f"  Shape: {embedding.shape}")
    logger.info(f"  Norm: {np.linalg.norm(embedding):.4f}")

    # Load corresponding text
    sample_id = Path(embedding_file).stem
    with open("data/llm_outputs/truthfulqa_outputs.jsonl", 'r') as f:
        for line in f:
            sample = json.loads(line)
            if sample['id'] == sample_id:
                text = sample['llm_response']
                break
        else:
            logger.warning(f"Could not find text for {sample_id}")
            return

    logger.info(f"  Text length: {len(text)} chars")
    logger.info(f"  Text preview: {text[:100]}...")

    # Compute signals
    computer = create_signal_computer()
    result = computer.compute_all_signals(embedding, text)

    logger.info(f"\n  Computed signals:")
    logger.info(f"    D̂={result.D_hat:.4f} (valid={result.D_hat_valid})")
    logger.info(f"    coh★={result.coh_star:.4f} (valid={result.coh_star_valid})")
    logger.info(f"    r_LZ={result.r_LZ:.4f} (valid={result.r_LZ_valid})")
    logger.info(f"    Overall valid: {result.is_valid()}")

    if result.warnings:
        logger.info(f"    Warnings: {result.warnings}")

    # Basic sanity checks
    assert not np.isnan(result.D_hat), "D̂ should not be NaN"
    assert not np.isnan(result.coh_star), "coh★ should not be NaN"
    assert not np.isnan(result.r_LZ), "r_LZ should not be NaN"

    logger.info("\n✅ Real embedding test passed!\n")


if __name__ == '__main__':
    try:
        logger.info("="*60)
        logger.info("Starting SignalComputer Test Suite")
        logger.info("="*60)

        test_intrinsic_dimensionality()
        test_coherence()
        test_lz_complexity()
        test_compute_all_signals()
        test_error_handling()
        test_with_real_embedding()

        print("\n" + "="*60)
        print("🎉 ALL TESTS PASSED! Signal computer is production-ready.")
        print("="*60)

    except Exception as e:
        logger.error(f"\n❌ Tests failed: {e}", exc_info=True)
        sys.exit(1)
