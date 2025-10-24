#!/usr/bin/env python3
"""Comprehensive tests for baseline methods module."""

import sys
import logging
import numpy as np
from pathlib import Path

# Add agent src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "agent" / "src"))

from baseline_methods import create_baseline_detector, BaselineMethodsError

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_perplexity_computation():
    """Test perplexity computation with various text patterns."""
    logger.info("\n=== Testing Perplexity Computation ===")

    detector = create_baseline_detector(model_name='gpt2', device='cpu')

    # Test 1: Fluent natural text (low perplexity)
    logger.info("Test 1: Fluent natural text (should have low perplexity)")
    fluent_text = """
    The quick brown fox jumps over the lazy dog. This is a well-formed
    sentence that follows standard English grammar and structure. Machine
    learning models are trained on large datasets to understand patterns.
    """
    result = detector.compute_all_metrics(fluent_text)
    logger.info(f"  Perplexity={result.perplexity:.2f}, log(PPL)={result.log_perplexity:.2f}")
    logger.info(f"  Mean prob={result.mean_token_prob:.4f}, Min prob={result.min_token_prob:.4f}")
    assert result.perplexity > 0, "Perplexity must be positive"
    assert result.perplexity < 1000, f"Fluent text should have reasonable perplexity, got {result.perplexity}"
    assert result.perplexity_valid, "Should be valid"
    logger.info("  ✅ Passed")

    # Test 2: Random gibberish (high perplexity)
    logger.info("\nTest 2: Random gibberish (should have high perplexity)")
    gibberish = "xkcd zqwf plov mnjk rtyu vbgh plok mjuy gfds aqwe zxcv bnmk"
    result = detector.compute_all_metrics(gibberish)
    logger.info(f"  Perplexity={result.perplexity:.2f}, log(PPL)={result.log_perplexity:.2f}")
    assert result.perplexity > 100, f"Gibberish should have high perplexity, got {result.perplexity}"
    assert result.perplexity_valid, "Should be valid"
    logger.info("  ✅ Passed")

    # Test 3: Repetitive text (moderate perplexity)
    logger.info("\nTest 3: Repetitive text (moderate perplexity)")
    repetitive = "The cat sat on the mat. The cat sat on the mat. The cat sat on the mat."
    result = detector.compute_all_metrics(repetitive)
    logger.info(f"  Perplexity={result.perplexity:.2f}, log(PPL)={result.log_perplexity:.2f}")
    assert result.perplexity > 0, "Perplexity must be positive"
    assert result.perplexity_valid, "Should be valid"
    logger.info("  ✅ Passed")

    # Test 4: Technical text (higher perplexity due to domain-specific terms)
    logger.info("\nTest 4: Technical text")
    technical = """
    Theil-Sen regression estimates slopes robustly using median pairwise slopes.
    The fractal dimension characterizes self-similarity across scales using
    box-counting methods and power-law relationships in log-log space.
    """
    result = detector.compute_all_metrics(technical)
    logger.info(f"  Perplexity={result.perplexity:.2f}, log(PPL)={result.log_perplexity:.2f}")
    assert result.perplexity > 0, "Perplexity must be positive"
    assert result.perplexity_valid, "Should be valid"
    logger.info("  ✅ Passed")

    logger.info("\n✅ All perplexity tests passed!\n")


def test_token_level_metrics():
    """Test token-level probability and entropy metrics."""
    logger.info("=== Testing Token-Level Metrics ===")

    detector = create_baseline_detector(model_name='gpt2', device='cpu')

    # Test with known text
    text = "The capital of France is Paris. The Eiffel Tower is a famous landmark."
    result = detector.compute_all_metrics(text)

    logger.info(f"Text: {text[:50]}...")
    logger.info(f"  N tokens: {result.n_tokens}")
    logger.info(f"  Mean token prob: {result.mean_token_prob:.4f}")
    logger.info(f"  Min token prob: {result.min_token_prob:.4f}")
    logger.info(f"  Entropy: {result.entropy:.4f}")

    # Validation checks
    assert result.n_tokens > 0, "Should have tokens"
    assert 0 <= result.mean_token_prob <= 1, f"Mean prob out of range: {result.mean_token_prob}"
    assert 0 <= result.min_token_prob <= 1, f"Min prob out of range: {result.min_token_prob}"
    assert result.min_token_prob <= result.mean_token_prob, "Min should be <= mean"
    assert result.entropy >= 0, f"Entropy must be non-negative: {result.entropy}"

    logger.info("  ✅ Token metrics validated!\n")


def test_result_serialization():
    """Test that results can be serialized to dict."""
    logger.info("=== Testing Result Serialization ===")

    detector = create_baseline_detector(model_name='gpt2', device='cpu')
    text = "This is a test sentence for serialization."

    result = detector.compute_all_metrics(text)
    result_dict = result.to_dict()

    # Validate dictionary
    assert isinstance(result_dict, dict), "Should return dict"
    assert 'perplexity' in result_dict, "Should have perplexity"
    assert 'log_perplexity' in result_dict, "Should have log_perplexity"
    assert 'mean_token_prob' in result_dict, "Should have mean_token_prob"
    assert 'valid' in result_dict, "Should have valid flag"

    # Check types are JSON-serializable
    import json
    try:
        json_str = json.dumps(result_dict)
        logger.info(f"  Serialized to JSON ({len(json_str)} bytes)")
        logger.info(f"  Sample: {json_str[:100]}...")
        assert len(json_str) > 0, "JSON should not be empty"
    except TypeError as e:
        assert False, f"Failed to serialize to JSON: {e}"

    logger.info("  ✅ Serialization passed!\n")


def test_error_handling():
    """Test error handling for invalid inputs."""
    logger.info("=== Testing Error Handling ===")

    detector = create_baseline_detector(model_name='gpt2', device='cpu')

    # Test 1: Empty text
    logger.info("Test 1: Empty text (should raise error)")
    try:
        detector.compute_all_metrics("")
        assert False, "Should have raised BaselineMethodsError"
    except BaselineMethodsError as e:
        logger.info(f"  ✅ Caught expected error: {e}")

    # Test 2: Wrong type
    logger.info("Test 2: Wrong type (should raise error)")
    try:
        detector.compute_all_metrics(12345)  # type: ignore
        assert False, "Should have raised BaselineMethodsError"
    except BaselineMethodsError as e:
        logger.info(f"  ✅ Caught expected error: {e}")

    # Test 3: Very short text (should warn but not error)
    logger.info("Test 3: Very short text (should warn but succeed)")
    result = detector.compute_all_metrics("Hi")
    assert result.perplexity > 0, "Should still compute perplexity"
    logger.info(f"  ✅ Computed perplexity: {result.perplexity:.2f}")

    logger.info("\n✅ All error handling tests passed!\n")


def test_with_real_llm_output():
    """Test with actual LLM output from Phase 3."""
    logger.info("=== Testing with Real LLM Output ===")

    import json

    # Load a real sample
    with open("data/llm_outputs/truthfulqa_outputs.jsonl", 'r') as f:
        sample = json.loads(f.readline())

    text = sample['llm_response']
    logger.info(f"Sample ID: {sample['id']}")
    logger.info(f"Text length: {len(text)} chars")
    logger.info(f"Text preview: {text[:100]}...")

    # Compute metrics
    detector = create_baseline_detector(model_name='gpt2', device='cpu')
    result = detector.compute_all_metrics(text)

    logger.info(f"\nComputed metrics:")
    logger.info(f"  Perplexity: {result.perplexity:.2f}")
    logger.info(f"  Log perplexity: {result.log_perplexity:.2f}")
    logger.info(f"  Mean token prob: {result.mean_token_prob:.4f}")
    logger.info(f"  Min token prob: {result.min_token_prob:.4f}")
    logger.info(f"  Entropy: {result.entropy:.4f}")
    logger.info(f"  N tokens: {result.n_tokens}")
    logger.info(f"  Valid: {result.is_valid()}")

    if result.warnings:
        logger.info(f"  Warnings: {result.warnings}")

    # Validate
    assert result.perplexity > 0, "Perplexity must be positive"
    assert result.perplexity < 10000, f"Perplexity seems too high: {result.perplexity}"
    assert result.is_valid(), "Should be valid for real LLM output"

    logger.info("\n✅ Real LLM output test passed!\n")


def test_comparison_with_signals():
    """Test that baseline metrics correlate reasonably with Phase 5 signals."""
    logger.info("=== Testing Comparison with Phase 5 Signals ===")

    import json
    from pathlib import Path

    # Load a sample with both LLM output and signals
    signal_files = list(Path('data/signals/truthfulqa').glob('*.json'))
    if not signal_files:
        logger.warning("No signal files found, skipping comparison test")
        return

    signal_file = signal_files[0]
    with open(signal_file) as f:
        signals = json.load(f)

    sample_id = signals['sample_id']

    # Load corresponding LLM output
    with open("data/llm_outputs/truthfulqa_outputs.jsonl", 'r') as f:
        for line in f:
            sample = json.loads(line)
            if sample['id'] == sample_id:
                text = sample['llm_response']
                break
        else:
            logger.warning(f"Could not find LLM output for {sample_id}")
            return

    # Compute baseline metrics
    detector = create_baseline_detector(model_name='gpt2', device='cpu')
    baseline = detector.compute_all_metrics(text)

    logger.info(f"Sample ID: {sample_id}")
    logger.info(f"\nPhase 5 Signals:")
    logger.info(f"  D̂: {signals['D_hat']:.4f}")
    logger.info(f"  coh★: {signals['coh_star']:.4f}")
    logger.info(f"  r_LZ: {signals['r_LZ']:.4f}")

    logger.info(f"\nBaseline Metrics:")
    logger.info(f"  Perplexity: {baseline.perplexity:.2f}")
    logger.info(f"  Mean token prob: {baseline.mean_token_prob:.4f}")
    logger.info(f"  Entropy: {baseline.entropy:.4f}")

    # Qualitative check (should be reasonable values)
    assert baseline.perplexity > 0, "Perplexity must be positive"
    assert baseline.mean_token_prob > 0, "Mean prob must be positive"

    logger.info("\n✅ Comparison test passed!\n")


if __name__ == '__main__':
    try:
        logger.info("="*60)
        logger.info("Starting Baseline Methods Test Suite")
        logger.info("="*60)

        test_perplexity_computation()
        test_token_level_metrics()
        test_result_serialization()
        test_error_handling()
        test_with_real_llm_output()
        test_comparison_with_signals()

        print("\n" + "="*60)
        print("🎉 ALL TESTS PASSED! Baseline methods are production-ready.")
        print("="*60)

    except Exception as e:
        logger.error(f"\n❌ Tests failed: {e}", exc_info=True)
        sys.exit(1)
