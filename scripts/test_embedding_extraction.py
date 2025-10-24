#!/usr/bin/env python3
"""Test embedding extraction with a small sample."""

import sys
import json
import logging
import numpy as np
from pathlib import Path

# Add agent src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "agent" / "src"))

from embedding_extractor import create_extractor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_extraction():
    """Test embedding extraction with sample texts."""
    logger.info("=== Testing Embedding Extraction ===\n")

    # Test texts
    test_texts = [
        "The capital of France is Paris.",
        "Machine learning is a subset of artificial intelligence.",
        "The quick brown fox jumps over the lazy dog.",
    ]

    # Test each model
    models = ['gpt2', 'roberta-base', 'bert-base-uncased']

    for model_name in models:
        logger.info(f"Testing {model_name}...")

        try:
            # Create extractor
            extractor = create_extractor(
                model_name=model_name,
                pooling='mean',
                device='cpu',  # Use CPU for testing
                batch_size=8,
            )

            # Get model info
            info = extractor.get_model_info()
            logger.info(f"  Model info: {json.dumps(info, indent=2)}")

            # Extract embeddings
            embeddings = extractor.extract_batch(test_texts)

            # Validate
            logger.info(f"  Output shape: {embeddings.shape}")
            logger.info(f"  Expected: (3, {info['embedding_dim']})")

            assert embeddings.shape == (3, info['embedding_dim']), \
                f"Shape mismatch: {embeddings.shape}"

            # Check no NaN/Inf
            import numpy as np
            assert not np.isnan(embeddings).any(), "Contains NaN"
            assert not np.isinf(embeddings).any(), "Contains Inf"

            # Check normalized (if normalization enabled)
            if info['normalize']:
                norms = np.linalg.norm(embeddings, axis=1)
                logger.info(f"  Norms (should be ~1.0): {norms}")
                assert np.allclose(norms, 1.0, atol=1e-5), \
                    f"Not normalized: {norms}"

            logger.info(f"  ✅ {model_name} passed!\n")

        except Exception as e:
            logger.error(f"  ❌ {model_name} failed: {e}\n")
            raise

    logger.info("=== All tests passed! ===")


def test_with_real_sample():
    """Test with actual LLM output."""
    logger.info("\n=== Testing with Real LLM Output ===\n")

    # Load first sample from TruthfulQA
    input_file = "data/llm_outputs/truthfulqa_outputs.jsonl"

    with open(input_file, 'r') as f:
        sample = json.loads(f.readline())

    logger.info(f"Sample ID: {sample['id']}")
    logger.info(f"Text length: {len(sample['llm_response'])} chars")
    logger.info(f"Text preview: {sample['llm_response'][:100]}...\n")

    # Extract embedding with GPT-2
    extractor = create_extractor(model_name='gpt2', device='cpu')
    embedding = extractor.extract_single(sample['llm_response'])

    logger.info(f"Embedding shape: {embedding.shape}")
    logger.info(f"Embedding norm: {np.linalg.norm(embedding):.4f}")
    logger.info(f"First 10 values: {embedding[:10]}")

    # Validate
    assert embedding.shape == (768,), f"Wrong shape: {embedding.shape}"
    assert not np.isnan(embedding).any(), "Contains NaN"
    assert not np.isinf(embedding).any(), "Contains Inf"
    assert np.allclose(np.linalg.norm(embedding), 1.0, atol=1e-5), "Not normalized"

    logger.info("\n✅ Real sample test passed!")


if __name__ == '__main__':
    try:
        test_extraction()
        test_with_real_sample()
        print("\n" + "="*50)
        print("🎉 ALL TESTS PASSED! System is ready for extraction.")
        print("="*50)
    except Exception as e:
        print(f"\n❌ Tests failed: {e}")
        sys.exit(1)
