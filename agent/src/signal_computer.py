#!/usr/bin/env python3
"""
Signal Computer for ASV Evaluation

Computes three key signals from LLM outputs:
1. D̂ (intrinsic dimensionality) - Estimates manifold dimension of embeddings
2. coh★ (coherence score) - Measures semantic consistency
3. r_LZ (Lempel-Ziv complexity ratio) - Measures text compressibility

Design Principles:
- Rigorous input validation (no garbage in)
- Comprehensive error handling
- Deterministic reproducibility
- Clear logging and error messages
- Mathematical correctness verification
"""

import logging
import numpy as np
import zlib
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import pdist, squareform
from scipy.stats import ks_2samp
import warnings

warnings.filterwarnings('ignore', category=RuntimeWarning)

logger = logging.getLogger(__name__)


@dataclass
class SignalResult:
    """Container for computed signals with validation metadata."""
    D_hat: float  # Intrinsic dimensionality
    coh_star: float  # Coherence score
    r_LZ: float  # Lempel-Ziv complexity ratio

    # Metadata for validation
    embedding_dim: int  # Original embedding dimension
    n_samples: int  # Number of embeddings used
    text_length: int  # Length of text in characters

    # Quality indicators
    D_hat_valid: bool
    coh_star_valid: bool
    r_LZ_valid: bool
    warnings: List[str]

    def is_valid(self) -> bool:
        """Check if all signals are valid."""
        return self.D_hat_valid and self.coh_star_valid and self.r_LZ_valid

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        # Convert numpy types to Python types for JSON serialization
        return {
            'D_hat': float(self.D_hat),
            'coh_star': float(self.coh_star),
            'r_LZ': float(self.r_LZ),
            'embedding_dim': int(self.embedding_dim),
            'n_samples': int(self.n_samples),
            'text_length': int(self.text_length),
            'D_hat_valid': bool(self.D_hat_valid),
            'coh_star_valid': bool(self.coh_star_valid),
            'r_LZ_valid': bool(self.r_LZ_valid),
            'warnings': self.warnings,
            'valid': bool(self.is_valid()),
        }


class SignalComputationError(Exception):
    """Raised when signal computation fails."""
    pass


class SignalComputer:
    """Compute D̂, coh★, and r_LZ signals from LLM outputs."""

    def __init__(
        self,
        min_samples_for_D_hat: int = 2,
        min_samples_for_coh_star: int = 2,
        pca_variance_threshold: float = 0.95,
        random_seed: int = 42,
    ):
        """
        Initialize signal computer.

        Args:
            min_samples_for_D_hat: Minimum samples needed for D̂ computation
            min_samples_for_coh_star: Minimum samples needed for coh★
            pca_variance_threshold: Variance explained threshold for PCA
            random_seed: Random seed for reproducibility
        """
        self.min_samples_for_D_hat = min_samples_for_D_hat
        self.min_samples_for_coh_star = min_samples_for_coh_star
        self.pca_variance_threshold = pca_variance_threshold
        self.random_seed = random_seed

        # Set random seeds
        np.random.seed(random_seed)

        logger.info(f"SignalComputer initialized with seed={random_seed}")

    def compute_all_signals(
        self,
        embeddings: np.ndarray,
        text: str,
    ) -> SignalResult:
        """
        Compute all three signals from embeddings and text.

        Args:
            embeddings: Array of shape (n_samples, embedding_dim) or (embedding_dim,)
            text: Original text string

        Returns:
            SignalResult containing all computed signals

        Raises:
            SignalComputationError: If computation fails
        """
        # Validate inputs
        self._validate_embeddings(embeddings)
        self._validate_text(text)

        # Handle single embedding case
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)

        n_samples, embedding_dim = embeddings.shape
        text_length = len(text)
        warnings_list = []

        logger.info(f"Computing signals for {n_samples} embeddings, text length={text_length}")

        # Compute D̂ (intrinsic dimensionality)
        try:
            D_hat, D_hat_valid = self._compute_intrinsic_dimensionality(embeddings)
            if not D_hat_valid:
                warnings_list.append("D̂ computation uncertain: insufficient samples or variance")
        except Exception as e:
            logger.error(f"D̂ computation failed: {e}")
            D_hat = np.nan
            D_hat_valid = False
            warnings_list.append(f"D̂ computation failed: {e}")

        # Compute coh★ (coherence)
        try:
            coh_star, coh_star_valid = self._compute_coherence(embeddings)
            if not coh_star_valid:
                warnings_list.append("coh★ computation uncertain: insufficient samples")
        except Exception as e:
            logger.error(f"coh★ computation failed: {e}")
            coh_star = np.nan
            coh_star_valid = False
            warnings_list.append(f"coh★ computation failed: {e}")

        # Compute r_LZ (Lempel-Ziv complexity)
        try:
            r_LZ, r_LZ_valid = self._compute_lz_complexity(text)
            if not r_LZ_valid:
                warnings_list.append("r_LZ computation uncertain: text too short or compression issue")
        except Exception as e:
            logger.error(f"r_LZ computation failed: {e}")
            r_LZ = np.nan
            r_LZ_valid = False
            warnings_list.append(f"r_LZ computation failed: {e}")

        result = SignalResult(
            D_hat=D_hat,
            coh_star=coh_star,
            r_LZ=r_LZ,
            embedding_dim=embedding_dim,
            n_samples=n_samples,
            text_length=text_length,
            D_hat_valid=D_hat_valid,
            coh_star_valid=coh_star_valid,
            r_LZ_valid=r_LZ_valid,
            warnings=warnings_list,
        )

        logger.info(
            f"Signals computed: D̂={D_hat:.4f}, coh★={coh_star:.4f}, r_LZ={r_LZ:.4f} "
            f"(valid={result.is_valid()})"
        )

        return result

    def _validate_embeddings(self, embeddings: np.ndarray):
        """Validate embedding array."""
        if not isinstance(embeddings, np.ndarray):
            raise SignalComputationError(
                f"embeddings must be numpy array, got {type(embeddings)}"
            )

        if embeddings.ndim not in [1, 2]:
            raise SignalComputationError(
                f"embeddings must be 1D or 2D, got shape {embeddings.shape}"
            )

        if np.isnan(embeddings).any():
            raise SignalComputationError("embeddings contain NaN values")

        if np.isinf(embeddings).any():
            raise SignalComputationError("embeddings contain Inf values")

        if embeddings.size == 0:
            raise SignalComputationError("embeddings array is empty")

    def _validate_text(self, text: str):
        """Validate text input."""
        if not isinstance(text, str):
            raise SignalComputationError(
                f"text must be string, got {type(text)}"
            )

        if len(text) == 0:
            raise SignalComputationError("text is empty")

    def _compute_intrinsic_dimensionality(
        self,
        embeddings: np.ndarray,
    ) -> Tuple[float, bool]:
        """
        Compute intrinsic dimensionality D̂ using PCA.

        Method: Fit PCA and count components needed to explain
        pca_variance_threshold of variance (default 95%).

        Args:
            embeddings: Array of shape (n_samples, embedding_dim)

        Returns:
            (D_hat, valid): Intrinsic dimension and validity flag
        """
        n_samples, embedding_dim = embeddings.shape

        # Need at least 2 samples for PCA
        if n_samples < self.min_samples_for_D_hat:
            logger.warning(
                f"Insufficient samples for D̂: {n_samples} < {self.min_samples_for_D_hat}"
            )
            return float(embedding_dim), False

        # Check for zero variance
        if np.allclose(embeddings.std(axis=0), 0.0):
            logger.warning("All embeddings are identical, D̂=0")
            return 0.0, False

        # Fit PCA
        try:
            # Use min(n_samples, embedding_dim) components
            n_components = min(n_samples, embedding_dim)
            pca = PCA(n_components=n_components, random_state=self.random_seed)
            pca.fit(embeddings)

            # Count components needed for variance threshold
            explained_variance = pca.explained_variance_ratio_
            cumulative_variance = np.cumsum(explained_variance)

            # Find first component where cumulative >= threshold
            D_hat = np.searchsorted(
                cumulative_variance,
                self.pca_variance_threshold,
            ) + 1  # +1 because searchsorted is 0-indexed

            # Clamp to valid range [1, n_components]
            D_hat = max(1, min(D_hat, n_components))

            # Validity check: at least 2 components should contribute
            valid = (
                n_samples >= self.min_samples_for_D_hat
                and explained_variance[0] < 0.999  # Not dominated by one component
            )

            logger.debug(
                f"D̂={D_hat}, explained_variance[0]={explained_variance[0]:.3f}, "
                f"cumulative[D̂-1]={cumulative_variance[D_hat-1]:.3f}"
            )

            return float(D_hat), valid

        except Exception as e:
            logger.error(f"PCA failed: {e}")
            return float(embedding_dim), False

    def _compute_coherence(
        self,
        embeddings: np.ndarray,
    ) -> Tuple[float, bool]:
        """
        Compute coherence score coh★ using mean pairwise cosine similarity.

        Method: Compute all pairwise cosine similarities and take the mean.
        Higher values indicate more coherent/similar embeddings.

        Args:
            embeddings: Array of shape (n_samples, embedding_dim)

        Returns:
            (coh_star, valid): Coherence score in [0, 1] and validity flag
        """
        n_samples = embeddings.shape[0]

        # Single sample case
        if n_samples == 1:
            logger.warning("Only 1 sample, coh★=1.0 by definition")
            return 1.0, False

        # Need at least 2 samples for pairwise similarity
        if n_samples < self.min_samples_for_coh_star:
            logger.warning(
                f"Insufficient samples for coh★: {n_samples} < {self.min_samples_for_coh_star}"
            )
            return 1.0, False

        try:
            # Compute pairwise cosine similarity
            # sklearn's cosine_similarity returns (n_samples, n_samples) matrix
            similarity_matrix = cosine_similarity(embeddings)

            # Extract upper triangle (exclude diagonal)
            # We want pairwise similarities, not self-similarities
            triu_indices = np.triu_indices(n_samples, k=1)
            pairwise_similarities = similarity_matrix[triu_indices]

            # Mean similarity
            coh_star = float(np.mean(pairwise_similarities))

            # Validity check
            valid = (
                n_samples >= self.min_samples_for_coh_star
                and not np.isnan(coh_star)
                and -1.0 <= coh_star <= 1.0
            )

            # Clamp to [0, 1] range (cosine can be negative)
            coh_star = max(0.0, min(1.0, coh_star))

            logger.debug(
                f"coh★={coh_star:.4f}, n_pairs={len(pairwise_similarities)}, "
                f"std={np.std(pairwise_similarities):.4f}"
            )

            return coh_star, valid

        except Exception as e:
            logger.error(f"Cosine similarity computation failed: {e}")
            return 0.0, False

    def _compute_lz_complexity(
        self,
        text: str,
    ) -> Tuple[float, bool]:
        """
        Compute Lempel-Ziv complexity ratio r_LZ.

        Method: Compress text with zlib (LZ77 + Huffman) and compute ratio:
        r_LZ = compressed_size / original_size

        Lower values indicate more compressible (structured) text.

        Args:
            text: Input text string

        Returns:
            (r_LZ, valid): Compression ratio in [0, 1] and validity flag
        """
        # Validate text length
        if len(text) < 10:
            logger.warning(f"Text too short for r_LZ: {len(text)} chars")
            return 1.0, False

        try:
            # Encode text to bytes
            text_bytes = text.encode('utf-8')
            original_size = len(text_bytes)

            # Compress with zlib (level 6 is default, good balance)
            compressed = zlib.compress(text_bytes, level=6)
            compressed_size = len(compressed)

            # Compute ratio
            r_LZ = compressed_size / original_size

            # Validity checks
            valid = (
                original_size >= 10  # Minimum text length
                and compressed_size > 0  # Compression succeeded
                and 0.0 <= r_LZ <= 1.5  # Reasonable range (can exceed 1.0 for random data)
            )

            # Clamp to [0, 1] for consistency
            # (random data can have r_LZ > 1 due to compression overhead)
            r_LZ = max(0.0, min(1.0, r_LZ))

            logger.debug(
                f"r_LZ={r_LZ:.4f}, original={original_size}, "
                f"compressed={compressed_size}"
            )

            return r_LZ, valid

        except Exception as e:
            logger.error(f"LZ compression failed: {e}")
            return 1.0, False


def create_signal_computer(**kwargs) -> SignalComputer:
    """
    Convenience function to create a SignalComputer.

    Args:
        **kwargs: Keyword arguments passed to SignalComputer.__init__

    Returns:
        SignalComputer instance
    """
    return SignalComputer(**kwargs)
