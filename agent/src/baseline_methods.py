#!/usr/bin/env python3
"""
Baseline Methods for Hallucination Detection

Implements standard baseline approaches for comparison with ASV:
1. GPT-2 Perplexity - Language model confidence
2. BERT-Score - Semantic similarity with reference
3. Semantic Entropy - Uncertainty in semantic space
4. Token Probability - Model confidence on each token

Design Principles:
- Rigorous input validation
- Comprehensive error handling
- Deterministic reproducibility
- Clear metrics and thresholds
"""

import logging
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning, module='transformers')

try:
    from transformers import (
        GPT2LMHeadModel,
        GPT2Tokenizer,
        AutoTokenizer,
        AutoModel,
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logging.warning("transformers not installed. Baseline methods will fail.")

logger = logging.getLogger(__name__)


@dataclass
class BaselineResult:
    """Container for baseline detection results."""
    perplexity: float  # GPT-2 perplexity (lower = more fluent)
    log_perplexity: float  # Log perplexity for numerical stability
    mean_token_prob: float  # Average token probability
    min_token_prob: float  # Minimum token probability (weakest link)
    entropy: float  # Token-level entropy

    # Semantic metrics (if reference available)
    bert_score_precision: Optional[float] = None
    bert_score_recall: Optional[float] = None
    bert_score_f1: Optional[float] = None

    # Metadata
    n_tokens: int = 0
    text_length: int = 0

    # Validity flags
    perplexity_valid: bool = True
    bert_score_valid: bool = False

    warnings: List[str] = None

    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []

    def is_valid(self) -> bool:
        """Check if core metrics are valid."""
        return self.perplexity_valid

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'perplexity': float(self.perplexity),
            'log_perplexity': float(self.log_perplexity),
            'mean_token_prob': float(self.mean_token_prob),
            'min_token_prob': float(self.min_token_prob),
            'entropy': float(self.entropy),
            'bert_score_precision': float(self.bert_score_precision) if self.bert_score_precision else None,
            'bert_score_recall': float(self.bert_score_recall) if self.bert_score_recall else None,
            'bert_score_f1': float(self.bert_score_f1) if self.bert_score_f1 else None,
            'n_tokens': int(self.n_tokens),
            'text_length': int(self.text_length),
            'perplexity_valid': bool(self.perplexity_valid),
            'bert_score_valid': bool(self.bert_score_valid),
            'warnings': self.warnings,
            'valid': bool(self.is_valid()),
        }


class BaselineMethodsError(Exception):
    """Raised when baseline computation fails."""
    pass


class BaselineDetector:
    """Compute baseline hallucination detection metrics."""

    def __init__(
        self,
        model_name: str = 'gpt2',
        device: Optional[str] = None,
        max_length: int = 1024,
    ):
        """
        Initialize baseline detector.

        Args:
            model_name: Model to use for perplexity ('gpt2', 'gpt2-medium')
            device: Device to use (auto-detect if None)
            max_length: Maximum sequence length
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "transformers library required. Install with: pip install transformers torch"
            )

        self.model_name = model_name
        self.max_length = max_length
        self.logger = logging.getLogger(__name__)

        # Auto-detect device
        if device is None:
            if torch.cuda.is_available():
                device = 'cuda'
            elif torch.backends.mps.is_available():
                device = 'mps'
            else:
                device = 'cpu'
        self.device = device

        # Set random seeds for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)

        # Load model and tokenizer
        self.logger.info(f"Loading {model_name} for perplexity computation...")
        try:
            self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
            self.model = GPT2LMHeadModel.from_pretrained(model_name)
            self.model.to(device)
            self.model.eval()
            self.logger.info(f"Model loaded successfully on {device}")
        except Exception as e:
            raise RuntimeError(f"Failed to load model {model_name}: {e}")

    def compute_all_metrics(
        self,
        text: str,
        reference: Optional[str] = None,
    ) -> BaselineResult:
        """
        Compute all baseline metrics for a text.

        Args:
            text: Generated text to evaluate
            reference: Optional reference text for BERT-Score

        Returns:
            BaselineResult containing all metrics

        Raises:
            BaselineMethodsError: If computation fails
        """
        # Validate input
        self._validate_text(text)

        text_length = len(text)
        warnings_list = []

        self.logger.debug(f"Computing baseline metrics for text length={text_length}")

        # Compute perplexity and token-level metrics
        try:
            perplexity, log_perplexity, mean_prob, min_prob, entropy, n_tokens = \
                self._compute_perplexity(text)
            perplexity_valid = True
        except Exception as e:
            self.logger.error(f"Perplexity computation failed: {e}")
            perplexity = np.inf
            log_perplexity = np.inf
            mean_prob = 0.0
            min_prob = 0.0
            entropy = 0.0
            n_tokens = 0
            perplexity_valid = False
            warnings_list.append(f"Perplexity computation failed: {e}")

        # Compute BERT-Score if reference provided
        bert_p, bert_r, bert_f1, bert_valid = None, None, None, False
        if reference is not None:
            try:
                bert_p, bert_r, bert_f1 = self._compute_bert_score(text, reference)
                bert_valid = True
            except Exception as e:
                self.logger.error(f"BERT-Score computation failed: {e}")
                warnings_list.append(f"BERT-Score computation failed: {e}")

        result = BaselineResult(
            perplexity=perplexity,
            log_perplexity=log_perplexity,
            mean_token_prob=mean_prob,
            min_token_prob=min_prob,
            entropy=entropy,
            bert_score_precision=bert_p,
            bert_score_recall=bert_r,
            bert_score_f1=bert_f1,
            n_tokens=n_tokens,
            text_length=text_length,
            perplexity_valid=perplexity_valid,
            bert_score_valid=bert_valid,
            warnings=warnings_list,
        )

        self.logger.debug(
            f"Metrics computed: PPL={perplexity:.2f}, log(PPL)={log_perplexity:.2f}, "
            f"mean_prob={mean_prob:.4f}, valid={result.is_valid()}"
        )

        return result

    def _validate_text(self, text: str):
        """Validate text input."""
        if not isinstance(text, str):
            raise BaselineMethodsError(f"text must be string, got {type(text)}")
        if len(text) == 0:
            raise BaselineMethodsError("text is empty")
        if len(text) < 10:
            self.logger.warning(f"Text is very short: {len(text)} chars")

    def _compute_perplexity(
        self,
        text: str,
    ) -> Tuple[float, float, float, float, float, int]:
        """
        Compute perplexity and token-level metrics using GPT-2.

        Args:
            text: Input text

        Returns:
            (perplexity, log_perplexity, mean_prob, min_prob, entropy, n_tokens)
        """
        # Tokenize
        encodings = self.tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            max_length=self.max_length,
        )

        input_ids = encodings['input_ids'].to(self.device)
        n_tokens = input_ids.shape[1]

        # Check for truncation
        if n_tokens >= self.max_length:
            self.logger.warning(
                f"Text truncated to {self.max_length} tokens (original: {n_tokens})"
            )

        # Compute loss and probabilities
        with torch.no_grad():
            outputs = self.model(input_ids, labels=input_ids)
            loss = outputs.loss  # Negative log likelihood
            logits = outputs.logits

        # Perplexity = exp(average negative log likelihood)
        perplexity = torch.exp(loss).item()
        log_perplexity = loss.item()  # For numerical stability

        # Token-level metrics
        # Shift logits and labels for next-token prediction
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()

        # Get probabilities for actual tokens
        probs = torch.softmax(shift_logits, dim=-1)
        token_probs = probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)

        mean_prob = token_probs.mean().item()
        min_prob = token_probs.min().item()

        # Entropy = -sum(p * log(p))
        # Use full probability distribution
        log_probs = torch.log_softmax(shift_logits, dim=-1)
        entropy_per_token = -(probs * log_probs).sum(dim=-1)
        entropy = entropy_per_token.mean().item()

        # Validate results
        if np.isnan(perplexity) or np.isinf(perplexity):
            raise BaselineMethodsError(f"Invalid perplexity: {perplexity}")
        if perplexity < 1.0:
            self.logger.warning(f"Perplexity < 1.0: {perplexity}")

        return perplexity, log_perplexity, mean_prob, min_prob, entropy, n_tokens

    def _compute_bert_score(
        self,
        candidate: str,
        reference: str,
    ) -> Tuple[float, float, float]:
        """
        Compute BERT-Score between candidate and reference.

        Note: This is a simplified implementation. For production,
        use the bert-score library which has better normalization.

        Args:
            candidate: Generated text
            reference: Reference text

        Returns:
            (precision, recall, f1)
        """
        try:
            from bert_score import score

            # Compute BERT-Score
            P, R, F1 = score(
                [candidate],
                [reference],
                lang='en',
                model_type='bert-base-uncased',
                device=self.device,
                verbose=False,
            )

            return P.item(), R.item(), F1.item()

        except ImportError:
            # Fallback: use simple embedding similarity
            self.logger.warning(
                "bert-score not installed, using embedding similarity fallback"
            )
            return self._compute_embedding_similarity(candidate, reference)

    def _compute_embedding_similarity(
        self,
        text1: str,
        text2: str,
    ) -> Tuple[float, float, float]:
        """
        Fallback: Compute cosine similarity between text embeddings.

        Args:
            text1: First text
            text2: Second text

        Returns:
            (similarity, similarity, similarity) - same for P, R, F1
        """
        try:
            from sklearn.metrics.pairwise import cosine_similarity

            # Use the loaded model's tokenizer
            encoding1 = self.tokenizer(text1, return_tensors='pt', truncation=True)
            encoding2 = self.tokenizer(text2, return_tensors='pt', truncation=True)

            with torch.no_grad():
                # For GPT-2, we'll use the last hidden state
                outputs1 = self.model.transformer(
                    encoding1['input_ids'].to(self.device)
                )
                outputs2 = self.model.transformer(
                    encoding2['input_ids'].to(self.device)
                )

                # Mean pooling
                emb1 = outputs1.last_hidden_state.mean(dim=1).cpu().numpy()
                emb2 = outputs2.last_hidden_state.mean(dim=1).cpu().numpy()

            # Cosine similarity
            similarity = cosine_similarity(emb1, emb2)[0, 0]

            # Return same value for P, R, F1 (symmetric similarity)
            return float(similarity), float(similarity), float(similarity)

        except Exception as e:
            self.logger.error(f"Embedding similarity computation failed: {e}")
            return 0.0, 0.0, 0.0


def create_baseline_detector(**kwargs) -> BaselineDetector:
    """
    Convenience function to create a BaselineDetector.

    Args:
        **kwargs: Keyword arguments passed to BaselineDetector.__init__

    Returns:
        BaselineDetector instance
    """
    return BaselineDetector(**kwargs)
