#!/usr/bin/env python3
"""
Embedding Extractor for LLM Outputs

Extracts token-level embeddings from LLM-generated text using transformer models.
Supports GPT-2, RoBERTa, and BERT with proper error handling and validation.

Design Principles:
- Rigorous input/output validation
- Comprehensive error handling
- Memory-efficient batch processing
- Deterministic reproducibility (fixed random seeds)
- Clear logging and progress tracking
"""

import logging
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import warnings

# Suppress transformer warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning, module='transformers')

try:
    from transformers import (
        AutoTokenizer,
        AutoModel,
        GPT2Tokenizer,
        GPT2Model,
        RobertaTokenizer,
        RobertaModel,
        BertTokenizer,
        BertModel,
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logging.warning("transformers not installed. Embedding extraction will fail.")


@dataclass
class EmbeddingConfig:
    """Configuration for embedding extraction."""
    model_name: str
    max_length: int  # Maximum sequence length
    pooling_strategy: str  # 'mean', 'cls', 'last'
    batch_size: int  # Number of texts to process at once
    device: str  # 'cuda', 'mps', or 'cpu'
    normalize: bool  # Whether to L2-normalize embeddings
    truncate: bool  # Whether to truncate long sequences

    def __post_init__(self):
        """Validate configuration after initialization."""
        # Validate model_name
        valid_models = ['gpt2', 'roberta-base', 'bert-base-uncased']
        if self.model_name not in valid_models:
            raise ValueError(
                f"Invalid model_name '{self.model_name}'. "
                f"Must be one of {valid_models}"
            )

        # Validate max_length
        if self.max_length <= 0:
            raise ValueError(f"max_length must be positive, got {self.max_length}")
        if self.max_length > 2048:
            logging.warning(f"max_length={self.max_length} is very large, may cause OOM")

        # Validate pooling_strategy
        valid_pooling = ['mean', 'cls', 'last']
        if self.pooling_strategy not in valid_pooling:
            raise ValueError(
                f"Invalid pooling_strategy '{self.pooling_strategy}'. "
                f"Must be one of {valid_pooling}"
            )

        # Validate cls pooling only for BERT/RoBERTa
        if self.pooling_strategy == 'cls' and self.model_name == 'gpt2':
            raise ValueError("pooling_strategy='cls' not supported for GPT-2 (no CLS token)")

        # Validate batch_size
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if self.batch_size > 128:
            logging.warning(f"batch_size={self.batch_size} is large, may cause OOM")

        # Validate device
        valid_devices = ['cuda', 'mps', 'cpu']
        if self.device not in valid_devices:
            raise ValueError(
                f"Invalid device '{self.device}'. "
                f"Must be one of {valid_devices}"
            )


class EmbeddingExtractor:
    """Extract embeddings from text using transformer models."""

    # Model-specific configurations
    MODEL_CONFIGS = {
        'gpt2': {
            'default_max_length': 1024,
            'embedding_dim': 768,
            'has_cls_token': False,
        },
        'roberta-base': {
            'default_max_length': 512,
            'embedding_dim': 768,
            'has_cls_token': True,
        },
        'bert-base-uncased': {
            'default_max_length': 512,
            'embedding_dim': 768,
            'has_cls_token': True,
        },
    }

    def __init__(self, config: EmbeddingConfig):
        """
        Initialize embedding extractor.

        Args:
            config: EmbeddingConfig specifying model and extraction parameters

        Raises:
            ImportError: If transformers library not available
            RuntimeError: If model loading fails
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "transformers library required for embedding extraction. "
                "Install with: pip install transformers torch"
            )

        self.config = config
        self.logger = logging.getLogger(__name__)

        # Set random seeds for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)

        # Validate device availability
        self._validate_device()

        # Load model and tokenizer
        self.logger.info(f"Loading model: {config.model_name}")
        try:
            self.tokenizer = self._load_tokenizer()
            self.model = self._load_model()
            self.model.eval()  # Set to evaluation mode
            self.logger.info(f"Model loaded successfully on {self.config.device}")
        except Exception as e:
            raise RuntimeError(f"Failed to load model {config.model_name}: {e}")

        # Get expected embedding dimension
        self.embedding_dim = self.MODEL_CONFIGS[config.model_name]['embedding_dim']
        self.logger.info(f"Expected embedding dimension: {self.embedding_dim}")

    def _validate_device(self):
        """Validate that requested device is available."""
        if self.config.device == 'cuda' and not torch.cuda.is_available():
            self.logger.warning("CUDA requested but not available, falling back to CPU")
            self.config.device = 'cpu'
        elif self.config.device == 'mps' and not torch.backends.mps.is_available():
            self.logger.warning("MPS requested but not available, falling back to CPU")
            self.config.device = 'cpu'

    def _load_tokenizer(self):
        """Load tokenizer for the specified model."""
        if self.config.model_name == 'gpt2':
            tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            # GPT-2 doesn't have a pad token, use eos_token
            tokenizer.pad_token = tokenizer.eos_token
            return tokenizer
        elif self.config.model_name == 'roberta-base':
            return RobertaTokenizer.from_pretrained('roberta-base')
        elif self.config.model_name == 'bert-base-uncased':
            return BertTokenizer.from_pretrained('bert-base-uncased')
        else:
            raise ValueError(f"Unknown model: {self.config.model_name}")

    def _load_model(self):
        """Load transformer model and move to device."""
        if self.config.model_name == 'gpt2':
            model = GPT2Model.from_pretrained('gpt2')
        elif self.config.model_name == 'roberta-base':
            model = RobertaModel.from_pretrained('roberta-base')
        elif self.config.model_name == 'bert-base-uncased':
            model = BertModel.from_pretrained('bert-base-uncased')
        else:
            raise ValueError(f"Unknown model: {self.config.model_name}")

        return model.to(self.config.device)

    def extract_batch(
        self,
        texts: List[str],
        return_attention_mask: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Extract embeddings for a batch of texts.

        Args:
            texts: List of input texts
            return_attention_mask: If True, also return attention masks

        Returns:
            embeddings: numpy array of shape (batch_size, embedding_dim)
            attention_masks: (optional) numpy array of shape (batch_size, seq_len)

        Raises:
            ValueError: If texts is empty or contains invalid data
            RuntimeError: If extraction fails
        """
        # Validate input
        if not texts:
            raise ValueError("texts list is empty")
        if not all(isinstance(t, str) for t in texts):
            raise ValueError("All texts must be strings")
        if any(len(t) == 0 for t in texts):
            self.logger.warning("Some texts are empty strings")

        # Check batch size
        if len(texts) > self.config.batch_size:
            raise ValueError(
                f"Batch size {len(texts)} exceeds configured "
                f"batch_size {self.config.batch_size}"
            )

        try:
            # Tokenize
            encoding = self.tokenizer(
                texts,
                padding=True,
                truncation=self.config.truncate,
                max_length=self.config.max_length,
                return_tensors='pt',
                return_attention_mask=True,
            )

            # Move to device
            input_ids = encoding['input_ids'].to(self.config.device)
            attention_mask = encoding['attention_mask'].to(self.config.device)

            # Check for truncation
            max_input_len = input_ids.shape[1]
            if max_input_len >= self.config.max_length and self.config.truncate:
                self.logger.warning(
                    f"Some texts truncated to {self.config.max_length} tokens"
                )

            # Extract embeddings
            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                hidden_states = outputs.last_hidden_state  # (batch, seq_len, dim)

            # Apply pooling
            embeddings = self._pool_embeddings(hidden_states, attention_mask)

            # Validate embedding dimensions
            expected_shape = (len(texts), self.embedding_dim)
            if embeddings.shape != expected_shape:
                raise RuntimeError(
                    f"Embedding shape mismatch. Expected {expected_shape}, "
                    f"got {embeddings.shape}"
                )

            # Normalize if requested
            if self.config.normalize:
                embeddings = self._normalize_embeddings(embeddings)

            # Convert to numpy
            embeddings_np = embeddings.cpu().numpy()

            # Validate no NaNs or Infs
            if np.isnan(embeddings_np).any():
                raise RuntimeError("Embeddings contain NaN values")
            if np.isinf(embeddings_np).any():
                raise RuntimeError("Embeddings contain Inf values")

            if return_attention_mask:
                attention_mask_np = attention_mask.cpu().numpy()
                return embeddings_np, attention_mask_np
            else:
                return embeddings_np

        except Exception as e:
            self.logger.error(f"Embedding extraction failed: {e}")
            raise RuntimeError(f"Embedding extraction failed: {e}")

    def _pool_embeddings(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Pool token embeddings into a single vector.

        Args:
            hidden_states: (batch, seq_len, dim)
            attention_mask: (batch, seq_len)

        Returns:
            pooled: (batch, dim)
        """
        if self.config.pooling_strategy == 'mean':
            # Mean pooling with attention mask
            mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size())
            sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            return sum_embeddings / sum_mask

        elif self.config.pooling_strategy == 'cls':
            # Use [CLS] token (first token)
            return hidden_states[:, 0, :]

        elif self.config.pooling_strategy == 'last':
            # Use last token before padding
            batch_size = hidden_states.shape[0]
            sequence_lengths = attention_mask.sum(dim=1) - 1  # -1 for 0-indexing
            return hidden_states[range(batch_size), sequence_lengths, :]

        else:
            raise ValueError(f"Unknown pooling strategy: {self.config.pooling_strategy}")

    def _normalize_embeddings(self, embeddings: torch.Tensor) -> torch.Tensor:
        """L2-normalize embeddings."""
        norms = torch.norm(embeddings, p=2, dim=1, keepdim=True)
        # Avoid division by zero
        norms = torch.clamp(norms, min=1e-12)
        return embeddings / norms

    def extract_single(self, text: str) -> np.ndarray:
        """
        Extract embedding for a single text.

        Args:
            text: Input text

        Returns:
            embedding: numpy array of shape (embedding_dim,)
        """
        embedding = self.extract_batch([text])
        return embedding[0]

    def get_model_info(self) -> Dict[str, any]:
        """Get information about the loaded model."""
        return {
            'model_name': self.config.model_name,
            'embedding_dim': self.embedding_dim,
            'max_length': self.config.max_length,
            'pooling_strategy': self.config.pooling_strategy,
            'device': self.config.device,
            'normalize': self.config.normalize,
            'truncate': self.config.truncate,
        }


def create_extractor(
    model_name: str = 'gpt2',
    pooling: str = 'mean',
    device: Optional[str] = None,
    batch_size: int = 32,
) -> EmbeddingExtractor:
    """
    Convenience function to create an embedding extractor.

    Args:
        model_name: 'gpt2', 'roberta-base', or 'bert-base-uncased'
        pooling: 'mean', 'cls', or 'last'
        device: 'cuda', 'mps', or 'cpu' (auto-detected if None)
        batch_size: Batch size for processing

    Returns:
        EmbeddingExtractor instance
    """
    # Auto-detect device if not specified
    if device is None:
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'

    # Get default max_length for model
    model_config = EmbeddingExtractor.MODEL_CONFIGS.get(model_name)
    if model_config is None:
        raise ValueError(f"Unknown model: {model_name}")

    config = EmbeddingConfig(
        model_name=model_name,
        max_length=model_config['default_max_length'],
        pooling_strategy=pooling,
        batch_size=batch_size,
        device=device,
        normalize=True,  # Always normalize for downstream use
        truncate=True,   # Always truncate to avoid errors
    )

    return EmbeddingExtractor(config)
