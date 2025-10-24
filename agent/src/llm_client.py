"""
LLM Client for generating responses and extracting embeddings.

Supports:
- OpenAI GPT-4 for text generation
- Token-level hidden states extraction via transformers
- Rate limiting and retry logic
- Cost tracking
"""

import os
import time
import json
from typing import List, Dict, Any, Optional, Tuple
import logging

# Always import numpy (needed for type hints)
try:
    import numpy as np
except ImportError:
    # Fallback for type hints only
    np = Any

# OpenAI API (for generation)
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logging.warning("OpenAI library not installed. Install with: pip install openai")

# HuggingFace transformers (for embeddings)
try:
    from transformers import AutoTokenizer, AutoModel
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logging.warning("Transformers library not installed. Install with: pip install transformers torch")


class LLMClient:
    """
    Client for LLM operations: text generation and embedding extraction.

    Usage:
        client = LLMClient(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.generate(prompt="What is 2+2?")
        embeddings = client.get_token_embeddings(text="Hello world", model="local")
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        generation_model: str = "gpt-4-turbo-preview",
        embedding_model: str = "sentence-transformers/all-mpnet-base-v2",
        max_retries: int = 3,
        retry_delay: float = 2.0,
        timeout: int = 60,
    ):
        """
        Initialize LLM client.

        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            generation_model: Model for text generation (GPT-4)
            embedding_model: HuggingFace model for embeddings
            max_retries: Max retry attempts on failure
            retry_delay: Delay between retries (seconds)
            timeout: Request timeout (seconds)
        """
        self.generation_model = generation_model
        self.embedding_model = embedding_model
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.timeout = timeout

        # OpenAI client for generation
        if OPENAI_AVAILABLE:
            self.openai_client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        else:
            self.openai_client = None

        # Local transformers for embeddings
        self.tokenizer = None
        self.model = None
        self._embedding_model_loaded = False

        # Cost tracking
        self.total_cost = 0.0
        self.total_tokens = 0
        self.num_requests = 0

        # Cost per 1K tokens (approximate, as of 2024)
        self.cost_per_1k_tokens = {
            "gpt-4-turbo-preview": 0.01,  # Input
            "gpt-4": 0.03,
            "gpt-3.5-turbo": 0.0015,
        }

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 512,
        temperature: float = 0.7,
        model_override: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generate text response using OpenAI API.

        Args:
            prompt: User prompt
            system_prompt: Optional system instruction
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature (0.0-2.0)
            model_override: Override default generation model

        Returns:
            Dict with:
                - response: Generated text
                - model: Model used
                - tokens: Token counts (prompt, completion, total)
                - cost: Estimated cost in USD
        """
        if not OPENAI_AVAILABLE or self.openai_client is None:
            raise RuntimeError("OpenAI library not available or API key not set")

        model = model_override or self.generation_model

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        for attempt in range(self.max_retries):
            try:
                response = self.openai_client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    timeout=self.timeout,
                )

                # Extract response
                response_text = response.choices[0].message.content
                prompt_tokens = response.usage.prompt_tokens
                completion_tokens = response.usage.completion_tokens
                total_tokens = response.usage.total_tokens

                # Estimate cost
                cost_per_1k = self.cost_per_1k_tokens.get(model, 0.01)
                cost = (total_tokens / 1000.0) * cost_per_1k

                # Update tracking
                self.total_cost += cost
                self.total_tokens += total_tokens
                self.num_requests += 1

                return {
                    "response": response_text,
                    "model": model,
                    "tokens": {
                        "prompt": prompt_tokens,
                        "completion": completion_tokens,
                        "total": total_tokens,
                    },
                    "cost": cost,
                }

            except Exception as e:
                logging.warning(f"Generation attempt {attempt+1} failed: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (2 ** attempt))  # Exponential backoff
                else:
                    raise

    def _load_embedding_model(self):
        """Lazy load local embedding model."""
        if self._embedding_model_loaded:
            return

        if not TRANSFORMERS_AVAILABLE:
            raise RuntimeError("Transformers library not available")

        logging.info(f"Loading embedding model: {self.embedding_model}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.embedding_model)
        self.model = AutoModel.from_pretrained(self.embedding_model)
        self.model.eval()  # Set to evaluation mode
        self._embedding_model_loaded = True
        logging.info("Embedding model loaded successfully")

    def get_token_embeddings(
        self,
        text: str,
        model: str = "local",
        layer: int = -1,
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Extract token-level embeddings from text.

        Args:
            text: Input text
            model: "local" for HuggingFace or "openai" for OpenAI embeddings
            layer: Which transformer layer to use (-1 = last layer)

        Returns:
            Tuple of:
                - embeddings: numpy array [num_tokens, embedding_dim]
                - tokens: List of token strings
        """
        if model == "local":
            return self._get_local_embeddings(text, layer)
        elif model == "openai":
            return self._get_openai_embeddings(text)
        else:
            raise ValueError(f"Unknown embedding model: {model}")

    def _get_local_embeddings(
        self,
        text: str,
        layer: int = -1,
    ) -> Tuple[np.ndarray, List[str]]:
        """Extract embeddings using local HuggingFace model."""
        self._load_embedding_model()

        # Tokenize
        inputs = self.tokenizer(text, return_tensors="pt", padding=False, truncation=True)

        # Get hidden states
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states  # Tuple of (num_layers, batch_size, seq_len, hidden_dim)

            # Extract specified layer
            layer_hidden = hidden_states[layer]  # [batch_size, seq_len, hidden_dim]
            embeddings = layer_hidden.squeeze(0).numpy()  # [seq_len, hidden_dim]

        # Decode tokens
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

        return embeddings, tokens

    def _get_openai_embeddings(self, text: str) -> Tuple[np.ndarray, List[str]]:
        """
        Get embeddings from OpenAI API.

        Note: OpenAI embeddings API returns sentence-level embeddings, not token-level.
        For token-level trajectories, use local model instead.
        """
        if not OPENAI_AVAILABLE or self.openai_client is None:
            raise RuntimeError("OpenAI library not available")

        response = self.openai_client.embeddings.create(
            model="text-embedding-3-large",
            input=text,
        )

        embedding = np.array(response.data[0].embedding)
        # OpenAI returns single vector, not token-level
        # Return as [1, dim] array with text as single "token"
        return embedding.reshape(1, -1), [text]

    def get_stats(self) -> Dict[str, Any]:
        """Get cost and usage statistics."""
        return {
            "total_requests": self.num_requests,
            "total_tokens": self.total_tokens,
            "total_cost_usd": round(self.total_cost, 4),
            "avg_tokens_per_request": (
                self.total_tokens / self.num_requests if self.num_requests > 0 else 0
            ),
            "avg_cost_per_request": (
                self.total_cost / self.num_requests if self.num_requests > 0 else 0
            ),
        }

    def save_stats(self, filepath: str):
        """Save statistics to JSON file."""
        with open(filepath, "w") as f:
            json.dump(self.get_stats(), f, indent=2)


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Check if API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY not set. Set it with:")
        print("  export OPENAI_API_KEY='your-key-here'")

    client = LLMClient()

    # Test generation
    if OPENAI_AVAILABLE and os.getenv("OPENAI_API_KEY"):
        print("\n=== Test Generation ===")
        result = client.generate(
            prompt="What is the capital of France?",
            max_tokens=50,
            temperature=0.0,
        )
        print(f"Response: {result['response']}")
        print(f"Tokens: {result['tokens']}")
        print(f"Cost: ${result['cost']:.6f}")

    # Test embeddings
    if TRANSFORMERS_AVAILABLE:
        print("\n=== Test Embeddings ===")
        embeddings, tokens = client.get_token_embeddings(
            text="Hello world!",
            model="local",
        )
        print(f"Embeddings shape: {embeddings.shape}")
        print(f"Tokens: {tokens}")
        print(f"First token embedding (first 5 dims): {embeddings[0][:5]}")

    # Print stats
    print("\n=== Stats ===")
    print(json.dumps(client.get_stats(), indent=2))
