"""
Language Encoder for VLA Model
==============================
Processes natural language task instructions into feature vectors.

Architecture: DistilBERT
- 6 transformer layers (vs 12 for BERT)
- 66M parameters (vs 110M for BERT)
- 97% of BERT's performance at 60% the size
- Fast inference on CPU

Key Concepts:
1. Tokenization: Convert text to token IDs
2. Contextual Embeddings: Each word's meaning depends on context
3. [CLS] Token: Special token that represents the whole sentence
"""

import torch
import torch.nn as nn
from transformers import DistilBertModel, DistilBertTokenizer
from typing import List, Union, Optional


class LanguageEncoder(nn.Module):
    """
    Language encoder that extracts features from text instructions.

    Input:  Text string or list of strings
    Output: Feature vector (B, output_dim)

    Example:
        >>> encoder = LanguageEncoder(output_dim=512)
        >>> features = encoder(["pick up the red block", "push the blue cube"])
        >>> features.shape
        torch.Size([2, 512])
    """

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        output_dim: int = 512,
        freeze_backbone: bool = False,
        max_length: int = 32,
    ):
        """
        Args:
            model_name: HuggingFace model name
            output_dim: Dimension of output feature vector
            freeze_backbone: If True, freeze transformer weights
            max_length: Maximum token sequence length
        """
        super().__init__()

        self.output_dim = output_dim
        self.max_length = max_length

        # =====================================================================
        # Step 1: Load pretrained tokenizer and model
        # =====================================================================
        # Tokenizer: Converts text → token IDs
        #   "pick up the red block" → [7820, 2039, 1996, 2417, 3796]
        #
        # Model: DistilBERT transformer
        #   - 6 transformer layers
        #   - Hidden size: 768
        #   - Attention heads: 12

        print(f"[LanguageEncoder] Loading {model_name}...")
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        self.transformer = DistilBertModel.from_pretrained(model_name)

        # Get hidden dimension from model config
        self.hidden_dim = self.transformer.config.hidden_size  # 768 for DistilBERT
        print(f"[LanguageEncoder] Hidden dim: {self.hidden_dim}")

        # =====================================================================
        # Step 2: Optionally freeze transformer weights
        # =====================================================================
        if freeze_backbone:
            print("[LanguageEncoder] Freezing transformer weights")
            for param in self.transformer.parameters():
                param.requires_grad = False

        # =====================================================================
        # Step 3: Add projection layer
        # =====================================================================
        # Project from transformer hidden dim (768) to our output dim (512)
        # This lets us match the vision encoder dimension

        self.projection = nn.Sequential(
            nn.Linear(self.hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )

        # Track parameters
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[LanguageEncoder] Total params: {total_params:,}")
        print(f"[LanguageEncoder] Trainable params: {trainable_params:,}")

        # Cache for tokenized inputs (avoid re-tokenizing same text)
        self._token_cache = {}

    def tokenize(
        self,
        texts: Union[str, List[str]],
        device: Optional[torch.device] = None
    ) -> dict:
        """
        Tokenize text inputs.

        Args:
            texts: Single string or list of strings
            device: Target device for tensors

        Returns:
            Dictionary with input_ids and attention_mask tensors
        """
        if isinstance(texts, str):
            texts = [texts]

        # Check cache for single texts
        if len(texts) == 1 and texts[0] in self._token_cache:
            cached = self._token_cache[texts[0]]
            if device is not None:
                return {k: v.to(device) for k, v in cached.items()}
            return cached

        # Tokenize with padding and truncation
        # - padding='max_length': Pad all sequences to max_length
        # - truncation=True: Cut sequences longer than max_length
        # - return_tensors='pt': Return PyTorch tensors
        encoded = self.tokenizer(
            texts,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

        # Cache single text tokenizations
        if len(texts) == 1:
            self._token_cache[texts[0]] = encoded

        if device is not None:
            encoded = {k: v.to(device) for k, v in encoded.items()}

        return encoded

    def forward(
        self,
        texts: Union[str, List[str], dict],
        attention_output: bool = False
    ) -> torch.Tensor:
        """
        Extract features from text instructions.

        Args:
            texts: Text string(s) or pre-tokenized dict
            attention_output: If True, return per-token features (for cross-attention)

        Returns:
            If attention_output=False: (B, output_dim) - [CLS] token features
            If attention_output=True:  (B, seq_len, output_dim) - all token features
        """
        # Tokenize if needed
        if isinstance(texts, dict):
            encoded = texts
        else:
            # Get device from model parameters
            device = next(self.parameters()).device
            encoded = self.tokenize(texts, device=device)

        # Get transformer outputs
        # Returns: last_hidden_state of shape (B, seq_len, hidden_dim)
        outputs = self.transformer(
            input_ids=encoded['input_ids'],
            attention_mask=encoded['attention_mask']
        )

        hidden_states = outputs.last_hidden_state  # (B, seq_len, 768)

        if attention_output:
            # Return all token features for cross-attention
            # Project: (B, seq_len, 768) → (B, seq_len, 512)
            features = self.projection(hidden_states)
            return features
        else:
            # Return [CLS] token feature (first token)
            # The [CLS] token is trained to represent the whole sentence
            cls_token = hidden_states[:, 0, :]  # (B, 768)

            # Project: (B, 768) → (B, 512)
            features = self.projection(cls_token)
            return features

    def get_output_dim(self) -> int:
        """Return the output feature dimension."""
        return self.output_dim

    def get_attention_mask(self, texts: Union[str, List[str]]) -> torch.Tensor:
        """
        Get attention mask for texts (useful for cross-attention).

        Args:
            texts: Text string(s)

        Returns:
            Attention mask tensor (B, seq_len)
        """
        device = next(self.parameters()).device
        encoded = self.tokenize(texts, device=device)
        return encoded['attention_mask']


# =============================================================================
# Quick test
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Testing Language Encoder")
    print("=" * 60)

    # Test 1: Single text
    print("\n--- Test 1: Single Text ---")
    encoder = LanguageEncoder(output_dim=512)

    text = "pick up the red block"
    features = encoder(text)
    print(f"Input:  '{text}'")
    print(f"Output: {features.shape}")

    # Test 2: Batch of texts
    print("\n--- Test 2: Batch of Texts ---")
    texts = [
        "pick up the red block",
        "push the blue cube forward",
        "place the green block on the target"
    ]
    features = encoder(texts)
    print(f"Input:  {len(texts)} texts")
    print(f"Output: {features.shape}")

    # Test 3: Token-level features (for cross-attention)
    print("\n--- Test 3: Token-level Features ---")
    features_tokens = encoder(texts, attention_output=True)
    print(f"Input:  {len(texts)} texts")
    print(f"Output: {features_tokens.shape}")  # (B, seq_len, 512)

    # Test 4: Examine tokenization
    print("\n--- Test 4: Tokenization Example ---")
    text = "pick up the red block"
    tokens = encoder.tokenizer.tokenize(text)
    token_ids = encoder.tokenizer.encode(text)
    print(f"Text:    '{text}'")
    print(f"Tokens:  {tokens}")
    print(f"IDs:     {token_ids}")

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
