"""
Fusion Module for VLA Model
===========================
Combines vision and language features using cross-attention.

This is where the "magic" happens - the model learns to:
1. Use language to guide attention over visual features
2. Focus on task-relevant objects/regions
3. Create a unified representation for action prediction

Key Insight:
"pick up the RED block" → attend to red regions in the image
"push the BLUE cube" → attend to blue regions

Architecture Options:
1. Simple concatenation (baseline)
2. Cross-attention (what we'll use)
3. FiLM conditioning (alternative)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class CrossAttention(nn.Module):
    """
    Cross-attention layer where queries come from one modality
    and keys/values come from another.

    For VLA: Language queries attend to Vision key-values
    - Q: "What am I looking for?" (from language)
    - K: "What's available?" (from vision)
    - V: "What information to extract?" (from vision)

    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V
    """

    def __init__(
        self,
        embed_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        """
        Args:
            embed_dim: Dimension of input features
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Scale factor for attention scores
        self.scale = math.sqrt(self.head_dim)

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute cross-attention.

        Args:
            query: Query tensor from one modality (B, seq_q, embed_dim)
            key_value: Key/Value tensor from other modality (B, seq_kv, embed_dim)
            key_padding_mask: Mask for key positions to ignore (B, seq_kv)
                              True = ignore, False = attend

        Returns:
            output: Attended features (B, seq_q, embed_dim)
            attention_weights: Attention weights (B, num_heads, seq_q, seq_kv)
        """
        B, seq_q, _ = query.shape
        _, seq_kv, _ = key_value.shape

        # Project to Q, K, V
        Q = self.q_proj(query)      # (B, seq_q, embed_dim)
        K = self.k_proj(key_value)  # (B, seq_kv, embed_dim)
        V = self.v_proj(key_value)  # (B, seq_kv, embed_dim)

        # Reshape for multi-head attention
        # (B, seq, embed_dim) → (B, num_heads, seq, head_dim)
        Q = Q.view(B, seq_q, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, seq_kv, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, seq_kv, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute attention scores
        # (B, heads, seq_q, head_dim) @ (B, heads, head_dim, seq_kv)
        # → (B, heads, seq_q, seq_kv)
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        # Apply key padding mask (if provided)
        if key_padding_mask is not None:
            # Expand mask for heads: (B, seq_kv) → (B, 1, 1, seq_kv)
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            attention_scores = attention_scores.masked_fill(mask, float('-inf'))

        # Softmax to get attention weights
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Apply attention to values
        # (B, heads, seq_q, seq_kv) @ (B, heads, seq_kv, head_dim)
        # → (B, heads, seq_q, head_dim)
        output = torch.matmul(attention_weights, V)

        # Reshape back: (B, heads, seq_q, head_dim) → (B, seq_q, embed_dim)
        output = output.transpose(1, 2).contiguous().view(B, seq_q, self.embed_dim)

        # Final projection
        output = self.out_proj(output)

        return output, attention_weights


class VisionLanguageFusion(nn.Module):
    """
    Fuses vision and language features for robotic action prediction.

    Architecture:
    1. Language-to-Vision Cross-Attention
       - Language queries attend to spatial vision features
       - "Where should I look based on the instruction?"

    2. Self-Attention (optional)
       - Refine the fused representation

    3. Pooling
       - Aggregate into a single vector for action prediction
    """

    def __init__(
        self,
        embed_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
        use_self_attention: bool = True,
    ):
        """
        Args:
            embed_dim: Feature dimension (must match encoders)
            num_heads: Number of attention heads
            num_layers: Number of cross-attention layers
            dropout: Dropout probability
            use_self_attention: Whether to use self-attention after cross-attention
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.use_self_attention = use_self_attention

        # Cross-attention layers: Language attends to Vision
        self.cross_attention_layers = nn.ModuleList([
            CrossAttention(embed_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

        # Layer norms for residual connections
        self.cross_attn_norms = nn.ModuleList([
            nn.LayerNorm(embed_dim) for _ in range(num_layers)
        ])

        # Optional self-attention to refine fused features
        if use_self_attention:
            self.self_attention = nn.MultiheadAttention(
                embed_dim, num_heads, dropout=dropout, batch_first=True
            )
            self.self_attn_norm = nn.LayerNorm(embed_dim)

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout),
        )
        self.ffn_norm = nn.LayerNorm(embed_dim)

        # Final projection (optional pooling weights)
        self.output_projection = nn.Linear(embed_dim, embed_dim)

        print(f"[VisionLanguageFusion] embed_dim={embed_dim}, heads={num_heads}, layers={num_layers}")

    def forward(
        self,
        vision_features: torch.Tensor,
        language_features: torch.Tensor,
        vision_mask: Optional[torch.Tensor] = None,
        language_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> torch.Tensor:
        """
        Fuse vision and language features.

        Args:
            vision_features: Spatial vision features (B, num_patches, embed_dim)
                            e.g., (B, 49, 512) for 7x7 spatial grid
            language_features: Token-level language features (B, seq_len, embed_dim)
                              e.g., (B, 32, 512) for max 32 tokens
            vision_mask: Mask for vision features (B, num_patches) - True = ignore
            language_mask: Mask for language features (B, seq_len) - True = ignore
            return_attention: Whether to return attention weights

        Returns:
            fused_features: Fused representation (B, embed_dim)
            attention_weights: (optional) Cross-attention weights
        """
        # Language queries attend to Vision key-values
        # This answers: "Given the instruction, what visual features are relevant?"

        attended_features = language_features  # (B, seq_len, embed_dim)
        all_attention_weights = []

        # Apply cross-attention layers
        for i, (cross_attn, norm) in enumerate(zip(
            self.cross_attention_layers, self.cross_attn_norms
        )):
            # Cross-attention with residual
            attn_output, attn_weights = cross_attn(
                query=attended_features,
                key_value=vision_features,
                key_padding_mask=vision_mask,
            )

            # Residual connection + layer norm
            attended_features = norm(attended_features + attn_output)

            if return_attention:
                all_attention_weights.append(attn_weights)

        # Optional self-attention to refine
        if self.use_self_attention:
            self_attn_output, _ = self.self_attention(
                attended_features, attended_features, attended_features,
                key_padding_mask=language_mask,
            )
            attended_features = self.self_attn_norm(attended_features + self_attn_output)

        # Feed-forward network with residual
        ffn_output = self.ffn(attended_features)
        attended_features = self.ffn_norm(attended_features + ffn_output)

        # Pool to single vector (use [CLS] position or mean)
        # Option 1: Take first token (like [CLS])
        pooled = attended_features[:, 0, :]  # (B, embed_dim)

        # Option 2: Mean pooling (alternative)
        # if language_mask is not None:
        #     mask = ~language_mask.unsqueeze(-1)  # (B, seq_len, 1)
        #     pooled = (attended_features * mask).sum(1) / mask.sum(1)
        # else:
        #     pooled = attended_features.mean(dim=1)

        # Final projection
        output = self.output_projection(pooled)

        if return_attention:
            return output, all_attention_weights

        return output

    def get_output_dim(self) -> int:
        """Return output feature dimension."""
        return self.embed_dim


# =============================================================================
# Simpler Fusion Options (for comparison/ablation)
# =============================================================================

class ConcatFusion(nn.Module):
    """
    Simple concatenation-based fusion (baseline).

    Just concatenates vision and language vectors and projects down.
    No attention mechanism - doesn't "understand" the relationship.
    """

    def __init__(self, vision_dim: int = 512, language_dim: int = 512, output_dim: int = 512):
        super().__init__()

        self.projection = nn.Sequential(
            nn.Linear(vision_dim + language_dim, output_dim * 2),
            nn.LayerNorm(output_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(output_dim * 2, output_dim),
        )

    def forward(self, vision_features: torch.Tensor, language_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            vision_features: (B, vision_dim) - pooled vision features
            language_features: (B, language_dim) - pooled language features

        Returns:
            fused: (B, output_dim)
        """
        # Simple concatenation
        concat = torch.cat([vision_features, language_features], dim=-1)
        return self.projection(concat)


# =============================================================================
# Quick test
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Testing Fusion Module")
    print("=" * 60)

    batch_size = 4

    # Test 1: Cross-attention fusion
    print("\n--- Test 1: VisionLanguageFusion ---")
    fusion = VisionLanguageFusion(embed_dim=512, num_heads=8, num_layers=2)

    # Simulate encoder outputs
    vision_features = torch.randn(batch_size, 49, 512)   # 7x7 spatial grid
    language_features = torch.randn(batch_size, 32, 512)  # 32 tokens

    fused = fusion(vision_features, language_features)
    print(f"Vision input:   {vision_features.shape}")
    print(f"Language input: {language_features.shape}")
    print(f"Fused output:   {fused.shape}")

    # Test 2: With attention weights
    print("\n--- Test 2: With Attention Weights ---")
    fused, attn_weights = fusion(
        vision_features, language_features, return_attention=True
    )
    print(f"Attention weights: {len(attn_weights)} layers")
    print(f"  Layer 0 shape: {attn_weights[0].shape}")  # (B, heads, lang_seq, vis_seq)

    # Test 3: Simple concatenation baseline
    print("\n--- Test 3: Concat Fusion (Baseline) ---")
    concat_fusion = ConcatFusion(512, 512, 512)

    vision_pooled = vision_features.mean(dim=1)    # Pool to (B, 512)
    language_pooled = language_features[:, 0, :]   # [CLS] token (B, 512)

    fused_concat = concat_fusion(vision_pooled, language_pooled)
    print(f"Vision pooled:   {vision_pooled.shape}")
    print(f"Language pooled: {language_pooled.shape}")
    print(f"Fused output:    {fused_concat.shape}")

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
