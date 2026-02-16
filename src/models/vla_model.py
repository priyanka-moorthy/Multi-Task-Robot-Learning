"""
Vision-Language-Action (VLA) Model
==================================
Complete model that takes camera images + language instructions
and outputs robot actions.

Architecture:
┌─────────────┐    ┌──────────────┐
│   Image     │    │    Text      │
│ (224x224x3) │    │ "pick up..." │
└──────┬──────┘    └───────┬──────┘
       │                   │
       ▼                   ▼
┌─────────────┐    ┌──────────────┐
│   Vision    │    │   Language   │
│   Encoder   │    │   Encoder    │
│(EfficientNet)│   │ (DistilBERT) │
└──────┬──────┘    └───────┬──────┘
       │                   │
       │   ┌───────────────┘
       │   │
       ▼   ▼
   ┌────────────┐
   │   Fusion   │
   │  (Cross-   │
   │ Attention) │
   └─────┬──────┘
         │
         ▼
   ┌────────────┐
   │   Action   │
   │    Head    │
   └─────┬──────┘
         │
         ▼
  [dx, dy, dz, gripper]
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Union, List
import numpy as np

from .vision_encoder import VisionEncoder, get_image_transform
from .language_encoder import LanguageEncoder
from .fusion_module import VisionLanguageFusion
from .action_head import ActionHead


class VLAModel(nn.Module):
    """
    Vision-Language-Action Model for multi-task robotic manipulation.

    This is the main model that combines all components:
    1. VisionEncoder: EfficientNet-B0 for image features
    2. LanguageEncoder: DistilBERT for text features
    3. VisionLanguageFusion: Cross-attention to combine modalities
    4. ActionHead: MLP to predict robot actions

    Example:
        >>> model = VLAModel()
        >>> image = torch.randn(1, 3, 224, 224)
        >>> text = "pick up the red block"
        >>> actions = model(image, text)
        >>> actions['delta_position'].shape
        torch.Size([1, 3])
    """

    def __init__(
        self,
        # Vision encoder config
        vision_model: str = "efficientnet_b0",
        vision_pretrained: bool = True,
        freeze_vision: bool = False,
        # Language encoder config
        language_model: str = "distilbert-base-uncased",
        freeze_language: bool = False,
        max_text_length: int = 32,
        # Fusion config
        embed_dim: int = 512,
        fusion_heads: int = 8,
        fusion_layers: int = 2,
        # Action head config
        action_hidden_dim: int = 256,
        action_dim: int = 3,
        max_action: float = 0.05,
        # General config
        dropout: float = 0.1,
    ):
        """
        Args:
            vision_model: timm model name for vision encoder
            vision_pretrained: Use ImageNet pretrained weights
            freeze_vision: Freeze vision encoder weights
            language_model: HuggingFace model name for language encoder
            freeze_language: Freeze language encoder weights
            max_text_length: Maximum text sequence length
            embed_dim: Shared embedding dimension
            fusion_heads: Number of attention heads in fusion
            fusion_layers: Number of cross-attention layers
            action_hidden_dim: Hidden dimension in action head
            action_dim: Position action dimension (3 for xyz)
            max_action: Maximum action magnitude
            dropout: Dropout probability
        """
        super().__init__()

        self.embed_dim = embed_dim

        print("=" * 60)
        print("Initializing VLA Model")
        print("=" * 60)

        # =====================================================================
        # 1. Vision Encoder
        # =====================================================================
        print("\n[1/4] Vision Encoder")
        self.vision_encoder = VisionEncoder(
            model_name=vision_model,
            output_dim=embed_dim,
            pretrained=vision_pretrained,
            freeze_backbone=freeze_vision,
            spatial_features=True,  # Keep spatial info for cross-attention
        )

        # Image preprocessing transform
        self.image_transform = get_image_transform(image_size=224)

        # =====================================================================
        # 2. Language Encoder
        # =====================================================================
        print("\n[2/4] Language Encoder")
        self.language_encoder = LanguageEncoder(
            model_name=language_model,
            output_dim=embed_dim,
            freeze_backbone=freeze_language,
            max_length=max_text_length,
        )

        # =====================================================================
        # 3. Fusion Module
        # =====================================================================
        print("\n[3/4] Fusion Module")
        self.fusion = VisionLanguageFusion(
            embed_dim=embed_dim,
            num_heads=fusion_heads,
            num_layers=fusion_layers,
            dropout=dropout,
        )

        # =====================================================================
        # 4. Action Head
        # =====================================================================
        print("\n[4/4] Action Head")
        self.action_head = ActionHead(
            input_dim=embed_dim,
            hidden_dim=action_hidden_dim,
            action_dim=action_dim,
            max_action=max_action,
            dropout=dropout,
        )

        # Count total parameters
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        print("\n" + "=" * 60)
        print(f"VLA Model Initialized")
        print(f"  Total params:     {total_params:,}")
        print(f"  Trainable params: {trainable_params:,}")
        print("=" * 60)

    def forward(
        self,
        images: torch.Tensor,
        texts: Union[str, List[str], dict],
        return_attention: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass: Image + Text → Actions

        Args:
            images: RGB images (B, 3, 224, 224), normalized
            texts: Task instructions (string, list, or tokenized dict)
            return_attention: If True, include attention weights

        Returns:
            Dictionary containing:
                - delta_position: (B, 3) end-effector displacement
                - gripper: (B, 1) gripper command
                - action: (B, 4) concatenated action
                - attention_weights: (optional) cross-attention weights
        """
        # 1. Encode vision (spatial features)
        vision_features = self.vision_encoder(images)  # (B, num_patches, embed_dim)

        # 2. Encode language (token features)
        language_features = self.language_encoder(texts, attention_output=True)  # (B, seq_len, embed_dim)

        # 3. Fuse modalities
        if return_attention:
            fused_features, attention_weights = self.fusion(
                vision_features, language_features, return_attention=True
            )
        else:
            fused_features = self.fusion(vision_features, language_features)

        # 4. Predict actions
        actions = self.action_head(fused_features)

        if return_attention:
            actions['attention_weights'] = attention_weights

        return actions

    def predict(
        self,
        image: np.ndarray,
        text: str,
    ) -> Dict[str, np.ndarray]:
        """
        Convenience method for inference with numpy inputs.

        Args:
            image: RGB image as numpy array (H, W, 3), values in [0, 255]
            text: Task instruction string

        Returns:
            Dictionary with numpy action arrays
        """
        self.eval()

        with torch.no_grad():
            # Preprocess image
            device = next(self.parameters()).device
            image_tensor = self.image_transform(image).unsqueeze(0).to(device)

            # Forward pass
            outputs = self.forward(image_tensor, text)

            # Convert to numpy
            return {
                'delta_position': outputs['delta_position'].cpu().numpy()[0],
                'gripper': outputs['gripper'].cpu().numpy()[0],
            }

    def get_action(self, image: np.ndarray, text: str):
        """
        Get action in the format expected by the environment.

        Returns an Action dataclass-compatible dict.
        """
        pred = self.predict(image, text)
        return {
            'delta_position': pred['delta_position'],
            'gripper': float(pred['gripper'][0]),
        }


# =============================================================================
# Model factory
# =============================================================================

def create_vla_model(
    config: str = "default",
    **kwargs,
) -> VLAModel:
    """
    Create VLA model with predefined configurations.

    Configs:
        - "default": Balanced config for CPU training
        - "small": Smaller model for faster training/inference
        - "large": Larger model for better performance
        - "frozen": Freeze encoders, only train fusion + action head
    """
    configs = {
        "default": {
            "vision_model": "efficientnet_b0",
            "embed_dim": 512,
            "fusion_heads": 8,
            "fusion_layers": 2,
            "action_hidden_dim": 256,
        },
        "small": {
            "vision_model": "mobilenetv3_small_100",
            "embed_dim": 256,
            "fusion_heads": 4,
            "fusion_layers": 1,
            "action_hidden_dim": 128,
        },
        "large": {
            "vision_model": "efficientnet_b2",
            "embed_dim": 768,
            "fusion_heads": 12,
            "fusion_layers": 4,
            "action_hidden_dim": 512,
        },
        "frozen": {
            "vision_model": "efficientnet_b0",
            "freeze_vision": True,
            "freeze_language": True,
            "embed_dim": 512,
            "fusion_heads": 8,
            "fusion_layers": 2,
        },
    }

    if config not in configs:
        raise ValueError(f"Unknown config: {config}. Available: {list(configs.keys())}")

    # Merge config with overrides
    model_config = {**configs[config], **kwargs}

    return VLAModel(**model_config)


# =============================================================================
# Quick test
# =============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Testing VLA Model")
    print("=" * 60)

    # Create model
    model = create_vla_model("default")

    # Test forward pass
    print("\n--- Forward Pass Test ---")
    batch_size = 2

    # Dummy inputs
    images = torch.randn(batch_size, 3, 224, 224)
    texts = ["pick up the red block", "push the blue cube"]

    # Forward pass
    outputs = model(images, texts)

    print(f"\nInputs:")
    print(f"  Images: {images.shape}")
    print(f"  Texts:  {texts}")

    print(f"\nOutputs:")
    print(f"  delta_position: {outputs['delta_position'].shape}")
    print(f"  gripper:        {outputs['gripper'].shape}")
    print(f"  action:         {outputs['action'].shape}")

    # Test with attention
    print("\n--- With Attention Weights ---")
    outputs_attn = model(images, texts, return_attention=True)
    print(f"  Attention layers: {len(outputs_attn['attention_weights'])}")

    # Test predict method (numpy interface)
    print("\n--- Numpy Interface Test ---")
    dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    result = model.predict(dummy_image, "pick up the red block")
    print(f"  delta_position: {result['delta_position']}")
    print(f"  gripper: {result['gripper']}")

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
