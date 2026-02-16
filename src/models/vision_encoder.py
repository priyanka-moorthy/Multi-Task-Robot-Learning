"""
Vision Encoder for VLA Model
============================
Processes camera images into feature vectors using transfer learning.

Architecture: EfficientNet-B0
- Lightweight (5.3M parameters vs 25M for ResNet-50)
- Good accuracy/efficiency trade-off
- Fast inference on CPU

Key Concepts:
1. Transfer Learning: Use pretrained ImageNet weights
2. Feature Extraction: Remove classification head, keep features
3. Spatial Features: Preserve some spatial information for manipulation
"""

import torch
import torch.nn as nn
import timm
from typing import Tuple, Optional


class VisionEncoder(nn.Module):
    """
    Vision encoder that extracts features from camera images.

    Input:  RGB image tensor (B, 3, 224, 224)
    Output: Feature vector (B, output_dim) OR spatial features (B, H*W, output_dim)

    Example:
        >>> encoder = VisionEncoder(output_dim=512)
        >>> image = torch.randn(1, 3, 224, 224)  # Batch of 1 image
        >>> features = encoder(image)
        >>> features.shape
        torch.Size([1, 512])
    """

    def __init__(
        self,
        model_name: str = "efficientnet_b0",
        output_dim: int = 512,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        spatial_features: bool = False,
    ):
        """
        Args:
            model_name: timm model name (efficientnet_b0, mobilenetv3_small_100, etc.)
            output_dim: Dimension of output feature vector
            pretrained: Whether to use ImageNet pretrained weights
            freeze_backbone: If True, freeze all backbone weights (feature extraction only)
            spatial_features: If True, return spatial feature map instead of pooled vector
        """
        super().__init__()

        self.output_dim = output_dim
        self.spatial_features = spatial_features

        # =====================================================================
        # Step 1: Load pretrained backbone from timm
        # =====================================================================
        # timm (PyTorch Image Models) provides 700+ pretrained models
        # EfficientNet-B0: Good balance of speed and accuracy
        #
        # Key parameters:
        # - pretrained=True: Load ImageNet weights (1000-class classification)
        # - num_classes=0: Remove the classification head (we don't need it)
        # - global_pool='': Don't pool features yet (we'll do it ourselves)

        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,      # Remove classification head
            global_pool='',     # Don't pool (keep spatial dims)
        )

        # Get the feature dimension from the backbone
        # EfficientNet-B0 outputs 1280 channels
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, 224, 224)
            dummy_output = self.backbone(dummy_input)
            backbone_dim = dummy_output.shape[1]  # Channel dimension
            self.spatial_size = dummy_output.shape[2:]  # (H, W) after backbone

        print(f"[VisionEncoder] Backbone: {model_name}")
        print(f"[VisionEncoder] Backbone output: {backbone_dim} channels, spatial size: {self.spatial_size}")

        # =====================================================================
        # Step 2: Optionally freeze backbone weights
        # =====================================================================
        # Freezing = Don't update weights during training
        # - Faster training (fewer gradients to compute)
        # - Less overfitting (fewer parameters to optimize)
        # - Use when: Small dataset, limited compute, or when pretrained
        #   features are already good enough

        if freeze_backbone:
            print("[VisionEncoder] Freezing backbone weights")
            for param in self.backbone.parameters():
                param.requires_grad = False

        # =====================================================================
        # Step 3: Add projection layer to match output_dim
        # =====================================================================
        # The backbone outputs 1280-dim features
        # We project to output_dim (512) for:
        # - Consistency with language encoder
        # - Reduced memory usage
        # - Better fusion with language features

        if spatial_features:
            # For spatial features: 1x1 conv to project channels
            self.projection = nn.Sequential(
                nn.Conv2d(backbone_dim, output_dim, kernel_size=1),
                nn.BatchNorm2d(output_dim),
                nn.ReLU(inplace=True),
            )
        else:
            # For pooled features: Global Average Pooling + MLP
            self.global_pool = nn.AdaptiveAvgPool2d(1)  # Pool to (B, C, 1, 1)
            self.projection = nn.Sequential(
                nn.Linear(backbone_dim, output_dim),
                nn.LayerNorm(output_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
            )

        # Track number of parameters
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[VisionEncoder] Total params: {total_params:,}")
        print(f"[VisionEncoder] Trainable params: {trainable_params:,}")

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract features from images.

        Args:
            images: RGB images, shape (B, 3, H, W), values in [0, 1]

        Returns:
            If spatial_features=False: (B, output_dim) - pooled features
            If spatial_features=True:  (B, H*W, output_dim) - spatial features
        """
        # Get backbone features: (B, 1280, 7, 7) for 224x224 input
        features = self.backbone(images)

        if self.spatial_features:
            # Project channels: (B, 1280, 7, 7) → (B, 512, 7, 7)
            features = self.projection(features)

            # Flatten spatial dims: (B, 512, 7, 7) → (B, 49, 512)
            B, C, H, W = features.shape
            features = features.view(B, C, H * W)  # (B, C, H*W)
            features = features.permute(0, 2, 1)   # (B, H*W, C)
        else:
            # Global average pooling: (B, 1280, 7, 7) → (B, 1280)
            features = self.global_pool(features)
            features = features.view(features.size(0), -1)

            # Project to output_dim: (B, 1280) → (B, 512)
            features = self.projection(features)

        return features

    def get_output_dim(self) -> int:
        """Return the output feature dimension."""
        return self.output_dim

    def get_num_spatial_tokens(self) -> int:
        """Return number of spatial tokens (for transformer input)."""
        if self.spatial_features:
            return self.spatial_size[0] * self.spatial_size[1]
        return 1


# =============================================================================
# Preprocessing utilities
# =============================================================================

def get_image_transform(image_size: int = 224):
    """
    Get preprocessing transform for images.

    This normalizes images to match ImageNet statistics (what the model was trained on).

    Args:
        image_size: Target image size

    Returns:
        Transform function that converts numpy array to normalized tensor
    """
    import torchvision.transforms as T

    # ImageNet normalization statistics
    # These are the mean and std of ImageNet training data
    # We use them because our backbone was pretrained on ImageNet
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    transform = T.Compose([
        T.ToPILImage(),                           # numpy → PIL
        T.Resize((image_size, image_size)),       # Resize to target
        T.ToTensor(),                             # PIL → tensor, scales to [0, 1]
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),  # Normalize
    ])

    return transform


# =============================================================================
# Quick test
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Testing Vision Encoder")
    print("=" * 60)

    # Test 1: Pooled features (default)
    print("\n--- Test 1: Pooled Features ---")
    encoder = VisionEncoder(output_dim=512, pretrained=True)

    # Dummy batch of 4 images
    batch = torch.randn(4, 3, 224, 224)
    features = encoder(batch)
    print(f"Input shape:  {batch.shape}")
    print(f"Output shape: {features.shape}")
    assert features.shape == (4, 512), "Unexpected output shape!"

    # Test 2: Spatial features (for cross-attention)
    print("\n--- Test 2: Spatial Features ---")
    encoder_spatial = VisionEncoder(
        output_dim=512,
        pretrained=True,
        spatial_features=True
    )

    features_spatial = encoder_spatial(batch)
    print(f"Input shape:  {batch.shape}")
    print(f"Output shape: {features_spatial.shape}")
    # Should be (B, num_patches, dim) = (4, 49, 512) for 7x7 spatial grid

    # Test 3: Frozen backbone
    print("\n--- Test 3: Frozen Backbone ---")
    encoder_frozen = VisionEncoder(
        output_dim=256,
        pretrained=True,
        freeze_backbone=True
    )

    # Verify gradients
    batch.requires_grad = True
    output = encoder_frozen(batch)
    loss = output.sum()
    loss.backward()

    # Check that backbone params have no gradients
    backbone_grads = [p.grad for p in encoder_frozen.backbone.parameters() if p.grad is not None]
    print(f"Backbone gradients: {len(backbone_grads)} (should be 0)")

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
