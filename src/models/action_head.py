"""
Action Head for VLA Model
=========================
Predicts robot actions from fused vision-language features.

Output: End-effector control
- delta_position: [dx, dy, dz] - move gripper by this amount
- gripper: [0, 1] - gripper openness (0=open, 1=closed)

Architecture Options:
1. Deterministic MLP (what we'll use)
2. Gaussian MLP (for stochastic policies)
3. Mixture Density Network (for multi-modal actions)

For learning, we can use:
- Behavior Cloning: MSE loss on expert demonstrations
- Reinforcement Learning: Policy gradient methods
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict
import math


class ActionHead(nn.Module):
    """
    Predicts robot actions from fused features.

    Input:  Fused features (B, input_dim)
    Output: Actions dict with:
            - delta_position: (B, 3) end-effector displacement
            - gripper: (B, 1) gripper command

    Example:
        >>> head = ActionHead(input_dim=512)
        >>> features = torch.randn(4, 512)
        >>> actions = head(features)
        >>> actions['delta_position'].shape
        torch.Size([4, 3])
    """

    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 256,
        num_layers: int = 2,
        action_dim: int = 3,  # dx, dy, dz
        max_action: float = 0.05,  # Maximum action magnitude (5cm)
        dropout: float = 0.1,
    ):
        """
        Args:
            input_dim: Dimension of input features
            hidden_dim: Hidden layer dimension
            num_layers: Number of hidden layers
            action_dim: Dimension of position action (usually 3 for xyz)
            max_action: Maximum action value (for tanh scaling)
            dropout: Dropout probability
        """
        super().__init__()

        self.input_dim = input_dim
        self.action_dim = action_dim
        self.max_action = max_action

        # Build MLP layers
        layers = []
        current_dim = input_dim

        for i in range(num_layers):
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ])
            current_dim = hidden_dim

        self.mlp = nn.Sequential(*layers)

        # Separate heads for position and gripper
        # This allows different output ranges and activations

        # Position head: outputs delta (dx, dy, dz)
        # Uses tanh to bound outputs to [-max_action, max_action]
        self.position_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, action_dim),
            nn.Tanh(),  # Bound to [-1, 1], then scale
        )

        # Gripper head: outputs gripper command
        # Uses sigmoid to bound to [0, 1] (0=open, 1=closed)
        self.gripper_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid(),  # Bound to [0, 1]
        )

        # Initialize weights
        self._init_weights()

        # Count parameters
        total_params = sum(p.numel() for p in self.parameters())
        print(f"[ActionHead] Total params: {total_params:,}")

    def _init_weights(self):
        """Initialize weights with small values for stable training."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.01)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        features: torch.Tensor,
        return_dict: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Predict actions from fused features.

        Args:
            features: Fused vision-language features (B, input_dim)
            return_dict: If True, return dict; else return tuple

        Returns:
            If return_dict=True:
                {
                    'delta_position': (B, 3),
                    'gripper': (B, 1),
                    'action': (B, 4) - concatenated
                }
            Else:
                (delta_position, gripper)
        """
        # Shared MLP
        hidden = self.mlp(features)

        # Position prediction
        delta_position = self.position_head(hidden)
        delta_position = delta_position * self.max_action  # Scale to [-max, max]

        # Gripper prediction
        gripper = self.gripper_head(hidden)

        if return_dict:
            return {
                'delta_position': delta_position,
                'gripper': gripper,
                'action': torch.cat([delta_position, gripper], dim=-1),
            }
        else:
            return delta_position, gripper

    def get_action_dim(self) -> int:
        """Return total action dimension (position + gripper)."""
        return self.action_dim + 1


class GaussianActionHead(nn.Module):
    """
    Stochastic action head that outputs a Gaussian distribution.

    Useful for:
    - Exploration in RL
    - Modeling uncertainty
    - Handling multi-modal actions

    Output: Mean and log_std for each action dimension
    """

    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 256,
        action_dim: int = 4,
        log_std_min: float = -5.0,
        log_std_max: float = 2.0,
    ):
        super().__init__()

        self.action_dim = action_dim
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )

        # Mean head
        self.mean_head = nn.Linear(hidden_dim, action_dim)

        # Log standard deviation head
        self.log_std_head = nn.Linear(hidden_dim, action_dim)

    def forward(
        self,
        features: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            features: (B, input_dim)
            deterministic: If True, return mean (no sampling)

        Returns:
            action: Sampled action (B, action_dim)
            mean: Action mean (B, action_dim)
            log_std: Action log std (B, action_dim)
        """
        hidden = self.mlp(features)

        mean = self.mean_head(hidden)
        log_std = self.log_std_head(hidden)

        # Clamp log_std for numerical stability
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        if deterministic:
            action = mean
        else:
            # Reparameterization trick: action = mean + std * epsilon
            std = torch.exp(log_std)
            noise = torch.randn_like(mean)
            action = mean + std * noise

        return action, mean, log_std

    def log_prob(
        self,
        features: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute log probability of action under the policy.

        Useful for policy gradient methods (PPO, SAC, etc.)
        """
        _, mean, log_std = self.forward(features, deterministic=True)
        std = torch.exp(log_std)

        # Gaussian log probability
        var = std ** 2
        log_prob = -0.5 * (
            ((action - mean) ** 2) / var
            + 2 * log_std
            + math.log(2 * math.pi)
        )

        # Sum over action dimensions
        return log_prob.sum(dim=-1)


# =============================================================================
# Loss functions for action prediction
# =============================================================================

def action_loss(
    predicted: Dict[str, torch.Tensor],
    target: Dict[str, torch.Tensor],
    position_weight: float = 1.0,
    gripper_weight: float = 1.0,
) -> Dict[str, torch.Tensor]:
    """
    Compute loss for behavior cloning (supervised learning from demonstrations).

    Args:
        predicted: Model predictions with 'delta_position' and 'gripper'
        target: Ground truth actions
        position_weight: Weight for position loss
        gripper_weight: Weight for gripper loss

    Returns:
        Dictionary of losses
    """
    # MSE loss for position (regression)
    position_loss = F.mse_loss(
        predicted['delta_position'],
        target['delta_position']
    )

    # Binary cross-entropy for gripper (classification-like)
    gripper_loss = F.binary_cross_entropy(
        predicted['gripper'],
        target['gripper']
    )

    # Total loss
    total_loss = position_weight * position_loss + gripper_weight * gripper_loss

    return {
        'total': total_loss,
        'position': position_loss,
        'gripper': gripper_loss,
    }


# =============================================================================
# Quick test
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Testing Action Head")
    print("=" * 60)

    batch_size = 4

    # Test 1: Deterministic action head
    print("\n--- Test 1: ActionHead (Deterministic) ---")
    head = ActionHead(input_dim=512, hidden_dim=256)

    features = torch.randn(batch_size, 512)
    actions = head(features)

    print(f"Input:          {features.shape}")
    print(f"delta_position: {actions['delta_position'].shape}")
    print(f"gripper:        {actions['gripper'].shape}")
    print(f"action:         {actions['action'].shape}")

    # Check action bounds
    print(f"Position range: [{actions['delta_position'].min():.4f}, {actions['delta_position'].max():.4f}]")
    print(f"Gripper range:  [{actions['gripper'].min():.4f}, {actions['gripper'].max():.4f}]")

    # Test 2: Loss computation
    print("\n--- Test 2: Loss Computation ---")
    target = {
        'delta_position': torch.randn(batch_size, 3) * 0.05,
        'gripper': torch.rand(batch_size, 1),
    }

    losses = action_loss(actions, target)
    print(f"Position loss: {losses['position']:.4f}")
    print(f"Gripper loss:  {losses['gripper']:.4f}")
    print(f"Total loss:    {losses['total']:.4f}")

    # Test 3: Gaussian action head (for RL)
    print("\n--- Test 3: GaussianActionHead (Stochastic) ---")
    gaussian_head = GaussianActionHead(input_dim=512, action_dim=4)

    action, mean, log_std = gaussian_head(features)
    print(f"Sampled action: {action.shape}")
    print(f"Mean:           {mean.shape}")
    print(f"Log std:        {log_std.shape}")

    # Test log probability
    log_prob = gaussian_head.log_prob(features, action)
    print(f"Log prob:       {log_prob.shape}")

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
