"""
PyTorch Dataset for Robot Demonstrations
========================================
Loads (image, instruction, action) tuples for behavior cloning.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path

try:
    from .demo_collector import DemoDataset, Demonstration
    from ..models.vision_encoder import get_image_transform
except ImportError:
    # When running as script
    from data.demo_collector import DemoDataset, Demonstration
    from models.vision_encoder import get_image_transform


class RobotDemoDataset(Dataset):
    """
    PyTorch Dataset for robot demonstrations.

    Each sample contains:
    - image: Camera observation (3, H, W), normalized
    - task: Task instruction string
    - action: Target action dict {delta_position, gripper}

    Usage:
        dataset = RobotDemoDataset.from_demos(demo_dataset)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

        for batch in dataloader:
            images = batch['image']           # (B, 3, 224, 224)
            tasks = batch['task']             # List[str]
            actions = batch['action']         # Dict of tensors
    """

    def __init__(
        self,
        images: List[np.ndarray],
        tasks: List[str],
        actions: List[Dict],
        image_size: int = 224,
    ):
        """
        Args:
            images: List of RGB images (H, W, 3)
            tasks: List of task instruction strings
            actions: List of action dicts
            image_size: Target image size after transform
        """
        assert len(images) == len(tasks) == len(actions), \
            "All lists must have same length"

        self.images = images
        self.tasks = tasks
        self.actions = actions

        # Image preprocessing
        self.transform = get_image_transform(image_size)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx: int) -> Dict:
        """
        Get a single training sample.

        Returns:
            Dict with 'image', 'task', 'action' keys
        """
        # Get raw data
        image = self.images[idx]
        task = self.tasks[idx]
        action = self.actions[idx]

        # Transform image: (H, W, 3) uint8 → (3, H, W) normalized float
        image_tensor = self.transform(image)

        # Convert action to tensors
        action_tensor = {
            'delta_position': torch.tensor(
                action['delta_position'], dtype=torch.float32
            ),
            'gripper': torch.tensor(
                [action['gripper']], dtype=torch.float32
            ),
        }

        return {
            'image': image_tensor,
            'task': task,
            'action': action_tensor,
        }

    @classmethod
    def from_demos(
        cls,
        demo_dataset: DemoDataset,
        image_size: int = 224,
    ) -> 'RobotDemoDataset':
        """
        Create dataset from collected demonstrations.

        Flattens all trajectories into individual transitions.

        Args:
            demo_dataset: DemoDataset with collected demos
            image_size: Target image size

        Returns:
            RobotDemoDataset ready for training
        """
        images = []
        tasks = []
        actions = []

        for demo in demo_dataset.demos:
            # Each demo has T observations and T-1 or T actions
            # We pair observation[t] with action[t]
            num_transitions = min(len(demo.observations), len(demo.actions))

            for t in range(num_transitions):
                images.append(demo.observations[t])
                tasks.append(demo.task)
                actions.append(demo.actions[t])

        print(f"Created dataset with {len(images)} transitions from {len(demo_dataset)} demos")

        return cls(images, tasks, actions, image_size)

    @classmethod
    def from_file(
        cls,
        path: str,
        image_size: int = 224,
    ) -> 'RobotDemoDataset':
        """
        Load dataset from saved demo file.

        Args:
            path: Path to saved DemoDataset pickle
            image_size: Target image size

        Returns:
            RobotDemoDataset ready for training
        """
        demo_dataset = DemoDataset.load(path)
        return cls.from_demos(demo_dataset, image_size)


def collate_fn(batch: List[Dict]) -> Dict:
    """
    Custom collate function for DataLoader.

    Handles variable-length text (tasks) properly.

    Args:
        batch: List of samples from dataset

    Returns:
        Collated batch with proper tensor stacking
    """
    # Stack images
    images = torch.stack([sample['image'] for sample in batch])

    # Keep tasks as list of strings (tokenizer will handle later)
    tasks = [sample['task'] for sample in batch]

    # Stack actions
    actions = {
        'delta_position': torch.stack([
            sample['action']['delta_position'] for sample in batch
        ]),
        'gripper': torch.stack([
            sample['action']['gripper'] for sample in batch
        ]),
    }

    return {
        'image': images,
        'task': tasks,
        'action': actions,
    }


def create_dataloader(
    dataset: RobotDemoDataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """
    Create DataLoader for training.

    Args:
        dataset: RobotDemoDataset instance
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of data loading workers

    Returns:
        PyTorch DataLoader
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )


# =============================================================================
# Data augmentation (optional, for better generalization)
# =============================================================================

class AugmentedRobotDataset(RobotDemoDataset):
    """
    Dataset with data augmentation for better generalization.

    Augmentations:
    - Color jitter (brightness, contrast, saturation)
    - Random crop and resize
    - Gaussian noise

    Note: Be careful with augmentation in robotics!
    - Don't flip images (changes left/right semantics)
    - Don't rotate too much (changes spatial relationships)
    """

    def __init__(
        self,
        images: List[np.ndarray],
        tasks: List[str],
        actions: List[Dict],
        image_size: int = 224,
        augment: bool = True,
    ):
        super().__init__(images, tasks, actions, image_size)

        self.augment = augment

        if augment:
            import torchvision.transforms as T

            # Augmentation pipeline (applied after base transform)
            self.augmentation = T.Compose([
                T.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.05,
                ),
                T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
            ])

    def __getitem__(self, idx: int) -> Dict:
        sample = super().__getitem__(idx)

        if self.augment and self.training:
            sample['image'] = self.augmentation(sample['image'])

        return sample

    @property
    def training(self) -> bool:
        """Check if in training mode (could be set externally)."""
        return getattr(self, '_training', True)

    def train(self):
        self._training = True

    def eval(self):
        self._training = False


# =============================================================================
# Quick test
# =============================================================================

if __name__ == "__main__":
    print("Testing Dataset")
    print("=" * 40)

    # Create dummy data
    num_samples = 100
    images = [np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8) for _ in range(num_samples)]
    tasks = ["pick up the red block"] * num_samples
    actions = [{"delta_position": np.random.randn(3) * 0.01, "gripper": 0.0} for _ in range(num_samples)]

    # Create dataset
    dataset = RobotDemoDataset(images, tasks, actions)
    print(f"Dataset size: {len(dataset)}")

    # Get sample
    sample = dataset[0]
    print(f"Sample image shape: {sample['image'].shape}")
    print(f"Sample task: {sample['task']}")
    print(f"Sample action position: {sample['action']['delta_position']}")

    # Create dataloader
    dataloader = create_dataloader(dataset, batch_size=8)

    # Get batch
    batch = next(iter(dataloader))
    print(f"\nBatch shapes:")
    print(f"  Images: {batch['image'].shape}")
    print(f"  Tasks: {len(batch['task'])} strings")
    print(f"  Actions delta_pos: {batch['action']['delta_position'].shape}")
    print(f"  Actions gripper: {batch['action']['gripper'].shape}")

    print("\n✓ Dataset tests passed!")
