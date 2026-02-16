"""
VLA Model Trainer
=================
Training loop for behavior cloning with the VLA model.

Training Strategy:
1. Load expert demonstrations
2. Train model to imitate expert actions
3. Evaluate on held-out demonstrations
4. (Optional) Fine-tune in environment with RL
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from typing import Dict, Optional, List, Tuple
from pathlib import Path
import numpy as np
from tqdm import tqdm
import json
from datetime import datetime

try:
    from ..models.vla_model import VLAModel, create_vla_model
    from ..models.action_head import action_loss
    from ..data.dataset import RobotDemoDataset, create_dataloader
except ImportError:
    from models.vla_model import VLAModel, create_vla_model
    from models.action_head import action_loss
    from data.dataset import RobotDemoDataset, create_dataloader


class VLATrainer:
    """
    Trainer for Vision-Language-Action model.

    Implements behavior cloning (supervised learning from demonstrations).

    Usage:
        trainer = VLATrainer(model, train_loader, val_loader)
        trainer.train(num_epochs=100)
        trainer.save_checkpoint("checkpoints/best.pt")
    """

    def __init__(
        self,
        model: VLAModel,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        position_loss_weight: float = 1.0,
        gripper_loss_weight: float = 0.5,
        device: str = "cpu",
        checkpoint_dir: str = "experiments/checkpoints",
    ):
        """
        Args:
            model: VLA model to train
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            learning_rate: Learning rate for optimizer
            weight_decay: L2 regularization weight
            position_loss_weight: Weight for position prediction loss
            gripper_loss_weight: Weight for gripper prediction loss
            device: Device to train on ("cpu", "cuda", "mps")
            checkpoint_dir: Directory for saving checkpoints
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        self.position_loss_weight = position_loss_weight
        self.gripper_loss_weight = gripper_loss_weight

        # Optimizer with different learning rates for pretrained vs new params
        # Lower LR for pretrained encoders, higher for fusion/action head
        pretrained_params = list(model.vision_encoder.parameters()) + \
                           list(model.language_encoder.parameters())
        new_params = list(model.fusion.parameters()) + \
                    list(model.action_head.parameters())

        self.optimizer = optim.AdamW([
            {"params": pretrained_params, "lr": learning_rate * 0.1},  # Lower LR
            {"params": new_params, "lr": learning_rate},
        ], weight_decay=weight_decay)

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=100,  # Will be updated based on epochs
            eta_min=learning_rate * 0.01,
        )

        # Checkpoint management
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Training history
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_position_loss": [],
            "train_gripper_loss": [],
            "val_position_loss": [],
            "val_gripper_loss": [],
        }
        self.best_val_loss = float("inf")
        self.epoch = 0

    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.

        Returns:
            Dictionary of training metrics
        """
        self.model.train()

        total_loss = 0.0
        total_pos_loss = 0.0
        total_grip_loss = 0.0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch + 1}", leave=False)

        for batch in pbar:
            # Move to device
            images = batch["image"].to(self.device)
            tasks = batch["task"]  # List of strings
            target_actions = {
                "delta_position": batch["action"]["delta_position"].to(self.device),
                "gripper": batch["action"]["gripper"].to(self.device),
            }

            # Forward pass
            self.optimizer.zero_grad()
            predicted_actions = self.model(images, tasks)

            # Compute loss
            losses = action_loss(
                predicted_actions,
                target_actions,
                position_weight=self.position_loss_weight,
                gripper_weight=self.gripper_loss_weight,
            )

            # Backward pass
            losses["total"].backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            # Track metrics
            total_loss += losses["total"].item()
            total_pos_loss += losses["position"].item()
            total_grip_loss += losses["gripper"].item()
            num_batches += 1

            # Update progress bar
            pbar.set_postfix({
                "loss": f"{losses['total'].item():.4f}",
                "pos": f"{losses['position'].item():.4f}",
            })

        # Average metrics
        metrics = {
            "train_loss": total_loss / num_batches,
            "train_position_loss": total_pos_loss / num_batches,
            "train_gripper_loss": total_grip_loss / num_batches,
        }

        return metrics

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """
        Validate on held-out data.

        Returns:
            Dictionary of validation metrics
        """
        if self.val_loader is None:
            return {}

        self.model.eval()

        total_loss = 0.0
        total_pos_loss = 0.0
        total_grip_loss = 0.0
        num_batches = 0

        for batch in self.val_loader:
            images = batch["image"].to(self.device)
            tasks = batch["task"]
            target_actions = {
                "delta_position": batch["action"]["delta_position"].to(self.device),
                "gripper": batch["action"]["gripper"].to(self.device),
            }

            predicted_actions = self.model(images, tasks)

            losses = action_loss(
                predicted_actions,
                target_actions,
                position_weight=self.position_loss_weight,
                gripper_weight=self.gripper_loss_weight,
            )

            total_loss += losses["total"].item()
            total_pos_loss += losses["position"].item()
            total_grip_loss += losses["gripper"].item()
            num_batches += 1

        metrics = {
            "val_loss": total_loss / num_batches,
            "val_position_loss": total_pos_loss / num_batches,
            "val_gripper_loss": total_grip_loss / num_batches,
        }

        return metrics

    def train(
        self,
        num_epochs: int = 100,
        save_every: int = 10,
        early_stopping_patience: int = 20,
    ) -> Dict:
        """
        Full training loop.

        Args:
            num_epochs: Number of epochs to train
            save_every: Save checkpoint every N epochs
            early_stopping_patience: Stop if val loss doesn't improve for N epochs

        Returns:
            Training history dictionary
        """
        print("=" * 60)
        print("Starting VLA Training")
        print("=" * 60)
        print(f"  Device: {self.device}")
        print(f"  Epochs: {num_epochs}")
        print(f"  Train batches: {len(self.train_loader)}")
        if self.val_loader:
            print(f"  Val batches: {len(self.val_loader)}")
        print()

        # Update scheduler for actual epochs
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=num_epochs,
            eta_min=1e-6,
        )

        patience_counter = 0

        for epoch in range(num_epochs):
            self.epoch = epoch

            # Train
            train_metrics = self.train_epoch()

            # Validate
            val_metrics = self.validate()

            # Update learning rate
            self.scheduler.step()

            # Log metrics
            for key, value in train_metrics.items():
                self.history[key].append(value)
            for key, value in val_metrics.items():
                self.history[key].append(value)

            # Print progress
            current_lr = self.optimizer.param_groups[1]["lr"]
            msg = f"Epoch {epoch + 1}/{num_epochs} | "
            msg += f"Train Loss: {train_metrics['train_loss']:.4f} | "
            if val_metrics:
                msg += f"Val Loss: {val_metrics['val_loss']:.4f} | "
            msg += f"LR: {current_lr:.2e}"
            print(msg)

            # Save best model
            if val_metrics and val_metrics["val_loss"] < self.best_val_loss:
                self.best_val_loss = val_metrics["val_loss"]
                self.save_checkpoint("best.pt")
                patience_counter = 0
                print(f"  → New best model saved!")
            else:
                patience_counter += 1

            # Save periodic checkpoint
            if (epoch + 1) % save_every == 0:
                self.save_checkpoint(f"epoch_{epoch + 1}.pt")

            # Early stopping
            if patience_counter >= early_stopping_patience:
                print(f"\nEarly stopping at epoch {epoch + 1}")
                break

        print("\n" + "=" * 60)
        print("Training Complete!")
        print("=" * 60)
        print(f"  Best val loss: {self.best_val_loss:.4f}")
        print(f"  Checkpoints saved to: {self.checkpoint_dir}")

        # Save final model and history
        self.save_checkpoint("final.pt")
        self.save_history()

        return self.history

    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        path = self.checkpoint_dir / filename
        torch.save({
            "epoch": self.epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "history": self.history,
        }, path)

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.epoch = checkpoint["epoch"]
        self.best_val_loss = checkpoint["best_val_loss"]
        self.history = checkpoint["history"]
        print(f"Loaded checkpoint from epoch {self.epoch}")

    def save_history(self):
        """Save training history to JSON."""
        path = self.checkpoint_dir / "history.json"
        with open(path, "w") as f:
            json.dump(self.history, f, indent=2)


# =============================================================================
# Convenience function
# =============================================================================

def train_vla_model(
    demo_path: str,
    model_config: str = "default",
    num_epochs: int = 50,
    batch_size: int = 16,
    learning_rate: float = 1e-4,
    val_split: float = 0.1,
    device: str = "cpu",
    checkpoint_dir: str = "experiments/checkpoints",
) -> Tuple[VLAModel, Dict]:
    """
    Convenience function to train VLA model from demos.

    Args:
        demo_path: Path to saved demonstrations
        model_config: Model configuration name
        num_epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate
        val_split: Fraction of data for validation
        device: Device to train on
        checkpoint_dir: Directory for checkpoints

    Returns:
        Trained model and training history
    """
    print("Loading demonstrations...")
    dataset = RobotDemoDataset.from_file(demo_path)

    # Split into train/val
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size

    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size]
    )

    print(f"Train size: {train_size}, Val size: {val_size}")

    # Create dataloaders
    train_loader = create_dataloader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = create_dataloader(val_dataset, batch_size=batch_size, shuffle=False)

    # Create model
    print("\nCreating model...")
    model = create_vla_model(model_config)

    # Create trainer
    trainer = VLATrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=learning_rate,
        device=device,
        checkpoint_dir=checkpoint_dir,
    )

    # Train
    history = trainer.train(num_epochs=num_epochs)

    return model, history


# =============================================================================
# Quick test
# =============================================================================

if __name__ == "__main__":
    print("Testing Trainer")
    print("=" * 40)

    # This would require real data and model
    print("Note: Full test requires collected demos")
    print("Run 'python train.py' for full training")
