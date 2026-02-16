"""
VLA Model Training Script
=========================
Train the Vision-Language-Action model using behavior cloning.

Usage:
    # Collect demos first (if not already done)
    python collect_demos.py --num_demos 10 --output data/demos.pkl

    # Train model
    python train.py --demos data/demos.pkl --epochs 50

    # Resume training
    python train.py --demos data/demos.pkl --resume experiments/checkpoints/epoch_20.pt
"""

import argparse
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import torch
from pathlib import Path

# Import our modules
from models.vla_model import create_vla_model
from data.dataset import RobotDemoDataset, create_dataloader
from training.trainer import VLATrainer, train_vla_model
from torch.utils.data import random_split


def main():
    parser = argparse.ArgumentParser(description="Train VLA model")
    parser.add_argument(
        "--demos",
        type=str,
        default="data/demos.pkl",
        help="Path to demonstration data"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Training batch size"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--model_config",
        type=str,
        default="frozen",
        choices=["default", "small", "large", "frozen"],
        help="Model configuration"
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="experiments/checkpoints",
        help="Directory for checkpoints"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device: 'cpu', 'cuda', 'mps', or 'auto'"
    )

    args = parser.parse_args()

    # Auto-detect device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device

    print("=" * 60)
    print("VLA Model Training")
    print("=" * 60)
    print(f"  Demos: {args.demos}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Model config: {args.model_config}")
    print(f"  Device: {device}")
    print()

    # Check if demos exist
    if not Path(args.demos).exists():
        print(f"Error: Demos not found at {args.demos}")
        print("Run: python collect_demos.py --num_demos 10 --output data/demos.pkl")
        sys.exit(1)

    # Load dataset
    print("[1/4] Loading demonstrations...")
    dataset = RobotDemoDataset.from_file(args.demos)
    print(f"  Total transitions: {len(dataset)}")

    # Split into train/val
    val_size = max(1, int(len(dataset) * 0.1))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    print(f"  Train: {train_size}, Val: {val_size}")

    # Create dataloaders
    train_loader = create_dataloader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,  # Avoid multiprocessing issues on macOS
    )
    val_loader = create_dataloader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    # Create model
    print("\n[2/4] Creating model...")
    model = create_vla_model(args.model_config)

    # Create trainer
    print("\n[3/4] Setting up trainer...")
    trainer = VLATrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=args.lr,
        device=device,
        checkpoint_dir=args.checkpoint_dir,
    )

    # Resume from checkpoint if specified
    if args.resume:
        print(f"  Resuming from: {args.resume}")
        trainer.load_checkpoint(args.resume)

    # Train
    print("\n[4/4] Training...")
    history = trainer.train(
        num_epochs=args.epochs,
        save_every=10,
        early_stopping_patience=20,
    )

    # Print final results
    print("\n" + "=" * 60)
    print("Training Summary")
    print("=" * 60)
    print(f"  Final train loss: {history['train_loss'][-1]:.4f}")
    if history['val_loss']:
        print(f"  Final val loss: {history['val_loss'][-1]:.4f}")
        print(f"  Best val loss: {min(history['val_loss']):.4f}")
    print(f"  Checkpoints: {args.checkpoint_dir}")
    print("\nTo evaluate, run:")
    print(f"  python evaluate.py --checkpoint {args.checkpoint_dir}/best.pt")


if __name__ == "__main__":
    main()
