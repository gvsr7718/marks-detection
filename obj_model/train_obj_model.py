"""
Step 4: Training Pipeline for ObjNet
=====================================
Trains the ObjNet model on labeled objective sheet score images.

Features:
    - Heavy data augmentation to overcome small dataset (71 images)
    - Synthetic handwritten digit generation for additional training data
    - Adam optimizer with StepLR scheduler
    - Saves best model based on validation accuracy

Usage:
    python train_obj_model.py
    python train_obj_model.py --epochs 50 --batch_size 16
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, ConcatDataset, random_split
import torchvision.transforms as transforms
import numpy as np
import cv2
import os
import glob
import time
import random

from obj_net import ObjNet


# ─────────────────────────────────────────────────────────────────────────────
#  DATASET CLASSES
# ─────────────────────────────────────────────────────────────────────────────

class ObjScoreDataset(Dataset):
    """
    Dataset of labeled objective sheet score images.
    
    Expected directory structure:
        obj_dataset/
            0/  *.png
            1/  *.png
            ...
            10/ *.png
    """

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []  # list of (image_path, label)

        for label in range(11):
            class_dir = os.path.join(root_dir, str(label))
            if not os.path.exists(class_dir):
                continue
            for img_path in glob.glob(os.path.join(class_dir, "*.png")):
                self.samples.append((img_path, label))

        if not self.samples:
            print(f"[WARNING] No labeled images found in {root_dir}")
            print(f"          Run prepare_data.py and label_data.py first!")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Load as grayscale
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            # Fallback: return a blank image
            img = np.zeros((64, 64), dtype=np.uint8)

        # Resize maintaining aspect ratio to fit inside 50x50, then pad to 64x64
        h, w = img.shape
        scale = 50.0 / max(h, w)
        new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        canvas = np.full((64, 64), 255, dtype=np.uint8)
        x_off = (64 - new_w) // 2
        y_off = (64 - new_h) // 2
        canvas[y_off:y_off+new_h, x_off:x_off+new_w] = resized
        img = canvas

        # Convert to PIL for torchvision transforms
        from PIL import Image
        pil_img = Image.fromarray(img, mode='L')

        if self.transform:
            tensor = self.transform(pil_img)
        else:
            tensor = transforms.ToTensor()(pil_img)

        return tensor, label


class SyntheticScoreDataset(Dataset):
    """
    Generates synthetic handwritten digit images for scores 0-10.
    
    Uses OpenCV to render digits in various styles, sizes, and orientations
    to augment the small real dataset.
    """

    def __init__(self, num_per_class=200, transform=None):
        self.num_per_class = num_per_class
        self.transform = transform
        self.total = num_per_class * 11  # 11 classes (0-10)
        
        # OpenCV font faces for variety
        self.fonts = [
            cv2.FONT_HERSHEY_SIMPLEX,
            cv2.FONT_HERSHEY_DUPLEX,
            cv2.FONT_HERSHEY_COMPLEX,
            cv2.FONT_HERSHEY_TRIPLEX,
            cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
            cv2.FONT_HERSHEY_SCRIPT_COMPLEX,
        ]

    def __len__(self):
        return self.total

    def __getitem__(self, idx):
        label = idx % 11
        text = str(label)

        # Create a blank white image
        img = np.full((64, 64), 255, dtype=np.uint8)

        # Random style variations
        font = random.choice(self.fonts)
        font_scale = random.uniform(1.0, 2.2)
        thickness = random.randint(1, 3)

        # Get text size for centering
        (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)

        # Random position (roughly centered with jitter)
        cx = (64 - tw) // 2 + random.randint(-8, 8)
        cy = (64 + th) // 2 + random.randint(-8, 8)
        cx = max(0, min(cx, 64 - tw))
        cy = max(th, min(cy, 63))

        # Draw black digit on white background
        cv2.putText(img, text, (cx, cy), font, font_scale, 0, thickness)

        # Add random noise
        if random.random() > 0.3:
            noise = np.random.normal(0, random.uniform(5, 25), img.shape).astype(np.int16)
            img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        # Random slight rotation
        if random.random() > 0.3:
            angle = random.uniform(-20, 20)
            M = cv2.getRotationMatrix2D((32, 32), angle, 1.0)
            img = cv2.warpAffine(img, M, (64, 64), borderValue=255)

        # Random Gaussian blur (simulates camera blur)
        if random.random() > 0.5:
            ksize = random.choice([3, 5])
            img = cv2.GaussianBlur(img, (ksize, ksize), 0)

        # Sometimes add a circle around the digit (like the teacher does)
        if random.random() > 0.5:
            center = (32 + random.randint(-5, 5), 32 + random.randint(-5, 5))
            radius = random.randint(20, 30)
            cv2.circle(img, center, radius, 0, thickness=random.randint(1, 2))

        # Sometimes add "/10" text below or beside the main digit
        if random.random() > 0.4:
            small_font_scale = random.uniform(0.4, 0.8)
            slash_text = "/10"
            sx = cx + tw + random.randint(0, 5)
            sy = cy + random.randint(-5, 5)
            if sx + 30 < 64:
                cv2.putText(img, slash_text, (sx, sy), font, small_font_scale, 0, 1)

        # Convert to PIL
        from PIL import Image
        pil_img = Image.fromarray(img, mode='L')

        if self.transform:
            tensor = self.transform(pil_img)
        else:
            tensor = transforms.ToTensor()(pil_img)

        return tensor, label


# ─────────────────────────────────────────────────────────────────────────────
#  TRAINING FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def get_transforms():
    """
    Returns (train_transform, val_transform).
    Heavy augmentation for training, minimal for validation.
    """
    train_transform = transforms.Compose([
        transforms.RandomRotation(20),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.1, 0.1),
            scale=(0.85, 1.15),
            shear=10
        ),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.15)),
    ])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])

    return train_transform, val_transform


def train(data_dir="obj_dataset", epochs=30, batch_size=32, lr=0.001,
          use_synthetic=True, synthetic_per_class=200):
    """
    Train the ObjNet model.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Training on: {device}")

    train_transform, val_transform = get_transforms()

    # 1. Load real labeled data
    real_dataset = ObjScoreDataset(data_dir, transform=train_transform)
    real_val_dataset = ObjScoreDataset(data_dir, transform=val_transform)

    if len(real_dataset) == 0:
        print("[-] No real training data found!")
        print("    Run prepare_data.py and label_data.py first.")
        if not use_synthetic:
            return
        print("[*] Training on synthetic data only...")

    # Split real data into train/val (80/20)
    if len(real_dataset) > 5:
        n_val = max(1, int(len(real_dataset) * 0.2))
        n_train = len(real_dataset) - n_val
        real_train_base, real_val_base = random_split(real_val_dataset, [n_train, n_val])
        
        # MASSIVELY UPSAMPLE REAL DATA (x30) to counter the 2200 synthetic images!
        real_train = ConcatDataset([real_train_base] * 30)
        real_val = ConcatDataset([real_val_base] * 5)
        
        print(f"[*] Real data: {n_train} train (upsampled to {len(real_train)}), {n_val} validation")
    else:
        real_train = real_dataset
        real_val = None
        print(f"[*] Real data: {len(real_dataset)} (all used for training, no validation split)")

    # 2. Generate synthetic data
    if use_synthetic:
        synthetic_train = SyntheticScoreDataset(
            num_per_class=synthetic_per_class,
            transform=train_transform
        )
        synthetic_val = SyntheticScoreDataset(
            num_per_class=max(20, synthetic_per_class // 5),
            transform=val_transform
        )
        print(f"[*] Synthetic data: {len(synthetic_train)} train, {len(synthetic_val)} validation")

        # Combine real + synthetic
        if len(real_dataset) > 0:
            train_dataset = ConcatDataset([real_train, synthetic_train])
        else:
            train_dataset = synthetic_train

        if real_val is not None:
            val_dataset = ConcatDataset([real_val, synthetic_val])
        else:
            val_dataset = synthetic_val
    else:
        train_dataset = real_train
        val_dataset = real_val

    print(f"[*] Total training samples: {len(train_dataset)}")
    if val_dataset:
        print(f"[*] Total validation samples: {len(val_dataset)}")

    # 3. Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=0)
    val_loader = None
    if val_dataset and len(val_dataset) > 0:
        val_loader = DataLoader(val_dataset, batch_size=batch_size,
                                shuffle=False, num_workers=0)

    # 4. Initialize model
    model = ObjNet(num_classes=11).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    best_accuracy = 0.0
    os.makedirs("models", exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  TRAINING ObjNet — {epochs} epochs, batch_size={batch_size}, lr={lr}")
    print(f"{'='*60}\n")

    for epoch in range(1, epochs + 1):
        # --- Training ---
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        start_time = time.time()

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_acc = 100 * correct / total if total > 0 else 0
        avg_loss = running_loss / len(train_loader) if len(train_loader) > 0 else 0
        epoch_time = time.time() - start_time

        # --- Validation ---
        val_acc = 0.0
        if val_loader:
            val_acc = evaluate(model, val_loader, device)

        print(f"  Epoch [{epoch:2d}/{epochs}]  "
              f"Loss: {avg_loss:.4f}  "
              f"Train Acc: {train_acc:.1f}%  "
              f"Val Acc: {val_acc:.1f}%  "
              f"Time: {epoch_time:.1f}s")

        # Save best model (by validation accuracy, or train acc if no val)
        check_acc = val_acc if val_acc > 0 else train_acc
        if check_acc > best_accuracy:
            best_accuracy = check_acc
            save_path = os.path.join("models", "obj_marks_model.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': check_acc,
                'num_classes': 11,
            }, save_path)
            print(f"  ✓ New best model saved! Accuracy: {check_acc:.1f}%")

        scheduler.step()

    print(f"\n{'='*60}")
    print(f"  Training complete! Best accuracy: {best_accuracy:.1f}%")
    print(f"  Model saved to: models/obj_marks_model.pth")
    print(f"{'='*60}")


def evaluate(model, loader, device):
    """Evaluate model on a data loader."""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100 * correct / total if total > 0 else 0.0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train ObjNet for objective marks detection")
    parser.add_argument("--data_dir", default="obj_dataset",
                        help="Directory with labeled score images")
    parser.add_argument("--epochs", type=int, default=30,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Learning rate")
    parser.add_argument("--no_synthetic", action="store_true",
                        help="Disable synthetic data generation")
    parser.add_argument("--synthetic_per_class", type=int, default=200,
                        help="Number of synthetic samples per class")
    args = parser.parse_args()

    train(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        use_synthetic=not args.no_synthetic,
        synthetic_per_class=args.synthetic_per_class,
    )
