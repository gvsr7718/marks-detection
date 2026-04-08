"""
Step 3: ObjNet — CNN Architecture for Objective Sheet Marks Detection
=====================================================================
A lightweight CNN designed to classify handwritten marks (0-10) written
in red ink on objective answer sheets.

11 output classes: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10

This is COMPLETELY SEPARATE from the AlexNet model used for descriptive sheets.

Architecture:
    Input: 1×64×64 grayscale
    Conv1(1→32)  → BN → ReLU → MaxPool → 32×32×32
    Conv2(32→64) → BN → ReLU → MaxPool → 64×16×16
    Conv3(64→128) → BN → ReLU → MaxPool → 128×8×8
    Conv4(128→128) → BN → ReLU → MaxPool → 128×4×4
    Flatten → FC(2048→256) → FC(256→64) → FC(64→11) → Softmax
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import os


class ObjNet(nn.Module):
    """
    Lightweight CNN for objective marks classification (0-10).
    
    Input: 1-channel 64×64 grayscale image
    Output: 11 classes (scores 0 through 10)
    """

    def __init__(self, num_classes=11):
        super(ObjNet, self).__init__()

        self.features = nn.Sequential(
            # Block 1: 64×64 → 32×32
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 2: 32×32 → 16×16
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 3: 16×16 → 8×8
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 4: 8×8 → 4×4
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def predict(self, x):
        """
        Run inference on a single input tensor.
        
        Args:
            x: tensor of shape (1, 1, 64, 64)
        Returns:
            (predicted_class: int, confidence: float)
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = F.softmax(logits, dim=1)
            confidence, predicted = torch.max(probs, 1)
            return predicted.item(), confidence.item()


class ObjMarksRecognizer:
    """
    Standalone recognizer for objective sheet marks.
    Loads the ObjNet model and provides a simple inference API.
    
    Usage:
        recognizer = ObjMarksRecognizer()
        score, confidence = recognizer.predict_from_image(score_crop_gray)
    """

    def __init__(self, model_path=None):
        """
        Initialize the recognizer.
        
        Args:
            model_path: Path to the trained model weights. 
                        Defaults to obj_model/models/obj_marks_model.pth
        """
        if model_path is None:
            # Try relative path first (when running from obj_model/)
            model_path = os.path.join("models", "obj_marks_model.pth")
            if not os.path.exists(model_path):
                # Try from project root
                model_path = os.path.join("obj_model", "models", "obj_marks_model.pth")

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = ObjNet(num_classes=11).to(self.device)
        self.model_loaded = False

        if os.path.exists(model_path):
            print(f"[ObjNet] Loading model from {model_path}...")
            checkpoint = torch.load(model_path, map_location=self.device,
                                    weights_only=True)
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            self.model.eval()
            self.model_loaded = True
            print(f"[ObjNet] Model loaded successfully on {self.device}")
        else:
            print(f"[ObjNet] WARNING: Model not found at {model_path}")
            print(f"[ObjNet] Run train_obj_model.py first!")

        # Transform: resize to 64×64, normalize
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

    def predict_from_image(self, gray_image):
        """
        Predict the score from a grayscale score region image.
        
        Args:
            gray_image: numpy array (H, W) grayscale — white bg, black digits
                        
        Returns:
            (score: int 0-10, confidence: float 0-1)
        """
        if not self.model_loaded:
            return None, 0.0

        if gray_image is None or gray_image.size == 0:
            return None, 0.0

        # Ensure single channel
        if len(gray_image.shape) == 3:
            import cv2
            gray_image = cv2.cvtColor(gray_image, cv2.COLOR_BGR2GRAY)

        # Preprocess exactly like the training dataset (aspect-ratio 50x50 padded to 64x64)
        import cv2
        h, w = gray_image.shape
        scale = 50.0 / max(h, w)
        new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
        resized = cv2.resize(gray_image, (new_w, new_h), interpolation=cv2.INTER_AREA)

        canvas = np.full((64, 64), 255, dtype=np.uint8)
        x_off = (64 - new_w) // 2
        y_off = (64 - new_h) // 2
        canvas[y_off:y_off+new_h, x_off:x_off+new_w] = resized

        # Use minimalist transform to avoid double resizing
        basic_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        # Convert to PIL for transforms
        pil_img = Image.fromarray(canvas, mode='L')
        tensor = basic_transform(pil_img).unsqueeze(0).to(self.device)

        score, confidence = self.model.predict(tensor)
        return score, confidence

    def predict_from_file(self, image_path):
        """
        Predict from an image file path.
        
        Args:
            image_path: path to a grayscale score crop image
            
        Returns:
            (score: int 0-10, confidence: float 0-1)
        """
        import cv2
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None, 0.0
        return self.predict_from_image(img)


def get_obj_recognizer(model_path=None):
    """Global accessor for the ObjMarksRecognizer."""
    return ObjMarksRecognizer(model_path)
