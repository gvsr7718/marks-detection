"""
Digit Recognizer Module — Stage 2
AlexNet-based CNN for handwritten digit recognition (0-9).

Based on: "AI-Powered Mark Recognition in Assessment and Attainment Calculation"
         by J. Annrose et al. (ICTACT, Jan 2025)

The paper uses AlexNet architecture (Fig.3):
- 5 Convolution layers + 3 Pooling layers + 2 Fully Connected layers
- ReLU activation (intermediate), Softmax (output)
- Adam optimizer, Cross-entropy loss
- Trained on MNIST + EMNIST datasets
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from PIL import Image
import torchvision.transforms as transforms


class AlexNetDigit(nn.Module):
    """
    AlexNet-adapted architecture for digit recognition.
    
    Modified for 28×28 grayscale input (MNIST format) instead of
    the original 224×224 RGB input.
    
    Architecture (Paper Fig.3):
        Conv → Pool → Conv → Pool → Conv → Conv → Conv → Pool → FC → FC → Softmax
    """
    
    def __init__(self, num_classes=10):
        super(AlexNetDigit, self).__init__()
        
        self.features = nn.Sequential(
            # Conv1: 28×28 → 28×28 (padding=2 to maintain size)
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # → 14×14
            
            # Conv2: 14×14 → 14×14
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # → 7×7
            
            # Conv3
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # Conv4
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # Conv5
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # → 3×3
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(64 * 3 * 3, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )
    
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
    def predict(self, x):
        """
        Run inference and return predicted digit and confidence.
        
        Args:
            x: tensor of shape (1, 1, 28, 28)
        Returns:
            (digit: int, confidence: float)
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = F.softmax(logits, dim=1)
            confidence, predicted = torch.max(probs, 1)
            return predicted.item(), confidence.item()


class DigitRecognizer:
    """
    Singleton digit recognizer that loads the AlexNet model once
    and provides inference methods.
    """
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DigitRecognizer, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def _initialize(self):
        if self._initialized:
            return
        
        print("Initializing AlexNet Digit Recognizer...")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = AlexNetDigit().to(self.device)
        
        model_path = os.path.join("models", "alexnet_digits.pth")
        
        if os.path.exists(model_path):
            print(f"Loading model from {model_path}...")
            checkpoint = torch.load(model_path, map_location=self.device, 
                                    weights_only=True)
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            self.model.eval()
            self.has_model = True
            print("Model loaded successfully.")
        else:
            print(f"Warning: Model not found at {model_path}")
            print("Please run train_model.py first.")
            self.has_model = False
        
        # MNIST normalization
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        import easyocr
        import cv2
        print("Loading EasyOCR model...")
        self.reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available(), verbose=False)
        
        self._initialized = True
    
    def recognize_digit(self, digit_28x28: np.ndarray) -> tuple:
        """
        Recognize a single digit from a 28×28 binary image.
        
        Args:
            digit_28x28: numpy array of shape (28, 28)
        
        Returns:
            (digit: int, confidence: float)
        """
        self._initialize()
        
        if not self.has_model:
            return 0, 0.0
        
        # Convert to PIL Image for transforms
        img_pil = Image.fromarray(digit_28x28.astype(np.uint8))
        tensor = self.transform(img_pil).unsqueeze(0).to(self.device)
        
        digit, confidence = self.model.predict(tensor)
        return digit, confidence
    
    def recognize_marks_from_cell(self, digit_images: list) -> tuple:
        """
        Recognize marks from a list of 28×28 digit images extracted from a cell.
        
        Combines individual digit predictions into a multi-digit number.
        
        Args:
            digit_images: list of 28×28 numpy arrays
        
        Returns:
            (mark_value: int, avg_confidence: float)
        """
        self._initialize()
        
        if not digit_images:
            return 0, 0.0
        
        digits = []
        total_conf = 0.0
        
        for img in digit_images:
            digit, conf = self.recognize_digit(img)
            digits.append(digit)
            total_conf += conf
        
        # Combine digits: e.g., [1, 2] → 12
        if len(digits) == 1:
            mark_value = digits[0]
        else:
            mark_str = ''.join(str(d) for d in digits)
            try:
                mark_value = int(mark_str)
            except ValueError:
                mark_value = 0
        
        avg_conf = total_conf / len(digits) if digits else 0.0
        return mark_value, avg_conf
    
    def recognize_ht_number(self, ht_row_data, box_images=None) -> tuple:
        """
        Recognize HT number from the full row image with cell borders erased.
        Uses 4x upscaled full-row EasyOCR for best contextual recognition.
        
        Args:
            ht_row_data: tuple of (full_row_crop, cell_coords) for the row image.
            box_images: unused, kept for API compatibility.
        """
        import cv2
        self._initialize()
        
        if ht_row_data is None:
            return "", 0.0
        
        full_row_crop, cell_coords = ht_row_data
        if full_row_crop is None or full_row_crop.size == 0:
            return "", 0.0
        
        # Binarize: black text on white background
        _, binary = cv2.threshold(full_row_crop, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Erase the printed cell borders by drawing white lines over them
        h_img = binary.shape[0]
        for (cx, cy, cw, ch) in cell_coords:
            cv2.line(binary, (cx, 0), (cx, h_img), 255, 4)
            cv2.line(binary, (cx + cw, 0), (cx + cw, h_img), 255, 4)
        cv2.line(binary, (0, 0), (binary.shape[1], 0), 255, 4)
        cv2.line(binary, (0, h_img - 1), (binary.shape[1], h_img - 1), 255, 4)
        
        # Add white border padding
        bordered = cv2.copyMakeBorder(binary, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=255)
        
        # Save first debug image (before upscale)
        cv2.imwrite("debug_output/ht_full_row_cleaned.jpg", bordered)
        
        # Upscale 4x for better OCR recognition
        big = cv2.resize(bordered, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
        
        # Re-binarize after upscale to sharpen edges
        _, big_binary = cv2.threshold(big, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Add extra border after upscale
        final = cv2.copyMakeBorder(big_binary, 30, 30, 30, 30, cv2.BORDER_CONSTANT, value=255)
        
        # Use EasyOCR on the high-resolution cleaned row
        allowlist = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        rgb = cv2.cvtColor(final, cv2.COLOR_GRAY2RGB)
        results = self.reader.readtext(rgb, allowlist=allowlist)
        
        ht_text = ""
        total_conf = 0.0
        for _, text, conf in results:
            ht_text += text.strip().upper()
            total_conf += conf
        
        avg_conf = total_conf / len(results) if results else 0.0
        
        # Truncate to 10 characters
        if len(ht_text) > 10:
            ht_text = ht_text[:10]
        
        print(f"[DEBUG] Final HT Number: '{ht_text}', conf={avg_conf:.2f}")
        return ht_text, avg_conf
    
    def recognize_score(self, score_roi: np.ndarray) -> tuple:
        """
        Recognize MCQ score from the score region using EasyOCR.
        """
        import cv2
        if score_roi is None or score_roi.size == 0:
            return 0, 0.0
        
        try:
            roi_rgb = cv2.cvtColor(score_roi, cv2.COLOR_GRAY2RGB) if len(score_roi.shape) == 2 else score_roi
            results = self.reader.readtext(roi_rgb, allowlist='0123456789/')
            
            if results:
                best = max(results, key=lambda x: x[2])
                text = str(best[1]).strip()
                if '/' in text:
                    score_str = text.split('/')[0].strip()
                else:
                    score_str = text.strip()
                
                if score_str.isdigit():
                    return int(score_str), float(best[2])
            return 0, 0.0
        except Exception:
            return 0, 0.0


def get_digit_recognizer() -> DigitRecognizer:
    """Global accessor for the singleton DigitRecognizer."""
    return DigitRecognizer()
