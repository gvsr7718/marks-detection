"""
Step 5: Inference & Batch Testing for ObjNet
=============================================
Tests the trained ObjNet model on objective sheet images.

Can process:
    - A single image
    - An entire directory of images (batch mode)

Usage:
    python test_obj_model.py --image ../Obj_sheets_JPG/IMG_1566.jpg
    python test_obj_model.py --batch ../Obj_sheets_JPG
    python test_obj_model.py --batch ../Obj_sheets_JPG --debug
"""

import cv2
import numpy as np
import os
import glob
import argparse
import sys

# Add parent directory to path so we can import from obj_model
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from obj_net import ObjMarksRecognizer
from prepare_data import isolate_red_ink


def extract_and_predict(image_path, recognizer, debug_dir=None):
    """
    Full pipeline: extract red-ink score region → predict marks.
    
    Args:
        image_path: path to objective sheet image
        recognizer: ObjMarksRecognizer instance
        debug_dir: optional debug output directory
        
    Returns:
        (score: int or None, confidence: float)
    """
    img = cv2.imread(image_path)
    if img is None:
        return None, 0.0

    basename = os.path.splitext(os.path.basename(image_path))[0]
    H, W = img.shape[:2]

    # 1. Crop top-right corner
    crop_top = 0
    crop_bottom = int(H * 0.35)
    crop_left = int(W * 0.55)
    crop_right = W
    roi = img[crop_top:crop_bottom, crop_left:crop_right]

    # 2. Isolate red ink
    red_mask = isolate_red_ink(roi)

    # Morphological cleanup
    kernel_close = np.ones((5, 5), np.uint8)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel_close)
    kernel_dilate = np.ones((3, 3), np.uint8)
    red_mask = cv2.dilate(red_mask, kernel_dilate, iterations=1)

    # 3. Find contours
    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    valid_contours = [c for c in contours if cv2.contourArea(c) > 100]

    if not valid_contours:
        return None, 0.0

    # Find cohesive bounding box (ignores stray marks far from the main circle)
    largest_c = max(valid_contours, key=cv2.contourArea)
    M = cv2.moments(largest_c)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
    else:
        bx, by, bw, bh = cv2.boundingRect(largest_c)
        cx, cy = bx + bw // 2, by + bh // 2

    cohesive_contours = []
    for c in valid_contours:
        bx, by, bw, bh = cv2.boundingRect(c)
        bcx, bcy = bx + bw / 2, by + bh / 2
        dist = np.sqrt((bcx - cx)**2 + (bcy - cy)**2)
        if dist < 120:  # Threshold to group the score circle and ignore distant stray marks
            cohesive_contours.append(c)

    if not cohesive_contours:
        cohesive_contours = [largest_c]

    all_points = np.vstack(cohesive_contours)
    x, y, w, h = cv2.boundingRect(all_points)

    pad = 15
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(roi.shape[1], x + w + pad)
    y2 = min(roi.shape[0], y + h + pad)

    # Create clean score crop
    red_ink_clean = np.full(roi.shape[:2], 255, dtype=np.uint8)
    red_ink_clean[red_mask > 0] = 0
    score_crop = red_ink_clean[y1:y2, x1:x2]

    if score_crop.size == 0 or score_crop.shape[0] < 10 or score_crop.shape[1] < 10:
        return None, 0.0

    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)
        cv2.imwrite(os.path.join(debug_dir, f"{basename}_score_crop.png"), score_crop)

    # 4. Predict using ObjNet
    score, confidence = recognizer.predict_from_image(score_crop)
    return score, confidence


def test_single(image_path, model_path=None, debug=False):
    """Test on a single image."""
    recognizer = ObjMarksRecognizer(model_path)

    if not recognizer.model_loaded:
        print("[-] Model not loaded. Train first!")
        return

    debug_dir = "test_debug" if debug else None
    score, conf = extract_and_predict(image_path, recognizer, debug_dir)

    basename = os.path.basename(image_path)
    if score is not None:
        print(f"\n{'='*50}")
        print(f"  {basename}")
        print(f"  Predicted Score: {score}/10  (confidence: {conf:.2f})")
        print(f"{'='*50}")
    else:
        print(f"\n  {basename} -> FAILED to detect marks")


def test_batch(batch_dir, model_path=None, debug=False):
    """Test on all images in a directory."""
    recognizer = ObjMarksRecognizer(model_path)

    if not recognizer.model_loaded:
        print("[-] Model not loaded. Train first!")
        return

    # Find images
    patterns = [os.path.join(batch_dir, "*.jpg"),
                os.path.join(batch_dir, "*.jpeg"),
                os.path.join(batch_dir, "*.png")]
    image_paths = []
    for p in patterns:
        image_paths.extend(glob.glob(p))
    image_paths.sort()

    if not image_paths:
        print(f"[-] No images found in {batch_dir}")
        return

    debug_dir = "test_debug" if debug else None

    print(f"\n{'='*60}")
    print(f"  ObjNet BATCH TEST -- {len(image_paths)} images")
    print(f"{'='*60}\n")

    results = []
    success = 0
    failed = 0

    for img_path in image_paths:
        basename = os.path.basename(img_path)
        score, conf = extract_and_predict(img_path, recognizer, debug_dir)

        if score is not None:
            results.append((basename, score, conf))
            status = f"[OK] {score:2d}/10  (conf: {conf:.2f})"
            success += 1
        else:
            results.append((basename, None, 0.0))
            status = "[FAIL]"
            failed += 1

        print(f"  {basename:20s} -> {status}")

    # Summary
    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    print(f"  Total:   {len(image_paths)}")
    print(f"  Success: {success}")
    print(f"  Failed:  {failed}")

    if success > 0:
        scores = [r[1] for r in results if r[1] is not None]
        avg_conf = np.mean([r[2] for r in results if r[1] is not None])
        print(f"  Avg Conf: {avg_conf:.2f}")
        print(f"  Score Distribution:")
        for s in range(11):
            count = scores.count(s)
            if count > 0:
                bar = "#" * count
                print(f"    {s:2d}/10: {count:2d} {bar}")

    print(f"{'='*60}\n")

    # Save results to file
    report_path = "obj_test_results.txt"
    with open(report_path, "w") as f:
        f.write(f"ObjNet Batch Test Results\n")
        f.write(f"{'='*50}\n")
        f.write(f"Total: {len(image_paths)}, Success: {success}, Failed: {failed}\n\n")
        for basename, score, conf in results:
            if score is not None:
                f.write(f"{basename:20s} -> {score}/10 (conf: {conf:.2f})\n")
            else:
                f.write(f"{basename:20s} -> FAILED\n")
    print(f"  Results saved to: {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test ObjNet on objective sheets")
    parser.add_argument("--image", help="Path to a single image")
    parser.add_argument("--batch", help="Path to directory of images")
    parser.add_argument("--model", help="Path to model weights", default=None)
    parser.add_argument("--debug", action="store_true", help="Save debug output")
    args = parser.parse_args()

    if args.image:
        test_single(args.image, args.model, args.debug)
    elif args.batch:
        test_batch(args.batch, args.model, args.debug)
    else:
        print("Usage:")
        print("  python test_obj_model.py --image path/to/image.jpg")
        print("  python test_obj_model.py --batch path/to/directory/")
