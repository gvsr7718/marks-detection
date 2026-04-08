"""
Step 1: Data Preparation for Objective Sheet Marks Detection
============================================================
Extracts the red-ink score region (e.g., "7/10") from the top-right corner
of each objective answer sheet image.

Outputs cropped, cleaned score images to obj_dataset/unlabeled/ for labeling.

Usage:
    python prepare_data.py
    python prepare_data.py --input_dir ../Obj_sheets_JPG --debug
"""

import cv2
import numpy as np
import os
import argparse
import glob


def isolate_red_ink(image_bgr):
    """
    Isolate red ink from a BGR image using dual HSV + RGB dominance masking.
    Returns a binary mask where red ink pixels are 255.
    """
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    # Red wraps around hue=0 in HSV, so we need two ranges
    # Range 1: hue 0-15 (orange-red to red)
    lower_red1 = np.array([0, 30, 30])
    upper_red1 = np.array([15, 255, 255])
    # Range 2: hue 160-180 (red to magenta-red)
    lower_red2 = np.array([155, 30, 30])
    upper_red2 = np.array([180, 255, 255])

    mask_hsv1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_hsv2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_hsv = cv2.bitwise_or(mask_hsv1, mask_hsv2)

    # RGB dominance: Red channel must be significantly higher than Green and Blue
    b, g, r = cv2.split(image_bgr)
    r_int = r.astype(np.int16)
    g_int = g.astype(np.int16)
    b_int = b.astype(np.int16)
    rgb_mask = ((r_int - g_int > 25) & (r_int - b_int > 25) & (r_int > 80))
    rgb_mask = (rgb_mask * 255).astype(np.uint8)

    # Combine both strategies
    combined = cv2.bitwise_or(mask_hsv, rgb_mask)
    return combined


def extract_score_region(image_path, output_dir="obj_dataset/unlabeled",
                         debug_dir=None):
    """
    Extract the red-ink score from the top-right corner of an objective sheet.

    Steps:
        1. Crop top-right corner (top 30% height × right 40% width)
        2. Isolate red ink
        3. Find bounding box of largest red-ink cluster
        4. Crop and save the score region

    Returns:
        (output_path, score_crop) or (None, None) on failure
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"  [-] Failed to load: {image_path}")
        return None, None

    H, W = img.shape[:2]
    basename = os.path.splitext(os.path.basename(image_path))[0]

    # --- Step 1: Crop top-right corner ---
    # The score mark (e.g., circled "7/10") is always in the top-right area.
    # We target top 35% of height, right 45% of width to be generous.
    crop_top = 0
    crop_bottom = int(H * 0.35)
    crop_left = int(W * 0.55)
    crop_right = W

    roi = img[crop_top:crop_bottom, crop_left:crop_right]

    if debug_dir:
        cv2.imwrite(os.path.join(debug_dir, f"{basename}_01_roi.png"), roi)

    # --- Step 2: Isolate red ink ---
    red_mask = isolate_red_ink(roi)

    # Morphological cleanup: close gaps in pen strokes, then dilate slightly
    kernel_close = np.ones((5, 5), np.uint8)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel_close)

    kernel_dilate = np.ones((3, 3), np.uint8)
    red_mask = cv2.dilate(red_mask, kernel_dilate, iterations=1)

    if debug_dir:
        cv2.imwrite(os.path.join(debug_dir, f"{basename}_02_red_mask.png"), red_mask)

    # --- Step 3: Find contours of red ink regions ---
    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print(f"  [-] No red ink found in ROI of {basename}")
        return None, None

    # Filter out tiny noise contours (< 100 pixels area)
    valid_contours = [c for c in contours if cv2.contourArea(c) > 100]

    if not valid_contours:
        print(f"  [-] Red ink too small/noisy in {basename}")
        return None, None

    # Find the bounding box that encloses all valid red contours
    # This captures the entire "7/10" or circled score region
    all_points = np.vstack(valid_contours)
    x, y, w, h = cv2.boundingRect(all_points)

    # Add generous padding
    pad = 25
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(roi.shape[1], x + w + pad)
    y2 = min(roi.shape[0], y + h + pad)

    # --- Step 4: Create the final score crop ---
    # We create a clean version: red ink → black on white background
    # This removes printed black text that might confuse the model
    red_ink_clean = np.full(roi.shape[:2], 255, dtype=np.uint8)  # white bg
    red_ink_clean[red_mask > 0] = 0  # red ink becomes black

    score_crop = red_ink_clean[y1:y2, x1:x2]

    if score_crop.size == 0 or score_crop.shape[0] < 10 or score_crop.shape[1] < 10:
        print(f"  [-] Score crop too small in {basename}")
        return None, None

    # Also save the original color crop for visual reference
    color_crop = roi[y1:y2, x1:x2]

    if debug_dir:
        cv2.imwrite(os.path.join(debug_dir, f"{basename}_03_score_color.png"), color_crop)
        cv2.imwrite(os.path.join(debug_dir, f"{basename}_04_score_clean.png"), score_crop)

    # Save the clean version for labeling/training
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{basename}_score.png")
    cv2.imwrite(output_path, score_crop)

    print(f"  [+] {basename} → score region saved ({score_crop.shape[1]}×{score_crop.shape[0]} px)")
    return output_path, score_crop


def batch_extract(input_dir, output_dir="obj_dataset/unlabeled", debug=False):
    """
    Process all images in the input directory.
    """
    debug_dir = "obj_dataset/debug" if debug else None
    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)

    # Find all JPG images
    patterns = [os.path.join(input_dir, "*.jpg"),
                os.path.join(input_dir, "*.jpeg"),
                os.path.join(input_dir, "*.png")]
    image_paths = []
    for p in patterns:
        image_paths.extend(glob.glob(p))

    image_paths.sort()

    if not image_paths:
        print(f"[-] No images found in {input_dir}")
        return

    print(f"[*] Found {len(image_paths)} images in {input_dir}")
    print(f"[*] Output directory: {output_dir}")
    print("=" * 60)

    success = 0
    failed = 0

    for img_path in image_paths:
        result, _ = extract_score_region(img_path, output_dir, debug_dir)
        if result:
            success += 1
        else:
            failed += 1

    print("=" * 60)
    print(f"[*] Done! Success: {success}/{len(image_paths)}, Failed: {failed}")
    print(f"[*] Extracted score images saved to: {output_dir}")
    if debug_dir:
        print(f"[*] Debug visualizations saved to: {debug_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract score regions from objective sheets")
    parser.add_argument("--input_dir", default="../Obj_sheets_JPG",
                        help="Directory containing objective sheet images")
    parser.add_argument("--output_dir", default="obj_dataset/unlabeled",
                        help="Output directory for extracted score images")
    parser.add_argument("--debug", action="store_true",
                        help="Save debug visualizations")
    args = parser.parse_args()

    batch_extract(args.input_dir, args.output_dir, args.debug)
