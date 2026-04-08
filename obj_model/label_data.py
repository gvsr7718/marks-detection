"""
Step 2: Labeling Tool for Objective Sheet Score Images
======================================================
Opens each extracted score image and lets you label it with a keypress.

Controls:
    0-9  → Label as that digit (marks out of 10)
    t    → Label as 10 (ten)
    s    → Skip (unclear/bad image)
    q    → Quit labeling

The labeled images are moved into class folders:
    obj_dataset/0/, obj_dataset/1/, ..., obj_dataset/10/

Usage:
    python label_data.py
    python label_data.py --input_dir obj_dataset/unlabeled
"""

import cv2
import os
import shutil
import glob
import argparse


def label_images(input_dir="obj_dataset/unlabeled", output_base="obj_dataset"):
    """
    Interactive labeling tool using OpenCV windows.
    """
    # Create class directories (0 through 10)
    for i in range(11):
        os.makedirs(os.path.join(output_base, str(i)), exist_ok=True)

    skip_dir = os.path.join(output_base, "skipped")
    os.makedirs(skip_dir, exist_ok=True)

    # Find all images
    image_paths = sorted(glob.glob(os.path.join(input_dir, "*.png")))

    if not image_paths:
        print(f"[-] No images found in {input_dir}")
        print("    Run prepare_data.py first to extract score regions.")
        return

    print("=" * 60)
    print("  OBJECTIVE SHEET SCORE LABELING TOOL")
    print("=" * 60)
    print(f"  Images to label: {len(image_paths)}")
    print(f"  Input:  {input_dir}")
    print(f"  Output: {output_base}/[0-10]/")
    print()
    print("  Controls:")
    print("    0-9  → Label as that digit")
    print("    t    → Label as 10")
    print("    s    → Skip (unclear)")
    print("    q    → Quit")
    print("=" * 60)

    labeled = 0
    skipped = 0

    for idx, img_path in enumerate(image_paths):
        basename = os.path.basename(img_path)
        img = cv2.imread(img_path)

        if img is None:
            print(f"  [-] Could not load: {basename}")
            continue

        # Resize for display (make it larger so you can see the handwriting)
        display_h = 300
        scale = display_h / img.shape[0]
        display_w = int(img.shape[1] * scale)
        display_img = cv2.resize(img, (display_w, display_h),
                                 interpolation=cv2.INTER_CUBIC)

        # Add info text above the image
        canvas_h = display_h + 80
        canvas = np.full((canvas_h, max(display_w, 400), 3), 240, dtype=np.uint8)
        
        # Header text
        cv2.putText(canvas, f"[{idx+1}/{len(image_paths)}] {basename}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        cv2.putText(canvas, "Press 0-9, t=10, s=skip, q=quit",
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)

        # Place image
        y_offset = 65
        canvas[y_offset:y_offset + display_h, 0:display_w] = display_img

        cv2.imshow("Label Score", canvas)

        while True:
            key = cv2.waitKey(0) & 0xFF

            if key == ord('q'):
                print(f"\n[*] Quit. Labeled: {labeled}, Skipped: {skipped}")
                cv2.destroyAllWindows()
                return

            elif key == ord('s'):
                dest = os.path.join(skip_dir, basename)
                shutil.move(img_path, dest)
                skipped += 1
                print(f"  [SKIP] {basename}")
                break

            elif key == ord('t'):
                # Label as 10
                dest = os.path.join(output_base, "10", basename)
                shutil.move(img_path, dest)
                labeled += 1
                print(f"  [10] {basename}")
                break

            elif chr(key).isdigit():
                digit = chr(key)
                dest = os.path.join(output_base, digit, basename)
                shutil.move(img_path, dest)
                labeled += 1
                print(f"  [{digit}] {basename}")
                break

            else:
                # Invalid key, ignore
                pass

    cv2.destroyAllWindows()
    print("=" * 60)
    print(f"[*] Labeling complete! Labeled: {labeled}, Skipped: {skipped}")
    print(f"[*] Class distribution:")
    for i in range(11):
        class_dir = os.path.join(output_base, str(i))
        count = len(glob.glob(os.path.join(class_dir, "*.png")))
        if count > 0:
            print(f"    Score {i:2d}: {count} images")
    print("=" * 60)


if __name__ == "__main__":
    import numpy as np  # needed for canvas creation
    
    parser = argparse.ArgumentParser(description="Label objective sheet score images")
    parser.add_argument("--input_dir", default="obj_dataset/unlabeled",
                        help="Directory with unlabeled score images")
    parser.add_argument("--output_base", default="obj_dataset",
                        help="Base output directory for class folders")
    args = parser.parse_args()

    label_images(args.input_dir, args.output_base)
