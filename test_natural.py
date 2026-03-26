import cv2
import numpy as np
import glob
import os
import easyocr
import re

from table_detector import preprocess_image
from mark_extractor import extract_ht_number_boxes

reader = easyocr.Reader(['en'], gpu=True, verbose=False)

def test_natural_stitched(ht_row_data, box_images):
    full_row_crop, cell_coords = ht_row_data
    
    # We want to create a blank white canvas of the same size as full_row_crop
    canvas = np.full(full_row_crop.shape, 255, dtype=np.uint8)
    
    # Enforce exactly 10 boxes (rightmost 10)
    boxes_to_process = box_images[-10:] if len(box_images) >= 10 else box_images
    coords_to_process = cell_coords[-10:] if len(cell_coords) >= 10 else cell_coords
    
    # For each box, place it back on the canvas at its relative position, but only the ink!
    for i, (box_img, coord) in enumerate(zip(boxes_to_process, coords_to_process)):
        x, y, w, h = coord
        
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(2, 2))
        enhanced = clahe.apply(box_img)
        
        # Threshold the box to get clean ink
        _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Place it on the canvas (x, y are relative to full_row_crop)
        # We replace the canvas patch with the cleaned binary box
        # But wait, `box_img` is cropped tightly in mark_extractor with padding!
        # In mark_extractor: crop = gray[..., max(0, x+pad_bx):...]
        # So `box_img` is smaller than w, h. Let's just use the thresholded full_row_crop
        pass

    # Better approach: Just threshold the full_row_crop and draw WHITE rectangles over the grid lines!
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(2, 2))
    enhanced_full = clahe.apply(full_row_crop)
    
    # We upscale first for better OCR
    big = cv2.resize(enhanced_full, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    _, binary_full = cv2.threshold(big, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # We need to mask out grid lines.
    # The cell_coords are for the 4x smaller image!
    canvas_inv = np.zeros_like(binary_full)
    
    for coord in coords_to_process:
        x, y, w, h = coord
        # Scale to 4x
        x *= 4
        y *= 4
        w *= 4
        h *= 4
        
        # Shrink the box a bit to exclude the cell borders
        pad_x = int(w * 0.15)
        pad_y = int(h * 0.15)
        
        x1, y1 = x + pad_x, y + pad_y
        x2, y2 = x + w - pad_x, y + h - pad_y
        
        # Copy the ink from binary_full to canvas_inv for this character
        canvas_inv[y1:y2, x1:x2] = binary_full[y1:y2, x1:x2]
        
    final_canvas = cv2.bitwise_not(canvas_inv)
    final_canvas = cv2.copyMakeBorder(final_canvas, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255)
    
    rgb = cv2.cvtColor(final_canvas, cv2.COLOR_GRAY2RGB)
    
    allowlist = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    res = reader.readtext(rgb, allowlist=allowlist, paragraph=False)
    
    if res:
        res.sort(key=lambda item: item[0][0][0])
        text = "".join([r[1] for r in res])
        text = re.sub(f'[^{allowlist}]', '', text.upper())
        return text
    return ""

input_dir = r"c:\Users\ragha\OneDrive\Desktop\Descriptive Sheets"
image_paths = glob.glob(os.path.join(input_dir, "*.[jJ][pP][gG]")) + \
              glob.glob(os.path.join(input_dir, "*.[jJ][pP][eE][gG]")) + \
              glob.glob(os.path.join(input_dir, "*.[pP][nN][gG]"))
image_paths.sort()

# Add test image indexes that are problematic
issues = [0, 1, 6, 10, 20, 23, 40, 41]
for idx in issues:
    img_path = image_paths[idx]
    with open(img_path, 'rb') as f:
        img_bytes = f.read()
    img_color, gray, thresh = preprocess_image(img_bytes)
    ht_row_data, box_images = extract_ht_number_boxes(gray, thresh)

    if box_images:
        res = test_natural_stitched(ht_row_data, box_images)
        print(f"Index {idx} ({os.path.basename(img_path)}): {res}")
