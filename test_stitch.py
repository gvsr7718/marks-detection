import cv2
import numpy as np
import glob
import os
import easyocr
import re

from table_detector import preprocess_image
from mark_extractor import extract_ht_number_boxes

reader = easyocr.Reader(['en'], gpu=True, verbose=False)

def test_stitched_row(box_images):
    # Process each box and collect them
    processed_boxes = []
    for box in box_images[-10:]:
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(2, 2))
        enhanced = clahe.apply(box)
        big = cv2.resize(enhanced, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
        _, binary = cv2.threshold(big, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Clean edges
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            c = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(c)
            if w > 5 and h > 10:
                char_crop = binary[y:y+h, x:x+w]
                char_inv = cv2.bitwise_not(char_crop)
                
                # Standardize height, keep aspect ratio
                target_h = 100
                scale = target_h / h
                target_w = int(w * scale)
                resized = cv2.resize(char_inv, (target_w, target_h))
                
                # Add horizontal padding
                padded = cv2.copyMakeBorder(resized, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=255)
                processed_boxes.append(padded)
                continue
        
        # fallback
        final_box_ez = cv2.bitwise_not(binary)
        target_h = 100
        scale = target_h / final_box_ez.shape[0]
        target_w = int(final_box_ez.shape[1] * scale)
        resized = cv2.resize(final_box_ez, (target_w, target_h))
        padded = cv2.copyMakeBorder(resized, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=255)
        processed_boxes.append(padded)

    if not processed_boxes:
        return ""
        
    # Stitch horizontally
    stitched = np.hstack(processed_boxes)
    # Add a border around the whole thing
    stitched = cv2.copyMakeBorder(stitched, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255)
    
    rgb = cv2.cvtColor(stitched, cv2.COLOR_GRAY2RGB)
    
    # Run EasyOCR
    allowlist = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    res = reader.readtext(rgb, allowlist=allowlist, paragraph=False)
    
    if res:
        # Combine all text found
        text = "".join([r[1] for r in res])
        text = re.sub(r'[^A-Z0-9]', '', text.upper())
        return text
    return ""

input_dir = r"c:\Users\ragha\OneDrive\Desktop\Descriptive Sheets"

image_paths = glob.glob(os.path.join(input_dir, "*.[jJ][pP][gG]")) + \
              glob.glob(os.path.join(input_dir, "*.[jJ][pP][eE][gG]")) + \
              glob.glob(os.path.join(input_dir, "*.[pP][nN][gG]"))

image_paths.sort()

for img_path in image_paths[:10]:
    with open(img_path, 'rb') as f:
        img_bytes = f.read()
    img_color, gray, thresh = preprocess_image(img_bytes)
    ht_row_data, box_images = extract_ht_number_boxes(gray, thresh)

    if box_images:
        res = test_stitched_row(box_images)
        print(f"{os.path.basename(img_path)}: {res}")
