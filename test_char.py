import cv2
import glob
import os
import easyocr
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

from table_detector import preprocess_image
from mark_extractor import extract_ht_number_boxes

reader = easyocr.Reader(['en'], gpu=True, verbose=False)

def test_cell(box, allowlist):
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(2, 2))
    enhanced = clahe.apply(box)
    big = cv2.resize(enhanced, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    _, binary = cv2.threshold(big, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # EasyOCR
    final_box_ez = cv2.bitwise_not(binary)
    final_box_ez = cv2.copyMakeBorder(final_box_ez, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255)
    rgb = cv2.cvtColor(final_box_ez, cv2.COLOR_GRAY2RGB)
    res_ez = reader.readtext(rgb, allowlist=allowlist, paragraph=False)
    ez_val = res_ez[0][1] if res_ez else ""
    
    # Tesseract
    # Tesseract works best on black text on white background
    config = f'-c tessedit_char_whitelist={allowlist} --psm 10'
    res_tess = pytesseract.image_to_string(final_box_ez, config=config).strip()
    
    return ez_val, res_tess

# Test first 3 images
input_dir = r"c:\Users\ragha\OneDrive\Desktop\Descriptive Sheets"

image_paths = glob.glob(os.path.join(input_dir, "*.[jJ][pP][gG]")) + \
              glob.glob(os.path.join(input_dir, "*.[jJ][pP][eE][gG]")) + \
              glob.glob(os.path.join(input_dir, "*.[pP][nN][gG]"))

image_paths.sort()

for img_path in image_paths[:3]:
    with open(img_path, 'rb') as f:
        img_bytes = f.read()
    img_color, gray, thresh = preprocess_image(img_bytes)
    ht_row_data, box_images = extract_ht_number_boxes(gray, thresh)

    if box_images:
        print(f"Testing {os.path.basename(img_path)}")
        for i, box in enumerate(box_images[-10:] if len(box_images) >= 10 else box_images):
            allowlist = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ' if i == 5 else '0123456789'
            ez, tess = test_cell(box, allowlist)
            print(f"Cell {i} ({allowlist[:10]}...): EasyOCR='{ez}' | Tesseract='{tess}'")
        print("-" * 40)
