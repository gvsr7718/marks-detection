import cv2
import glob
import os
from main import _process_descriptive
from digit_recognizer import get_digit_recognizer
from table_detector import preprocess_image

input_dir = r"c:\Users\ragha\OneDrive\Desktop\Descriptive Sheets"

image_paths = glob.glob(os.path.join(input_dir, "*.[jJ][pP][eE][gG]")) + glob.glob(os.path.join(input_dir, "*.[jJ][pP][gG]"))

if image_paths:
    path = image_paths[0]
    print(f"Testing on {os.path.basename(path)}")
    with open(path, 'rb') as f:
        img_bytes = f.read()
        
    img_color, gray, thresh = preprocess_image(img_bytes)
    recognizer = get_digit_recognizer()
    
    result = _process_descriptive(img_color, gray, thresh, recognizer)
    
    marks = result["marks"]
    total = result["descriptive_total"]
    
    print("\n--- RESULTS ---")
    for q, m in marks.items():
        print(f"{q}: {m['value']}")
        assert m['value'] <= 5, f"Constraint failed! {q} has {m['value']} marks (>5)"
        assert m['value'] >= 0, f"Constraint failed! {q} has negative marks"
        
    print(f"Total: {total}")
    assert total <= 20, f"Constraint failed! Total is {total} (>20)"
    
    print("HTNO: ", result["ht_no"])
    print("ALL CONSTRAINTS PASSED SUCCESSFULLY.")
else:
    print("No images found.")
