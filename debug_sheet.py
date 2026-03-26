import glob
import os
import cv2
from main import _process_descriptive
from digit_recognizer import get_digit_recognizer
from table_detector import preprocess_image

def debug_specific_sheet():
    input_dir = r"c:\Users\ragha\OneDrive\Desktop\Descriptive Sheets"
    image_paths = glob.glob(os.path.join(input_dir, "*.[jJ][pP][gG]")) + \
                  glob.glob(os.path.join(input_dir, "*.[jJ][pP][eE][gG]")) + \
                  glob.glob(os.path.join(input_dir, "*.[pP][nN][gG]"))

    recognizer = get_digit_recognizer()
    
    # We will look for an image where Q1a=3 and Q1b=1 originally, or just process them until we find one that matches the marks pattern.
    # Pattern: Q1a=3, Q1b=1, Q2a=4, Q3a=2, Q3b=2, Q4a=5
    print("Searching for the requested sheet...")
    for path in image_paths:
        try:
            with open(path, 'rb') as f:
                img_bytes = f.read()
            img_color, gray, thresh = preprocess_image(img_bytes)
            
            # Run without the rigorous full pipeline first just to check the HTNO or similar. 
            # We know the total is 17 or 20. Let's just run it. 
            res = _process_descriptive(img_color, gray, thresh, recognizer)
            marks = res["marks"]
            
            # Let's check if Q2a is 4, Q3a is 2, Q4a is 5
            # Since the current buggy code might output q1a=2, q1b=0, q2a=4, q3a=2, q3b=2, q4a=5
            if marks.get("q2a", {}).get("value") == 4 and marks.get("q4a", {}).get("value") == 5:
                # This might be it!
                print(f"--- CANDIDATE FOUND: {os.path.basename(path)} ---")
                print(f"Extracted Marks: {marks}")
                print(f"Extracted HTNO: {res['ht_no']}")
                
                # Save out the cell images directly from the preprocessing loop inside _process_descriptive to 'debug_output'
                break
        except Exception as e:
            continue

if __name__ == "__main__":
    debug_specific_sheet()
