import os
import glob
import traceback
from table_detector import preprocess_image
from mark_extractor import extract_ht_number_boxes
from digit_recognizer import get_digit_recognizer

input_dir = r"c:\Users\ragha\OneDrive\Desktop\Descriptive Sheets"

image_paths = glob.glob(os.path.join(input_dir, "*.[jJ][pP][gG]")) + \
              glob.glob(os.path.join(input_dir, "*.[jJ][pP][eE][gG]")) + \
              glob.glob(os.path.join(input_dir, "*.[pP][nN][gG]"))

image_paths.sort()

recognizer = get_digit_recognizer()

out_file = "htno_report.md"

with open(out_file, "w", encoding="utf-8") as f:
    f.write("| S.No | Image Name | HT No | Confidence |\n")
    f.write("|---|---|---|---|\n")

    for idx, path in enumerate(image_paths):
        basename = os.path.basename(path)
        print(f"[{idx+1}/{len(image_paths)}] Processing {basename}...")
        try:
            with open(path, 'rb') as img_f:
                img_bytes = img_f.read()
            img_color, gray, thresh = preprocess_image(img_bytes)
            
            ht_row_data, ht_boxes = extract_ht_number_boxes(gray, thresh)
            
            if ht_boxes:
                ht_no, ht_conf = recognizer.recognize_ht_number(ht_row_data, ht_boxes)
            else:
                ht_no, ht_conf = "Not Found", 0.0
                
            row_data = [str(idx + 1), basename, str(ht_no), f"{ht_conf:.2f}"]
            f.write("| " + " | ".join(row_data) + " |\n")
            f.flush()
        except Exception as e:
            print(f"Error processing {basename}: {e}")
            error_row = [str(idx + 1), basename, "ERROR", "0.00"]
            f.write("| " + " | ".join(error_row) + " |\n")

print(f"Report generation complete! Results saved to {out_file}")
