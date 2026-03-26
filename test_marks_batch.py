import cv2
import glob
import os
import json
from main import _process_descriptive
from digit_recognizer import get_digit_recognizer
from table_detector import preprocess_image

input_dir = r"c:\Users\ragha\OneDrive\Desktop\Descriptive Sheets"
image_paths = glob.glob(os.path.join(input_dir, "*.[jJ][pP][gG]")) + \
              glob.glob(os.path.join(input_dir, "*.[jJ][pP][eE][gG]")) + \
              glob.glob(os.path.join(input_dir, "*.[pP][nN][gG]"))

# Remove the test images that do not exist or duplicate names handling if needed
image_paths.sort()

output_file = "marks_report.md"

def generate_report():
    recognizer = get_digit_recognizer()
    print(f"Starting batch marks extraction for {len(image_paths)} images...")
    
    with open(output_file, 'w') as f:
        f.write("# Descriptive Marks Extraction Report\n\n")
        f.write("| S.No | Image Name | HT No | Q1a | Q1b | Q2a | Q2b | Q3a | Q3b | Q4a | Q4b | Q5a | Q5b | Q6a | Q6b | Total |\n")
        f.write("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n")

        for idx, img_path in enumerate(image_paths, 1):
            name = os.path.basename(img_path)
            print(f"[{idx}/{len(image_paths)}] Processing {name}...")
            
            try:
                with open(img_path, 'rb') as img_f:
                    img_bytes = img_f.read()
                
                img_color, gray, thresh = preprocess_image(img_bytes)
                result = _process_descriptive(img_color, gray, thresh, recognizer)
                
                ht = result["ht_no"]["value"]
                if not ht: ht = "Not Found"
                
                m = result["marks"]
                # Helper to safely get value or "-"
                def v(q): return str(m.get(q, {}).get("value", "-"))
                
                t = result["descriptive_total"]
                
                row = f"| {idx} | {name} | {ht} | {v('q1a')} | {v('q1b')} | {v('q2a')} | {v('q2b')} | {v('q3a')} | {v('q3b')} | {v('q4a')} | {v('q4b')} | {v('q5a')} | {v('q5b')} | {v('q6a')} | {v('q6b')} | **{t}** |\n"
                f.write(row)
                f.flush()
            except Exception as e:
                print(f"Error on {name}: {e}")
                err_row = f"| {idx} | {name} | Error | | | | | | | | | | | | | |\n"
                f.write(err_row)
                f.flush()

    print(f"\nReport generation complete! Saved to {output_file}")

if __name__ == "__main__":
    generate_report()
