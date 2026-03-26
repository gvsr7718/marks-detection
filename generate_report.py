import os
import glob
import traceback
from table_detector import preprocess_image
from main import _process_descriptive
from digit_recognizer import get_digit_recognizer

input_dir = r"c:\Users\ragha\OneDrive\Desktop\Descriptive Sheets"

image_paths = glob.glob(os.path.join(input_dir, "*.[jJ][pP][gG]")) + \
              glob.glob(os.path.join(input_dir, "*.[jJ][pP][eE][gG]")) + \
              glob.glob(os.path.join(input_dir, "*.[pP][nN][gG]"))

image_paths.sort()

recognizer = get_digit_recognizer()

out_file = "report.md"

question_labels = ['q1a', 'q1b', 'q2a', 'q2b', 'q3a', 'q3b', 'q4a', 'q4b', 'q5a', 'q5b', 'q6a', 'q6b']

header1 = "| S.No | Image Name | HT No | " + " | ".join(question_labels) + " | Total |\n"
header2 = "|---|---|---| " + " | ".join(["---"] * len(question_labels)) + " | --- |\n"

with open(out_file, "w", encoding="utf-8") as f:
    f.write(header1)
    f.write(header2)

    for idx, path in enumerate(image_paths):
        basename = os.path.basename(path)
        print(f"[{idx+1}/{len(image_paths)}] Processing {basename}...")
        try:
            with open(path, 'rb') as img_f:
                img_bytes = img_f.read()
            img_color, gray, thresh = preprocess_image(img_bytes)
            
            result = _process_descriptive(img_color, gray, thresh, recognizer)
            
            ht_no = str(result.get("ht_no", {}).get("value", "N/A"))
            marks = result.get("marks", {})
            total = str(result.get("descriptive_total", "N/A"))
            
            row_data = [str(idx + 1), basename, ht_no]
            for label in question_labels:
                m = marks.get(label, {})
                v = m.get("value", "")
                if v == 0:
                     v = "0"
                row_data.append(str(v))
            row_data.append(total)
            
            f.write("| " + " | ".join(row_data) + " |\n")
            f.flush()
        except Exception as e:
            print(f"Error processing {basename}: {e}")
            error_row = [str(idx + 1), basename, "ERROR"] + ["-"] * len(question_labels) + ["-"]
            f.write("| " + " | ".join(error_row) + " |\n")
            traceback.print_exc()

print("Report generation complete! Results saved to report.md")
