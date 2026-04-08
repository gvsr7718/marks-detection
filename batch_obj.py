import os
import glob
from objective_detector import extract_objective_marks

# Suppress some standard output clutter
import sys

img_dir = r"c:\Users\ragha\OneDrive\Desktop\VS CODE\Projects\marks_detection\Obj_sheets_JPG"
images = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))

print(f"Testing a sample of 10 images out of {len(images)}...")

results = {}
for img_path in images:
    filename = os.path.basename(img_path)
    print(f"\n--- Processing {filename} ---")
    
    # Run the extraction (with debug=False to avoid saving 10*4 debug images)
    try:
        score = extract_objective_marks(img_path, debug=False)
        results[filename] = score
    except Exception as e:
        print(f"Error on {filename}: {e}")
        results[filename] = "ERROR"
        
print("\n=== SUMMARY OF SAMPLE BATCH ===")
for fname, score in results.items():
    print(f"{fname}: {score}")
