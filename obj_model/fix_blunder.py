import os
import glob
import shutil
import re

print("Starting to fix the labeling blunder...")

# 1. REMOVE BAD WHATSAPP IMAGES
bad_files = glob.glob("obj_dataset/*/*.png")
removed = 0
for f in bad_files:
    if "WhatsApp" in os.path.basename(f):
        os.remove(f)
        removed += 1
print(f"Removed {removed} incorrectly labeled WhatsApp crop images.")

# 2. READ PREVIOUS PREDICTIONS (Baseline)
baseline = {}
try:
    with open("obj_test_results.txt", "r") as f:
        for line in f:
            if "->" in line and "FAILED" not in line:
                parts = line.split("->")
                fname = parts[0].strip()
                score_part = parts[1].strip().split("/")[0]
                baseline[fname] = int(score_part)
except FileNotFoundError:
    print("Could not find obj_test_results.txt! Aborting fix.")
    exit(1)

# 3. APPLY USER CORRECTIONS
corrections_text = """
3201 6
3203 2
3204 8
3205 8
3206 5
3207 7
3209 8
3211 6
3212 5
3213 2
3214 6
3217 4
3218 8
3221 3
3222 6
3223 8
3225 5
3226 7
3227 2
3229 2
3230 2
3232 8
3233 8
3235 4
3241 6
3242 7
3243 5
3244 9
3248 8
3249 6
3250 7
3251 7
3252 8
3253 7
3254 6
3255 7
3256 7
3257 7
3258 7
3259 6
3260 5
3261 3
3262 7
le1 7
le2 5
le3 8
le4 8
le5 4
le6 4
"""

for line in corrections_text.strip().split("\n"):
    parts = line.strip().split()
    if len(parts) >= 2:
        name_prefix = parts[0]
        correct_lbl = int(parts[1])
        # Find matching filename in baseline
        match_fname = f"{name_prefix}.jpeg"
        if match_fname in baseline:
            baseline[match_fname] = correct_lbl
        else:
            # Just in case they are different
            for k in baseline.keys():
                if k.startswith(name_prefix):
                    baseline[k] = correct_lbl

print(f"Constructed Ground Truth for {len(baseline)} batch 2 images.")

# 4. RUN EXTRACTION ON Obj_sheets_batch2_JPG
import sys
# importing the extraction logic directly
from prepare_data import extract_score_region

batch_dir = "Obj_sheets_batch2_JPG"
images = glob.glob(os.path.join(batch_dir, "*.jpeg")) + glob.glob(os.path.join(batch_dir, "*.jpg"))
extracted_dir = "obj_dataset/unlabeled"

if os.path.exists(extracted_dir):
    shutil.rmtree(extracted_dir)
os.makedirs(extracted_dir, exist_ok=True)

success_extract = 0
for img_path in images:
    out_path, crop = extract_score_region(img_path, output_dir=extracted_dir)
    if out_path:
        success_extract += 1

print(f"Extracted {success_extract} crops correctly named from {batch_dir}.")

# 5. MOVE TO CORRECT LABEL DIRECTORIES
new_crops = glob.glob(os.path.join(extracted_dir, "*.png"))
moved = 0
for crop in new_crops:
    # crop name is e.g. 3201_score.png, we need 3201.jpeg to lookup
    basename = os.path.basename(crop).replace("_score.png", ".jpeg")
    if basename in baseline:
        lbl = baseline[basename]
        os.makedirs(f"obj_dataset/{lbl}", exist_ok=True)
        dest = f"obj_dataset/{lbl}/{os.path.basename(crop)}"
        shutil.move(crop, dest)
        moved += 1
    else:
        print(f"Warning: No ground truth label for {basename}")

print(f"Moved {moved} correctly labeled crops to obj_dataset!")
