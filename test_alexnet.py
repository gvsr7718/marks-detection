import os
import glob
from mark_extractor import extract_ht_number_boxes
from table_detector import preprocess_image
from digit_recognizer import get_digit_recognizer
import cv2

recognizer = get_digit_recognizer()

input_dir = r"c:\Users\ragha\OneDrive\Desktop\Descriptive Sheets"
image_paths = glob.glob(os.path.join(input_dir, "*.[jJ][pP][gG]")) + \
              glob.glob(os.path.join(input_dir, "*.[jJ][pP][eE][gG]")) + \
              glob.glob(os.path.join(input_dir, "*.[pP][nN][gG]"))

image_paths.sort()

# Test the first 10 sheets
for img_path in image_paths[:10]:
    with open(img_path, 'rb') as f:
        img_bytes = f.read()
    img_color, gray, thresh = preprocess_image(img_bytes)
    ht_row_data, box_images = extract_ht_number_boxes(gray, thresh)

    if box_images:
        boxes_to_process = box_images[-10:] if len(box_images) >= 10 else box_images
        alexnet_digits = ""
        
        for i, box in enumerate(boxes_to_process):
            if i == 5:
                alexnet_digits += "A" # Skip letter
                continue
                
            # Process for AlexNet (requires 28x28 digit image like extract_digit_contours)
            # We can use recognizer's extract_digit_contours to find the digit contour
            from mark_extractor import extract_digit_contours
            digit_images = extract_digit_contours(box)
            if digit_images:
                val, conf = recognizer.recognize_marks_from_cell(digit_images)
                alexnet_digits += str(val)
            else:
                alexnet_digits += "?"
                
        print(f"{os.path.basename(img_path)}: {alexnet_digits}")
