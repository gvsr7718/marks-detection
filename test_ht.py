"""Quick test script for HT Number extraction."""
import traceback

try:
    from table_detector import preprocess_image, find_table_regions
    from mark_extractor import extract_ht_number_boxes
    from digit_recognizer import get_digit_recognizer

    img = open('descriptive_sheet.jpeg', 'rb').read()
    color, gray, thresh = preprocess_image(img)
    
    rec = get_digit_recognizer()
    rec._initialize()
    
    ht_row_data, ht_boxes = extract_ht_number_boxes(gray, thresh)
    
    print(f"Boxes found: {len(ht_boxes)}")
    
    ht_no, ht_conf = rec.recognize_ht_number(ht_row_data, ht_boxes)
    print(f"EXTRACTED HT NUMBER: {ht_no}")
    print(f"CONFIDENCE: {ht_conf:.2f}")
    print(f"EXPECTED:   23241A6A02")
    
except Exception as e:
    traceback.print_exc()
