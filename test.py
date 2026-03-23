import cv2
from table_detector import preprocess_image, detect_tables_morphological, find_table_regions
from mark_extractor import extract_ht_number_boxes

gray = cv2.imread('debug_output/01_gray.jpg', 0)
thresh = cv2.imread('debug_output/02_thresh.jpg', 0)

img_bin = detect_tables_morphological(thresh)
tables = find_table_regions(thresh)

print(f"Total tables: {len(tables)}")
for i, t in enumerate(tables):
    y = t['y']
    cells = len(t.get('cells', []))
    rows = len(t.get('rows', []))
    print(f"Table {i}: y={y}, cells={cells}, rows={rows}")
    
extract_ht_number_boxes(gray, thresh)
