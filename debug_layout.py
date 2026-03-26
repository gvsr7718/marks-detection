import glob
import os
import cv2
from table_detector import preprocess_image, find_table_regions
from mark_extractor import _detect_grid_cells_in_roi, group_cells_into_rows, _table_from_rows, _pick_marks_table

def debug_sheet_layout():
    input_dir = r"c:\Users\ragha\OneDrive\Desktop\Descriptive Sheets"
    paths = glob.glob(os.path.join(input_dir, "*.[jJ][pP][gG]")) + \
            glob.glob(os.path.join(input_dir, "*.[jJ][pP][eE][gG]"))

    path = [p for p in paths if "60815" in os.path.basename(p) or "10.12.47" in p][-1]
    print(f"Analyzing {os.path.basename(path)}")
    
    with open(path, 'rb') as f:
        img_bytes = f.read()
    img_color, gray, thresh = preprocess_image(img_bytes)

    # Detect table
    height, width = gray.shape
    search_h = int(height * 0.40)
    thresh_roi = thresh[0:search_h, :]
    cells = _detect_grid_cells_in_roi(thresh_roi, min_cell_area=200, max_cell_w=300)
    
    rows = group_cells_into_rows(cells, y_threshold=25)
    tables = []
    current_table_rows = [rows[0]]
    for i in range(1, len(rows)):
        prev_bottom = max(c['y'] + c['h'] for c in current_table_rows[-1])
        curr_top = min(c['y'] for c in rows[i])
        if curr_top - prev_bottom > 80:
            tables.append(_table_from_rows(current_table_rows))
            current_table_rows = [rows[i]]
        else:
            current_table_rows.append(rows[i])
    tables.append(_table_from_rows(current_table_rows))
    
    marks_table = _pick_marks_table(gray, img_color, tables, search_h)
    
    for ri, row in enumerate(marks_table['rows']):
        row_sorted = sorted(row, key=lambda c: c['x'])
        print(f"Row {ri}: {len(row_sorted)} cells")
        for ci, c in enumerate(row_sorted):
            print(f"  Col {ci}: x={c['x']}, w={c['w']}")

if __name__ == "__main__":
    debug_sheet_layout()
