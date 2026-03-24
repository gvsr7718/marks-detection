"""
Mark Extractor Module — Stage 1 (cont.)
Identifies the marks grid, extracts individual mark cells and H.T. Number.

Based on: "AI-Powered Mark Recognition in Assessment and Attainment Calculation"
         by J. Annrose et al. (ICTACT, Jan 2025)
"""

import cv2
import numpy as np
import pytesseract
from typing import List, Dict, Tuple, Optional

# Set Tesseract path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

from table_detector import (
    detect_tables_morphological, 
    group_cells_into_rows, 
    detect_red_regions,
    find_table_regions
)


def identify_grading_tables(gray: np.ndarray, 
                            tables: List[Dict]) -> List[Dict]:
    """
    Use Tesseract OCR to identify which tables are grading tables.
    Tables containing keywords 'ques', 'num', 'q.no', 'marks' are grading tables.
    """
    grading_tables = []
    
    for table in tables:
        x, y, w, h = table['x'], table['y'], table['w'], table['h']
        table_roi = gray[y:y+h, x:x+w]
        
        if table_roi.size == 0:
            continue
        
        try:
            text = pytesseract.image_to_string(table_roi, config='--psm 6').lower()
            if 'ques' in text or 'num' in text or 'q.no' in text or 'marks' in text or 'mark' in text or 'total' in text:
                table['ocr_text'] = text
                grading_tables.append(table)
        except Exception as e:
            print(f"Tesseract error on table: {e}")
            continue
    
    return grading_tables


def find_marked_table(img_color: np.ndarray, 
                      grading_tables: List[Dict]) -> Optional[Dict]:
    """
    Among grading tables, find the one with the most red handwritten marks.
    """
    if not grading_tables:
        return None
    
    if len(grading_tables) == 1:
        return grading_tables[0]
    
    red_mask = detect_red_regions(img_color)
    
    max_red_count = 0
    marked_table = grading_tables[0]
    
    for table in grading_tables:
        x, y, w, h = table['x'], table['y'], table['w'], table['h']
        table_red = red_mask[y:y+h, x:x+w]
        red_count = cv2.countNonZero(table_red)
        
        if red_count > max_red_count:
            max_red_count = red_count
            marked_table = table
    
    return marked_table


def extract_mark_cells(gray: np.ndarray, 
                       marked_table: Dict) -> Tuple[List[np.ndarray], Optional[np.ndarray]]:
    """
    Extract individual mark cells from the bottom row of the marked table.
    """
    rows = marked_table.get('rows', [])
    if not rows:
        return [], None
    
    # The MARKS row is the LAST (bottom-most) row
    marks_row = rows[-1]
    marks_row_sorted = sorted(marks_row, key=lambda c: c['x'])
    
    mark_cells = []
    total_cell = None
    pad = 6  # Restored padding to exclude cell borders completely
    
    for i, c in enumerate(marks_row_sorted):
        x1 = c['x'] + pad
        y1 = c['y'] + pad
        x2 = c['x'] + c['w'] - pad
        y2 = c['y'] + c['h'] - pad
        
        x1, y1 = max(0, x1), max(0, y1)
        x2 = min(gray.shape[1], x2)
        y2 = min(gray.shape[0], y2)
        
        if x2 <= x1 or y2 <= y1:
            continue
            
        crop = gray[y1:y2, x1:x2]
        if crop.size == 0:
            continue
            
        if i == 0:
            if c['w'] > c['h'] * 1.5:
                continue  # Skip label cell (MARKS)
        
        if i == len(marks_row_sorted) - 1:
            total_cell = crop
        else:
            mark_cells.append(crop)
    
    return mark_cells, total_cell


def extract_ht_number_boxes(gray: np.ndarray, thresh: np.ndarray) -> tuple:
    """
    Extract the H.T. Number row as a full-row crop plus cell coordinates.
    Returns (full_row_crop, cell_coords_relative) where cell_coords_relative
    is a list of (x, y, w, h) tuples relative to the full_row_crop.
    
    The H.T. number section is a single row of 11 cells ("H.T. No." label + 10 digit boxes).
    """
    height = gray.shape[0]
    width = gray.shape[1]
    cells = detect_tables_morphological(thresh)
    rows = group_cells_into_rows(cells)
    
    ht_row = None
    for row in rows:
        # Ignore rows in the bottom half of the image
        y_avg = sum(c['y'] for c in row) / len(row) if row else height
        if y_avg > height * 0.5:
            continue
            
        # H.T. No. + 10 boxes is ~11 cells. Allow some variance.
        if 10 <= len(row) <= 13:
            ht_row = row
            break
            
    if not ht_row:
        print("[DEBUG] No suitable HT Number row found!")
        return None, []
        
    print(f"[DEBUG] Found HT Number row: {len(ht_row)} cells")
    
    # Sort the cells from left to right
    cells_sorted = sorted(ht_row, key=lambda c: c['x'])
    
    # We want the remaining 10 boxes (skipping the label).
    box_cells = cells_sorted[-10:] if len(cells_sorted) >= 10 else cells_sorted
    
    # Compute the full bounding box for the 10 digit cells
    x_min = min(c['x'] for c in box_cells)
    y_min = min(c['y'] for c in box_cells)
    x_max = max(c['x'] + c['w'] for c in box_cells)
    y_max = max(c['y'] + c['h'] for c in box_cells)
    
    # Add small margins
    x_min = max(0, x_min - 2)
    y_min = max(0, y_min - 2)
    x_max = min(width, x_max + 2)
    y_max = min(height, y_max + 2)
    
    full_row_crop = gray[y_min:y_max, x_min:x_max]
    
    # Compute relative cell coordinates
    cell_coords = []
    for c in box_cells:
        rel_x = c['x'] - x_min
        rel_y = c['y'] - y_min
        cell_coords.append((rel_x, rel_y, c['w'], c['h']))
    
    # Also save individual boxes for debug
    boxes = []
    for c in box_cells:
        x, y, w, h = c['x'], c['y'], c['w'], c['h']
        pad_x = int(w * 0.15)
        pad_y = int(h * 0.15)
        crop = gray[max(0, y+pad_y):min(height, y+h-pad_y), max(0, x+pad_x):min(width, x+w-pad_x)]
        boxes.append(crop)
        
    return (full_row_crop, cell_coords), boxes


def extract_mcq_score_region(gray: np.ndarray) -> np.ndarray:
    """
    Extract the score region from an MCQ answer sheet.
    """
    height, width = gray.shape
    score_roi = gray[int(height * 0.02):int(height * 0.15),
                     int(width * 0.70):int(width * 0.95)]
    return score_roi


def extract_digit_contours(cell_img: np.ndarray) -> List[np.ndarray]:
    """
    Extract potential digit contours from a mark cell image.
    """
    if cell_img is None or cell_img.size == 0:
        return []
    
    if len(cell_img.shape) == 3:
        gray = cv2.cvtColor(cell_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = cell_img.copy()
        
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=1)
    
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    digit_images = []
    valid_contours = []
    cell_h, cell_w = gray.shape[:2]
    
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        aspect = w / float(h) if h > 0 else 0
        area = w * h
        bounding_area = cell_w * cell_h
        
        is_border = (h > cell_h * 0.7 and aspect < 0.3) and (x < 3 or x + w > cell_w - 3)
        
        if w >= 1 and h >= 8 and not is_border:
            valid_contours.append((x, y, w, h))
    
    valid_contours.sort(key=lambda b: b[0])
    
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    for x, y, w, h in valid_contours:
        margin = 4
        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(binary.shape[1], x + w + margin)
        y2 = min(binary.shape[0], y + h + margin)
        
        digit_crop = binary[y1:y2, x1:x2]
        if digit_crop.size == 0:
            continue
            
        h_c, w_c = digit_crop.shape
        max_dim = max(h_c, w_c)
        square = np.zeros((max_dim, max_dim), dtype=np.uint8)
        
        off_x = (max_dim - w_c) // 2
        off_y = (max_dim - h_c) // 2
        square[off_y:off_y+h_c, off_x:off_x+w_c] = digit_crop
        
        resized = cv2.resize(square, (20, 20), interpolation=cv2.INTER_AREA)
        final_28 = np.zeros((28, 28), dtype=np.uint8)
        final_28[4:24, 4:24] = resized
        digit_images.append(final_28)
    
    return digit_images
