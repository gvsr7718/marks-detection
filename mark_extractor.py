"""
Mark Extractor Module — Stage 1 (cont.)
Identifies the marks grid, extracts individual mark cells and H.T. Number.

Uses a template-aware approach:
  1. Constrains detection to the upper portion of the page
  2. Finds the marks grid by locating a dense cluster of small cells
  3. Filters out the answer-writing area by cell-size constraints
  4. Uses EasyOCR for robust digit recognition on individual cells
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

# ────────────────────────────────────────────────────────────────────────────
#  MARKS GRID EXTRACTION  (template-aware)
# ────────────────────────────────────────────────────────────────────────────

def _detect_grid_cells_in_roi(thresh_roi: np.ndarray,
                               min_cell_area: int = 400,
                               max_cell_w: int = 200,
                               max_cell_h: int = 120) -> List[Dict]:
    """
    Detect small rectangular cells in a binary ROI using morphological
    line detection.  Only cells within the size constraints are returned,
    which naturally excludes the large answer-writing area.
    """
    # Detect horizontal lines
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    h_lines = cv2.morphologyEx(thresh_roi, cv2.MORPH_OPEN, h_kernel)

    # Detect vertical lines
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
    v_lines = cv2.morphologyEx(thresh_roi, cv2.MORPH_OPEN, v_kernel)

    # Combine to form grid
    grid = cv2.add(h_lines, v_lines)
    kernel = np.ones((3, 3), np.uint8)
    grid = cv2.dilate(grid, kernel, iterations=1)

    contours, hierarchy = cv2.findContours(
        grid, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )

    cells = []
    if hierarchy is not None:
        for i, cnt in enumerate(contours):
            # Leaf nodes only (no children)
            if hierarchy[0][i][2] == -1:
                x, y, w, h = cv2.boundingRect(cnt)
                area = w * h
                # Filter: must be a reasonable cell size
                if (min_cell_area < area
                        and w <= max_cell_w and h <= max_cell_h
                        and w > 15 and h > 15):
                    cells.append({"x": x, "y": y, "w": w, "h": h})
    return cells


def extract_marks_grid_template(gray: np.ndarray,
                                 thresh: np.ndarray,
                                 img_color: np.ndarray,
                                 debug_dir: str = "debug_output"
                                 ) -> Tuple[List[np.ndarray], Optional[np.ndarray], Optional[Dict]]:
    """
    Template-aware marks grid extraction.

    Strategy:
      1.  Search only the top ~45 % of the image.
      2.  Use morphological line detection to find cells, but reject any
          cell wider than 200 px or taller than 120 px  ➜  this automatically
          excludes the answer-writing area.
      3.  Group the surviving cells into rows and pick the table whose
          bottom row is labelled "MARKS" (or, failing that, the table with
          the most red ink).
      4.  Return the individual mark-cell crops from the MARKS row.

    Returns:
        (mark_cells, total_cell, table_info_dict)
    """
    img_h, img_w = gray.shape[:2]
    search_h = int(img_h * 0.55)  # search top 55% for marks grid

    # Scale cell-size constraints relative to baseline image height of 2000px
    scale = img_h / 2000.0
    max_w = int(200 * scale)
    max_h = int(120 * scale)
    min_area = int(400 * scale)

    # --- Step 1: detect cells in the upper portion only ----------------------
    thresh_roi = thresh[0:search_h, :]
    cells = _detect_grid_cells_in_roi(thresh_roi, min_cell_area=min_area,
                                      max_cell_w=max_w, max_cell_h=max_h)

    if not cells:
        print(f"[MARKS] No grid cells found in upper region (search_h={search_h}, scale={scale:.2f})")
        return [], None, None

    print(f"[MARKS] Found {len(cells)} grid cells in upper region")

    # --- Step 2: group into rows and tables ----------------------------------
    rows = group_cells_into_rows(cells, y_threshold=25)

    # Build tables from rows (same logic as find_table_regions but local)
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

    print(f"[MARKS] {len(tables)} candidate table(s)")
    for ti, t in enumerate(tables):
        print(f"[MARKS]   Table {ti}: y={t['y']} rows={len(t['rows'])} cells={len(t['cells'])}")

    # --- Step 3: pick the grading / marks table ------------------------------
    marks_table = _pick_marks_table(gray, img_color, tables, search_h)

    if marks_table is None:
        print("[MARKS] Could not identify marks table")
        return [], None, None

    print(f"[MARKS] Selected table at y={marks_table['y']}, "
          f"rows={len(marks_table['rows'])}, cells={len(marks_table['cells'])}")

    # --- Step 4: debug visualisation ----------------------------------------
    if debug_dir:
        _debug_draw_table(gray, marks_table, debug_dir)

    # --- Step 5: extract mark cells from the BOTTOM row ---------------------
    return _extract_bottom_row_cells(gray, marks_table)


def _table_from_rows(rows):
    all_cells = [c for row in rows for c in row]
    xs = [c['x'] for c in all_cells]
    ys = [c['y'] for c in all_cells]
    x_max = [c['x'] + c['w'] for c in all_cells]
    y_max = [c['y'] + c['h'] for c in all_cells]
    return {
        "x": min(xs), "y": min(ys),
        "w": max(x_max) - min(xs),
        "h": max(y_max) - min(ys),
        "rows": rows,
        "cells": all_cells
    }


def _pick_marks_table(gray, img_color, tables, search_h):
    """Choose the table most likely to be the marks grid.
    
    Strategy (ordered by confidence):
      1. Tables with ≥3 rows whose OCR text contains 'mark'/'q.no' keywords
      2. Any table whose OCR text contains keywords (partial grid detection)
      3. Table with the most red ink (marks are often in red)
      4. Table with the most cells (last resort)
    """
    # Only consider tables with a proper structured grid layout (e.g., at least 3 rows: Header, Sublabels, Marks)
    # The HT row has 1 row. Random text blocks usually have 1 row.
    valid_tables = [t for t in tables if len(t.get('rows', [])) >= 3]
    
    if not valid_tables:
        print("[DEBUG] No tables with >= 3 rows found! Falling back to raw tables if OCR dictates.")
        valid_tables = tables

    mark_keywords = ('mark', 'q.no', 'qno', 'total', 'ques')

    # Pass 1: Strict OCR match on exact bounding box
    for t in valid_tables:
        if len(t['rows']) == 1 and _looks_like_ht_row(t):
            continue
        roi = gray[t['y']:t['y'] + t['h'], t['x']:t['x'] + t['w']]
        if roi.size == 0:
            continue
        try:
            text = pytesseract.image_to_string(roi, config='--psm 6').lower()
            if any(kw in text for kw in mark_keywords):
                t['ocr_text'] = text
                return t
        except Exception:
            continue

    # Pass 2: Any table + expanded-ROI OCR match
    # Collect ALL matches and pick the LOWEST one (marks grid is below HT row)
    candidates = []
    for t in valid_tables:
        y_start = max(0, t['y'] - 80)
        y_end = min(gray.shape[0], t['y'] + t['h'] + 20)
        x_start = max(0, t['x'] - 50)
        x_end = min(gray.shape[1], t['x'] + t['w'] + 50)
        roi = gray[y_start:y_end, x_start:x_end]
        if roi.size == 0:
            continue
        try:
            text = pytesseract.image_to_string(roi, config='--psm 6').lower()
            if any(kw in text for kw in mark_keywords):
                t['ocr_text'] = text
                candidates.append(t)
        except Exception:
            continue
    if candidates:
        # Pick the lowest table (highest y value)
        return max(candidates, key=lambda t: t['y'])

    # Pass 3: Table with the most red ink (skip 1-row HT-like tables)
    red_mask = detect_red_regions(img_color)
    best, best_score = None, 0
    for t in valid_tables:
        # Skip 1-row tables that look like HT number boxes
        # (all cells roughly same size, single row)
        if len(t['rows']) == 1 and _looks_like_ht_row(t):
            continue
        red_count = cv2.countNonZero(
            red_mask[t['y']:t['y'] + t['h'], t['x']:t['x'] + t['w']]
        )
        if red_count > best_score:
            best_score = red_count
            best = t
    if best and best_score > 100:
        return best

    # Pass 4: Table with the most cells (last resort, skip HT-like)
    non_ht = [t for t in valid_tables if not (len(t['rows']) == 1 and _looks_like_ht_row(t))]
    if non_ht:
        return max(non_ht, key=lambda t: len(t['cells']))
    if valid_tables:
        return max(valid_tables, key=lambda t: len(t['cells']))
    return None


def _looks_like_ht_row(table):
    """Check if a table looks like an HT number row (single row of uniform small cells)."""
    cells = table.get('cells', [])
    if not cells or len(cells) < 8:
        return False
    # HT row cells are all roughly the same size
    widths = [c['w'] for c in cells]
    heights = [c['h'] for c in cells]
    if not widths:
        return False
    med_w = sorted(widths)[len(widths)//2]
    med_h = sorted(heights)[len(heights)//2]
    # If most cells have similar width (within 50% of median), it's HT-like
    uniform = sum(1 for w in widths if abs(w - med_w) < med_w * 0.5)
    return uniform >= len(cells) * 0.7 and med_h < med_w * 2


def _debug_draw_table(gray, table, debug_dir):
    """Draw coloured rectangles on the cells for visual debugging."""
    debug_img = cv2.cvtColor(gray.copy(), cv2.COLOR_GRAY2BGR)
    rows = table.get('rows', [])
    for ri, row in enumerate(rows):
        for ci, c in enumerate(row):
            colour = (0, 0, 255) if ri == len(rows) - 1 else (0, 255, 0)
            cv2.rectangle(debug_img,
                          (c['x'], c['y']),
                          (c['x'] + c['w'], c['y'] + c['h']),
                          colour, 2)
            cv2.putText(debug_img, f"R{ri}C{ci}",
                        (c['x'] + 2, c['y'] + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, colour, 1)
    cv2.imwrite(f"{debug_dir}/05_marked_table_cells.jpg", debug_img)


def _extract_bottom_row_cells(gray, table):
    """Extract crops from the MARKS (bottom) row of the table using Absolute Span Interpolation."""
    rows = table.get('rows', [])
    if not rows:
        return [], None, table

    marks_row = rows[-1]
    marks_row_sorted = sorted(marks_row, key=lambda c: c['x'])

    mark_cells = []
    total_cell = None
    pad = 4  # padding to strip cell borders

    if len(marks_row_sorted) < 3:
        return [], None, table

    widths = [c['w'] for c in marks_row_sorted]
    median_w = sorted(widths)[len(widths)//2]

    # 1. Extract Label cell (always the leftmost, typically wide)
    if marks_row_sorted[0]['w'] > median_w * 1.5:
        label_cell = marks_row_sorted.pop(0)
    else:
        # If it wasn't merged, maybe it's missing? Fallback to standard 130px width
        label_cell = {
            'x': table['x'], 'y': marks_row_sorted[0]['y'],
            'w': 130, 'h': marks_row_sorted[0]['h']
        }

    # 2. Extract Total cell (always the rightmost, typically wide)
    if marks_row_sorted and marks_row_sorted[-1]['w'] > median_w * 1.5:
        total_box = marks_row_sorted.pop(-1)
    else:
        last_box = marks_row_sorted[-1] if marks_row_sorted else label_cell
        total_box = {
            'x': last_box['x'] + last_box['w'],
            'y': last_box['y'],
            'w': max(int(median_w * 3), (table['x'] + table['w']) - (last_box['x'] + last_box['w'])),
            'h': last_box['h']
        }

    # 3. Absolute Span Division
    # Calculate exactly the empty space between the Label Cell and the Total Cell
    span_start = label_cell['x'] + label_cell['w']
    span_end = total_box['x']
    span_width = span_end - span_start
    
    # Slice the void mathematically into 12 perfect fractions
    interpolated_boxes = []
    chunk_w = span_width / 12.0
    for idx in range(12):
        interpolated_boxes.append({
            'x': int(span_start + idx * chunk_w),
            'y': label_cell['y'],
            'w': int(chunk_w),
            'h': label_cell['h']
        })

    # 4. Execute exact coordinate cropping
    for c in interpolated_boxes:
        x1 = max(0, c['x'] + pad)
        y1 = max(0, c['y'] + pad)
        x2 = min(gray.shape[1], c['x'] + c['w'] - pad)
        y2 = min(gray.shape[0], c['y'] + c['h'] - pad)
        if x2 > x1 and y2 > y1:
            mark_cells.append(gray[y1:y2, x1:x2])

    # Crop Total Cell (with massive 2.0x downwards expansion to catch trailing handwriting)
    tx1 = max(0, total_box['x'] + pad)
    ty1 = max(0, total_box['y'] + pad)
    tx2 = min(gray.shape[1], total_box['x'] + int(total_box['w'] * 1.2) - pad)
    ty2 = min(gray.shape[0], total_box['y'] + int(total_box['h'] * 2.0) - pad)
    
    if tx2 > tx1 and ty2 > ty1:
        total_cell = gray[ty1:ty2, tx1:tx2]

    # Re-inject our geometrically perfect 14 boxes into the abstract table data struct
    # Exactly: [Label] + [12 Marks] + [Total] = 14 bounds!
    table['rows'][-1] = [label_cell] + interpolated_boxes + [total_box]

    return mark_cells, total_cell, table


# ────────────────────────────────────────────────────────────────────────────
#  HT NUMBER EXTRACTION  (unchanged but with better size filtering)
# ────────────────────────────────────────────────────────────────────────────

def extract_ht_number_boxes(gray: np.ndarray, thresh: np.ndarray) -> tuple:
    """
    Extract the H.T. Number row as a full-row crop plus cell coordinates.
    Returns (full_row_crop, cell_coords_relative) where cell_coords_relative
    is a list of (x, y, w, h) tuples relative to the full_row_crop.

    The H.T. number section is a single row of 11 cells ("H.T. No." label + 10 digit boxes).
    """
    height = gray.shape[0]
    width = gray.shape[1]

    # Scale search area and constraints based on image height
    scale = height / 2000.0
    search_h = int(height * 0.40)  # slightly more generous for HT row
    max_w = int(200 * scale)
    max_h = int(120 * scale)
    min_area = int(300 * scale)
    thresh_roi = thresh[0:search_h, :]

    cells = _detect_grid_cells_in_roi(
        thresh_roi, min_cell_area=min_area, max_cell_w=max_w, max_cell_h=max_h
    )

    if not cells:
        print(f"[DEBUG] No cells found for HT detection (search_h={search_h}, scale={scale:.2f})")
        return None, []

    rows = group_cells_into_rows(cells, y_threshold=int(20 * scale))

    ht_row = None
    for row in rows:
        # H.T. No. + 10 boxes ≈ 11 cells; allow variance
        if 9 <= len(row) <= 14:
            ht_row = row
            break

    if not ht_row:
        print("[DEBUG] No suitable HT Number row found!")
        return None, []

    print(f"[DEBUG] Found HT Number row: {len(ht_row)} cells")

    # Sort left-to-right
    cells_sorted = sorted(ht_row, key=lambda c: c['x'])
    
    start_idx = 0
    if len(cells_sorted) > 0 and cells_sorted[0]['w'] > 120:
        start_idx = 1
        
    box_cells = cells_sorted[start_idx:]
    # Take up to the right-most 10 boxes (in case of extra noise cells on the left)
    box_cells = box_cells[-10:] if len(box_cells) >= 10 else box_cells

    # Compute bounding box for the digit cells
    x_min = min(c['x'] for c in box_cells)
    y_min = min(c['y'] for c in box_cells)
    x_max = max(c['x'] + c['w'] for c in box_cells)
    y_max = max(c['y'] + c['h'] for c in box_cells)

    # Add padding to prevent OCR from clipping characters
    pad_x_right = 15
    pad_x_left = 45 if start_idx == 1 else 15
    pad_y = 5

    x_min = max(0, x_min - pad_x_left)
    y_min = max(0, y_min - pad_y)
    x_max = min(width, x_max + pad_x_right)
    y_max = min(height, y_max + pad_y)

    full_row_crop = gray[y_min:y_max, x_min:x_max]

    # Relative cell coordinates
    cell_coords = []
    for c in box_cells:
        rel_x = c['x'] - x_min
        rel_y = c['y'] - y_min
        cell_coords.append((rel_x, rel_y, c['w'], c['h']))

    boxes = []
    for c in box_cells:
        x, y, w, h = c['x'], c['y'], c['w'], c['h']
        pad_bx = int(w * 0.15)
        pad_by = int(h * 0.15)
        crop = gray[max(0, y + pad_by):min(height, y + h - pad_by),
                    max(0, x + pad_bx):min(width, x + w - pad_bx)]
        boxes.append(crop)

    return (full_row_crop, cell_coords), boxes


# ────────────────────────────────────────────────────────────────────────────
#  LEGACY HELPERS  (kept for API compatibility but no longer primary path)
# ────────────────────────────────────────────────────────────────────────────

def identify_grading_tables(gray, tables):
    """Legacy: Identify grading tables via OCR keywords."""
    grading = []
    for table in tables:
        x, y, w, h = table['x'], table['y'], table['w'], table['h']
        roi = gray[y:y+h, x:x+w]
        if roi.size == 0:
            continue
        try:
            text = pytesseract.image_to_string(roi, config='--psm 6').lower()
            if any(kw in text for kw in ('ques', 'num', 'q.no', 'marks', 'mark', 'total')):
                table['ocr_text'] = text
                grading.append(table)
        except Exception:
            continue
    return grading


def find_marked_table(img_color, grading_tables):
    """Legacy: Find table with the most red ink."""
    if not grading_tables:
        return None
    if len(grading_tables) == 1:
        return grading_tables[0]
    red_mask = detect_red_regions(img_color)
    best, best_score = grading_tables[0], 0
    for t in grading_tables:
        red = cv2.countNonZero(
            red_mask[t['y']:t['y']+t['h'], t['x']:t['x']+t['w']])
        if red > best_score:
            best_score = red
            best = t
    return best


def extract_mark_cells(gray, marked_table):
    """Legacy wrapper: delegates to _extract_bottom_row_cells."""
    cells, total, _ = _extract_bottom_row_cells(gray, marked_table)
    return cells, total


def extract_mcq_score_region(gray: np.ndarray) -> np.ndarray:
    """Extract the score region from an MCQ answer sheet."""
    height, width = gray.shape
    return gray[int(height * 0.02):int(height * 0.15),
                int(width * 0.70):int(width * 0.95)]


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

    contours, _ = cv2.findContours(
        dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    valid_contours = []
    cell_h, cell_w = gray.shape[:2]

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        is_border = (h > cell_h * 0.7 and (w / float(h) if h else 0) < 0.3
                     and (x < 3 or x + w > cell_w - 3))
        if w >= 1 and h >= 8 and not is_border:
            valid_contours.append((x, y, w, h))

    valid_contours.sort(key=lambda b: b[0])

    _, binary = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    digit_images = []
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
        square[off_y:off_y + h_c, off_x:off_x + w_c] = digit_crop

        resized = cv2.resize(square, (20, 20), interpolation=cv2.INTER_AREA)
        final_28 = np.zeros((28, 28), dtype=np.uint8)
        final_28[4:24, 4:24] = resized
        digit_images.append(final_28)

    return digit_images
