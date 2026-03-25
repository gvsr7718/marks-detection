"""
Table Detection Module — Stage 1
Handles image preprocessing and table detection using img2table library,
Hough Transform, and morphological operations.

Based on: "AI-Powered Mark Recognition in Assessment and Attainment Calculation"
         by J. Annrose et al. (ICTACT, Jan 2025)
"""

import cv2
import numpy as np
from typing import Tuple, List, Dict, Optional


def _order_points(pts: np.ndarray) -> np.ndarray:
    """Order coordinates: top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype="float32")
    pts_arr = np.array(pts)
    s = pts_arr.sum(axis=1)
    rect[0] = pts_arr[np.argmin(s)]
    rect[2] = pts_arr[np.argmax(s)]
    diff = np.diff(pts_arr, axis=1)
    rect[1] = pts_arr[np.argmin(diff)]
    rect[3] = pts_arr[np.argmax(diff)]
    return rect

def _four_point_transform(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Crop and warp bounding box to rectilinear top-down view."""
    rect = _order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

def preprocess_image(image_bytes: bytes) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Decode, detect paper border and deskew (crop to uniform rectangle), resize, and binarize.
    
    Returns:
        img_color: color image (for red pixel detection)
        gray: grayscale image (for OCR)
        thresh: binary image (for table/line detection)
    """
    # 1. Decode from bytes
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    
    if img is None:
        raise ValueError("Invalid image or cannot decode")

    # 2. Detect Paper Border / Document Contour
    # Convert to grayscale and blur to remove noise
    doc_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    doc_blur = cv2.GaussianBlur(doc_gray, (5, 5), 0)
    doc_edged = cv2.Canny(doc_blur, 75, 200)

    val_cnts, _ = cv2.findContours(doc_edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # Convert tuple of arrays to list before slicing to satisfy static type checker
    val_cnts_list = list(val_cnts)
    val_cnts_sorted = sorted(val_cnts_list, key=cv2.contourArea, reverse=True)[:5]
    
    doc_cnt = None
    for c in val_cnts_sorted:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            # Check if area is large enough to be the paper (at least 20% of image)
            if cv2.contourArea(c) > (img.shape[0]*img.shape[1] * 0.20):
                doc_cnt = approx
                break

    # Apply perspective transform if paper rectangle is found securely
    if doc_cnt is not None and len(doc_cnt) == 4:
        # Cast to numpy array to satisfy type checker
        doc_cnt_arr = np.array(doc_cnt).reshape(4, 2)
        img = _four_point_transform(img, doc_cnt_arr)
    
    # 3. Resize to 1500px width maintaining aspect ratio
    height, width = img.shape[:2]
    new_width = 1500
    ratio = new_width / float(width)
    new_height = int(height * ratio)
    img_resized = cv2.resize(img, (new_width, new_height))
    
    # 4. Grayscale
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    
    # 5. Deskew using Hough Transform (Fallback if paper crop missed minor rotation)
    thresh_deskew = cv2.adaptiveThreshold(
        cv2.GaussianBlur(gray, (9, 9), 0), 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )
    
    lines = cv2.HoughLinesP(thresh_deskew, 1, np.pi / 180, 100, 
                            minLineLength=100, maxLineGap=20)
    angle = 0.0
    if lines is not None:
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            a = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            if -30 < a < 30:
                angles.append(a)
        if len(angles) > 0:
            angle = np.median(angles)
    
    # Rotate if skew is significant
    if abs(angle) > 0.5:
        (h, w) = gray.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        gray = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_REPLICATE)
        img_resized = cv2.warpAffine(img_resized, M, (w, h), flags=cv2.INTER_CUBIC,
                                     borderMode=cv2.BORDER_REPLICATE)
    
    # 6. Binarization for grid detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )
    
    return img_resized, gray, thresh


def detect_tables_morphological(thresh: np.ndarray, 
                                 min_cell_area: int = 1000) -> List[Dict]:
    """
    Detect table cells using morphological operations (Hough Transform approach).
    
    Uses horizontal and vertical line detection kernels to find grid structure,
    then extracts individual cells from the grid.
    Introduced a pre-dilation step to actively bridge faded/faint ink gaps before
    opening, preventing internal cell walls from merging on badly printed sheets.
    
    Returns:
        List of cell dicts with keys: x, y, w, h
    """
    # Dynamic scaling based on the image size
    scale = thresh.shape[0] / 2000.0
    
    # 1. Detect horizontal lines (with pre-dilation to fix fading)
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (int(40 * scale) if scale > 0.5 else 40, 1))
    h_dilate_k = cv2.getStructuringElement(cv2.MORPH_RECT, (int(10 * scale) if scale > 0.5 else 10, 1))
    thresh_h_dilated = cv2.dilate(thresh, h_dilate_k, iterations=1)
    h_lines = cv2.morphologyEx(thresh_h_dilated, cv2.MORPH_OPEN, h_kernel)
    
    # 2. Detect vertical lines (with pre-dilation)
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, int(40 * scale) if scale > 0.5 else 40))
    v_dilate_k = cv2.getStructuringElement(cv2.MORPH_RECT, (1, int(10 * scale) if scale > 0.5 else 10))
    thresh_v_dilated = cv2.dilate(thresh, v_dilate_k, iterations=1)
    v_lines = cv2.morphologyEx(thresh_v_dilated, cv2.MORPH_OPEN, v_kernel)
    
    # Combine horizontal and vertical structures
    table_mask = cv2.add(h_lines, v_lines)
    
    # Apply closing to fuse crossing junctions seamlessly
    cross_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    table_mask = cv2.morphologyEx(table_mask, cv2.MORPH_CLOSE, cross_kernel)
    
    # Find contours on the mask
    contours, hierarchy = cv2.findContours(
        table_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    
    cells = []
    img_area = thresh.shape[0] * thresh.shape[1]
    
    if hierarchy is not None:
        for i, cnt in enumerate(contours):
            # Only look at leaf nodes (cells without children)
            if hierarchy[0][i][2] == -1:
                x, y, w, h = cv2.boundingRect(cnt)
                area = w * h
                if min_cell_area < area < (img_area * 0.5) and w > 15 and h > 15:
                    cells.append({"x": x, "y": y, "w": w, "h": h})
    
    return cells


def detect_tables_img2table(image_path: str) -> Optional[List]:
    """
    Detect tables using the img2table library (Paper §4.2).
    
    The img2table library uses OpenCV image processing and OCR to find
    table regions and individual cell positions.
    
    Returns: 
        List of extracted tables with cell positions, or None if library unavailable
    """
    try:
        from img2table.document import Image as Img2TableImage
        from img2table.ocr import TesseractOCR
        
        ocr = TesseractOCR(lang="eng")
        doc = Img2TableImage(src=image_path)
        tables = doc.extract_tables(ocr=ocr, implicit_rows=True, 
                                    implicit_columns=True,
                                    borderless_tables=False)
        return tables
    except ImportError:
        print("img2table not available, falling back to morphological detection")
        return None
    except Exception as e:
        print(f"img2table error: {e}, falling back to morphological detection")
        return None


def group_cells_into_rows(cells: List[Dict], 
                          y_threshold: int = 30) -> List[List[Dict]]:
    """
    Group detected cells into rows based on Y-coordinate proximity.
    
    Args:
        cells: list of cell dicts with x, y, w, h
        y_threshold: max vertical distance for cells to be in the same row
    
    Returns:
        List of rows, where each row is a list of cells sorted left-to-right
    """
    if not cells:
        return []
    
    # Sort by Y coordinate
    cells_sorted = sorted(cells, key=lambda c: c['y'])
    
    rows = []
    current_row = [cells_sorted[0]]
    
    for c in cells_sorted[1:]:
        if abs(c['y'] - current_row[0]['y']) < y_threshold:
            current_row.append(c)
        else:
            # Sort current row left-to-right
            current_row.sort(key=lambda c: c['x'])
            rows.append(current_row)
            current_row = [c]
    
    # Don't forget last row
    current_row.sort(key=lambda c: c['x'])
    rows.append(current_row)
    
    return rows


def detect_red_regions(img_color: np.ndarray) -> np.ndarray:
    """
    Detect red handwritten marks in the image (Paper §4.3).
    
    The table containing the most red elements is assumed to be the mark table,
    as marks are usually written with a red pen.
    
    Returns:
        Binary mask where red regions are white
    """
    hsv = cv2.cvtColor(img_color, cv2.COLOR_BGR2HSV)
    
    # Red color has two ranges in HSV (wraps around 0/180)
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])
    
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    
    red_mask = cv2.bitwise_or(mask1, mask2)
    
    # Clean up noise
    kernel = np.ones((3, 3), np.uint8)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
    
    return red_mask


def find_table_regions(thresh: np.ndarray) -> List[Dict]:
    """
    Find distinct table regions in the image.
    
    Returns:
        List of table bounding boxes: {x, y, w, h, cells: [...]}
    """
    cells = detect_tables_morphological(thresh)
    
    if not cells:
        return []
    
    rows = group_cells_into_rows(cells)
    
    # Group rows into distinct tables based on vertical gaps
    tables = []
    current_table_rows = [rows[0]]
    
    for i in range(1, len(rows)):
        prev_row_bottom = max(c['y'] + c['h'] for c in current_table_rows[-1])
        curr_row_top = min(c['y'] for c in rows[i])
        
        gap = curr_row_top - prev_row_bottom
        
        if gap > 100:  # Large gap = new table
            # Save current table
            all_cells = [c for row in current_table_rows for c in row]
            if all_cells:
                xs = [c['x'] for c in all_cells]
                ys = [c['y'] for c in all_cells]
                x_max = [c['x'] + c['w'] for c in all_cells]
                y_max = [c['y'] + c['h'] for c in all_cells]
                tables.append({
                    "x": min(xs), "y": min(ys),
                    "w": max(x_max) - min(xs),
                    "h": max(y_max) - min(ys),
                    "rows": current_table_rows,
                    "cells": all_cells
                })
            current_table_rows = [rows[i]]
        else:
            current_table_rows.append(rows[i])
    
    # Save last table
    all_cells = [c for row in current_table_rows for c in row]
    if all_cells:
        xs = [c['x'] for c in all_cells]
        ys = [c['y'] for c in all_cells]
        x_max = [c['x'] + c['w'] for c in all_cells]
        y_max = [c['y'] + c['h'] for c in all_cells]
        tables.append({
            "x": min(xs), "y": min(ys),
            "w": max(x_max) - min(xs),
            "h": max(y_max) - min(ys),
            "rows": current_table_rows,
            "cells": all_cells
        })
    
    return tables
