"""
FastAPI Application — AI-Powered Mark Recognition
Routes:
- POST /api/process (image + sheet_type → JSON with extracted marks)
- POST /api/export  (entries → Excel file download)

Based on: "AI-Powered Mark Recognition in Assessment and Attainment Calculation"
         by J. Annrose et al. (ICTACT, Jan 2025)
"""

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, StreamingResponse
from pydantic import BaseModel
from typing import List, Dict, Any
import os
import io
import traceback
import cv2
import numpy as np

from table_detector import preprocess_image, find_table_regions
from mark_extractor import (
    identify_grading_tables,
    find_marked_table,
    extract_mark_cells,
    extract_ht_number_boxes,
    extract_mcq_score_region,
    extract_digit_contours
)
from digit_recognizer import get_digit_recognizer
from spreadsheet_export import generate_combined_excel

# Debug directory
DEBUG_DIR = "debug_output"

app = FastAPI(title="AI-Powered Mark Recognition System")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    """Initialize directories and models on startup."""
    os.makedirs("models", exist_ok=True)
    os.makedirs("exports", exist_ok=True)
    os.makedirs("static", exist_ok=True)
    os.makedirs(DEBUG_DIR, exist_ok=True)
    # Pre-load the digit recognizer model
    get_digit_recognizer()


# Mount frontend
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def read_root():
    return RedirectResponse(url="/static/index.html")


@app.post("/api/process")
async def process_image(file: UploadFile = File(...), sheet_type: str = Form(...)):
    """
    Process an uploaded answer sheet image and extract marks.
    """
    image_bytes = await file.read()
    
    try:
        # Stage 1: Preprocessing
        img_color, gray, thresh = preprocess_image(image_bytes)
        recognizer = get_digit_recognizer()
        
        # Save debug preprocessed images
        cv2.imwrite(f"{DEBUG_DIR}/01_gray.jpg", gray)
        cv2.imwrite(f"{DEBUG_DIR}/02_thresh.jpg", thresh)
        cv2.imwrite(f"{DEBUG_DIR}/03_color.jpg", img_color)
        print(f"[DEBUG] Image size: {gray.shape}")
        
        if sheet_type == "descriptive":
            return _process_descriptive(img_color, gray, thresh, recognizer)
        elif sheet_type == "mcq":
            return _process_mcq(img_color, gray, thresh, recognizer)
        else:
            raise HTTPException(status_code=400, detail="Unknown sheet type.")
    
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


def _process_descriptive(img_color, gray, thresh, recognizer) -> Dict[str, Any]:
    """
    Process a descriptive answer sheet with debug output.
    """
    # Stage 1: Detect tables
    tables = find_table_regions(thresh)
    print(f"[DEBUG] Found {len(tables)} table regions")
    
    # Debug: draw all detected tables
    debug_img = cv2.cvtColor(gray.copy(), cv2.COLOR_GRAY2BGR)
    for i, t in enumerate(tables):
        cv2.rectangle(debug_img, (t['x'], t['y']), 
                      (t['x']+t['w'], t['y']+t['h']), (0, 255, 0), 2)
        cv2.putText(debug_img, f"Table {i} ({len(t.get('cells',[]))} cells)", 
                    (t['x'], t['y']-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        print(f"[DEBUG]   Table {i}: x={t['x']}, y={t['y']}, w={t['w']}, h={t['h']}, "
              f"rows={len(t.get('rows',[]))}, cells={len(t.get('cells',[]))}")
    cv2.imwrite(f"{DEBUG_DIR}/04_detected_tables.jpg", debug_img)
    
    # Identify grading tables
    grading_tables = identify_grading_tables(gray, tables)
    print(f"[DEBUG] Grading tables found: {len(grading_tables)}")
    
    if not grading_tables:
        grading_tables = tables
        print("[DEBUG] No grading tables via OCR, using all tables")
    
    # Find marked table
    marked_table = find_marked_table(img_color, grading_tables)
    
    marks = {}
    descriptive_total = 0
    
    if marked_table:
        print(f"[DEBUG] Marked table: x={marked_table['x']}, y={marked_table['y']}, "
              f"rows={len(marked_table.get('rows',[]))}")
        
        # Debug: draw marked table rows
        debug_marked = cv2.cvtColor(gray.copy(), cv2.COLOR_GRAY2BGR)
        for ri, row in enumerate(marked_table.get('rows', [])):
            for ci, c in enumerate(row):
                color = (0, 0, 255) if ri == len(marked_table.get('rows',[])) - 1 else (0, 255, 0)
                cv2.rectangle(debug_marked, (c['x'], c['y']), 
                              (c['x']+c['w'], c['y']+c['h']), color, 2)
                cv2.putText(debug_marked, f"R{ri}C{ci}", (c['x']+2, c['y']+15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        cv2.imwrite(f"{DEBUG_DIR}/05_marked_table_cells.jpg", debug_marked)
        
        mark_cells, total_cell = extract_mark_cells(gray, marked_table)
        print(f"[DEBUG] Extracted {len(mark_cells)} mark cells, total_cell={'yes' if total_cell is not None else 'no'}")
        
        # Debug: save each extracted cell
        for i, cell in enumerate(mark_cells):
            cv2.imwrite(f"{DEBUG_DIR}/06_cell_{i}.jpg", cell)
            print(f"[DEBUG]   Cell {i}: shape={cell.shape}")
        
        if total_cell is not None:
            cv2.imwrite(f"{DEBUG_DIR}/06_cell_total.jpg", total_cell)
        
        question_labels = ['q1a', 'q1b', 'q2a', 'q2b', 'q3a', 'q3b',
                          'q4a', 'q4b', 'q5a', 'q5b', 'q6a', 'q6b']
        
        for i, cell_img in enumerate(mark_cells):
            if i < len(question_labels):
                digit_images = extract_digit_contours(cell_img)
                print(f"[DEBUG]   {question_labels[i]}: {len(digit_images)} digit contours found")
                
                # Save digit contour images
                for di, dimg in enumerate(digit_images):
                    cv2.imwrite(f"{DEBUG_DIR}/07_digit_{question_labels[i]}_{di}.jpg", dimg)
                
                mark_val, conf = recognizer.recognize_marks_from_cell(digit_images)
                marks[question_labels[i]] = {
                    "value": mark_val,
                    "confidence": round(conf, 2)
                }
                print(f"[DEBUG]   {question_labels[i]}: recognized={mark_val}, conf={conf:.2f}")
        
        for label in question_labels:
            if label not in marks:
                marks[label] = {"value": 0, "confidence": 0.0}
        
        if total_cell is not None:
            digit_images = extract_digit_contours(total_cell)
            descriptive_total, _ = recognizer.recognize_marks_from_cell(digit_images)
            print(f"[DEBUG] OCR Total: {descriptive_total}")
            calc_total = sum(m["value"] for m in marks.values())
            if descriptive_total > 20 or descriptive_total != calc_total:
                print(f"[DEBUG] Fallback to calculated total: {calc_total} instead of {descriptive_total}")
                descriptive_total = calc_total
        else:
            descriptive_total = sum(m["value"] for m in marks.values())
    else:
        print("[DEBUG] No marked table found!")
    
    # Extract HT Number
    ht_row_data, ht_boxes = extract_ht_number_boxes(gray, thresh)
    print(f"[DEBUG] HT Number boxes found: {len(ht_boxes)}")
    for i, box in enumerate(ht_boxes):
        cv2.imwrite(f"{DEBUG_DIR}/08_ht_box_{i}.jpg", box)
    
    ht_no, ht_conf = recognizer.recognize_ht_number(ht_row_data, ht_boxes)
    print(f"[DEBUG] HT Number: {ht_no}, conf={ht_conf:.2f}")
    
    return {
        "ht_no": {"value": ht_no, "confidence": round(ht_conf, 2)},
        "marks": marks,
        "descriptive_total": descriptive_total,
        "max_marks": 20
    }


def _process_mcq(img_color, gray, thresh, recognizer) -> Dict[str, Any]:
    """
    Process an MCQ answer sheet.
    
    Extracts: HT No / Roll No, MCQ Score
    """
    # Extract HT/Roll Number
    ht_row_data, ht_boxes = extract_ht_number_boxes(gray, thresh)
    ht_no, ht_conf = recognizer.recognize_ht_number(ht_row_data, ht_boxes)
    
    # Extract Score
    score_roi = extract_mcq_score_region(gray)
    mcq_score, score_conf = recognizer.recognize_score(score_roi)
    
    return {
        "ht_no": {"value": ht_no, "confidence": round(ht_conf, 2)},
        "mcq_score": mcq_score,
        "max_marks": 10
    }


class ExportRequest(BaseModel):
    entries: List[Dict]


@app.post("/api/export")
async def export_excel(payload: ExportRequest):
    """
    Export collected entries to a combined Excel file.
    
    Format: S.No | HT No | Q1a..Q6b | Descriptive Total | MCQ Score | Total
    """
    entries = payload.entries
    
    if not entries:
        raise HTTPException(status_code=400, detail="No entries provided")
    
    try:
        excel_bytes = generate_combined_excel(entries)
        
        headers = {
            'Content-Disposition': 'attachment; filename="marks_report.xlsx"'
        }
        return StreamingResponse(
            io.BytesIO(excel_bytes),
            headers=headers,
            media_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
