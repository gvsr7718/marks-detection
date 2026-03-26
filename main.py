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
    extract_marks_grid_template,
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
    Process a descriptive answer sheet using template-aware grid extraction.
    """
    # ── Stage 1: Extract marks grid (template-aware) ────────────────────────
    mark_cells, total_cell, table_info = extract_marks_grid_template(
        gray, thresh, img_color, debug_dir=DEBUG_DIR
    )
    print(f"[DEBUG] Extracted {len(mark_cells)} mark cells, "
          f"total_cell={'yes' if total_cell is not None else 'no'}")

    # Debug: save each extracted cell
    for i, cell in enumerate(mark_cells):
        cv2.imwrite(f"{DEBUG_DIR}/06_cell_{i}.jpg", cell)
        print(f"[DEBUG]   Cell {i}: shape={cell.shape}")
    if total_cell is not None:
        cv2.imwrite(f"{DEBUG_DIR}/06_cell_total.jpg", total_cell)

    # ── Stage 2: Recognise digits in each cell ──────────────────────────────
    question_labels = ['q1a', 'q1b', 'q2a', 'q2b', 'q3a', 'q3b',
                       'q4a', 'q4b', 'q5a', 'q5b', 'q6a', 'q6b']
    marks = {}
    descriptive_total = 0

    for i, cell_img in enumerate(mark_cells):
        if i >= len(question_labels):
            break
        label = question_labels[i]

        # --- Stage 2.1: Pre-process cell (Center Crop to remove borders) ---
        h, w = cell_img.shape[:2]
        margin_h, margin_w = int(h * 0.05), int(w * 0.05)
        inner_cell = cell_img[margin_h:h-margin_h, margin_w:w-margin_w]

        # Primary: EasyOCR on the cell crop (more robust for handwritten)
        easyocr_val, easyocr_conf, _ = _easyocr_read_cell(recognizer, inner_cell, max_val_allowed=5)

        # Fallback: contour extraction + AlexNet
        digit_images = extract_digit_contours(inner_cell)
        alexnet_val, alexnet_conf = recognizer.recognize_marks_from_cell(digit_images)

        # Use EasyOCR if it returned a plausible value; else AlexNet
        if easyocr_val is not None and 0 <= easyocr_val <= 5:
            mark_val = easyocr_val
            conf = easyocr_conf
            src = "easyocr"
        else:
            mark_val = min(alexnet_val, 5) if isinstance(alexnet_val, int) else 5
            conf = alexnet_conf
            src = "alexnet"
            
        # Consensus Voter: Precision corrections for common handwritten confusions
        # We enforce a strict bias towards '1' or '2' if a '5' is disputed.
        # We also fix '3' hallucinated from a '5', and a '3' hallucinated as a '2'.
        if mark_val == 5:
            if alexnet_val == 1 or easyocr_val == 1:
                mark_val = 1
                src = "consensus_override (5->1)"
            elif alexnet_val == 2 or easyocr_val == 2:
                mark_val = 2
                src = "consensus_override (5->2)"
        elif mark_val == 3 and (alexnet_val == 5 or easyocr_val == 5):
            mark_val = 5
            src = "consensus_override (3->5)"
        elif mark_val == 2 and (alexnet_val == 3 or easyocr_val == 3):
            # EasyOCR frequently collapses sloppy '3's into '2's. We rescue the '3'.
            mark_val = 3
            src = "consensus_override (2->3)"
        elif mark_val == 0 and (alexnet_val == 1 or easyocr_val == 1):
            # Sometimes thin '1's are missed by one model
            mark_val = 1
            src = "consensus_override (0->1)"

        marks[label] = {"value": mark_val, "confidence": round(conf, 2)}
        print(f"[DEBUG]   {label}: {mark_val} (conf={conf:.2f}, src={src}) "
              f"[easyocr={easyocr_val}, alexnet={alexnet_val}]")

    # Fill any missing labels with 0
    for label in question_labels:
        if label not in marks:
            marks[label] = {"value": 0, "confidence": 0.0}

    # ── Stage 3: Total ──────────────────────────────────────────────────────
    if total_cell is not None:
        easyocr_total, _, total_candidates = _easyocr_read_cell(recognizer, total_cell, max_val_allowed=100)
        digit_images = extract_digit_contours(total_cell)
        alexnet_total, _ = recognizer.recognize_marks_from_cell(digit_images)
        calc_total = sum(m["value"] for m in marks.values())
        
        # Valid extracted total is easyocr or alexnet (trust easyocr first if valid)
        extracted_total = easyocr_total if (easyocr_total is not None and 0 <= easyocr_total <= 100) else alexnet_total

        # --- Total Validation Heuristics ---
        # Heuristic 1: Explicit Total Match
        # The handwriting for '2' is frequently misread by OCR as '9' (diff 7).
        # Since the TOTAL cell might contain fractions (e.g. '14', '20'), we check all extracted numbers!
        validation_candidates = set(total_candidates)
        if extracted_total is not None:
            validation_candidates.add(extracted_total)
            
        corrected_via_explicit_total = False
        for candidate_total in validation_candidates:
            if calc_total - candidate_total == 7:
                nines = [k for k, v in marks.items() if v["value"] == 9]
                if len(nines) == 1:
                    marks[nines[0]]["value"] = 2
                    print(f"[DEBUG] Total Validation Heuristic: corrected {nines[0]} from 9 to 2 to match explicit Total={candidate_total}")
                    calc_total = sum(m["value"] for m in marks.values())
                    extracted_total = candidate_total  # Lock in this valid total
                    corrected_via_explicit_total = True
                    break

        # Heuristic 2: Exam Max Marks Overflow boundary
        # If the explicit total cell was unreadable, but our computed total OVERFLOWS the absolute exam maximum (20),
        # we can mathematically deduce any extracted '9's MUST be hallucinations of '2'.
        if not corrected_via_explicit_total and calc_total > 20:
            nines = [k for k, v in marks.items() if v["value"] == 9]
            for nine_key in nines:
                marks[nine_key]["value"] = 2
                calc_total -= 7
                print(f"[DEBUG] Overflow Heuristic: corrected {nine_key} from 9 to 2 because stringently exceeded max_marks (20)")
                if calc_total <= 20:
                    break

        # Pick the most plausible total
        descriptive_total = calc_total
        if extracted_total is not None and extracted_total == calc_total:
            descriptive_total = extracted_total
            
        print(f"[DEBUG] Total: candidates={total_candidates}, alexnet={alexnet_total}, "
              f"calc={calc_total} → using {descriptive_total}")
    else:
        descriptive_total = sum(m["value"] for m in marks.values())

    # Absolute mathematical boundary applied globally to the test
    if descriptive_total > 20:
        print(f"[DEBUG] Mathematical Constraint: Total {descriptive_total} > 20. Forced down to 20.")
        descriptive_total = 20

    # ── Stage 4: HT Number ──────────────────────────────────────────────────
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


def _easyocr_read_cell(recognizer, cell_img, max_val_allowed=100):
    """
    Use EasyOCR to read a single digit (0-10) from a small cell crop.
    Uses CLAHE contrast enhancement to handle faint handwriting.
    Returns (int_value, confidence) or (None, 0.0) on failure.
    """
    import cv2 as _cv2
    import numpy as _np
    try:
        # Step 1: Enhance contrast with CLAHE (handles faint pencil marks)
        clahe = _cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
        enhanced = clahe.apply(cell_img)

        # Step 2: Upscale 4x for better OCR
        big = _cv2.resize(enhanced, None, fx=4, fy=4,
                          interpolation=_cv2.INTER_CUBIC)

        # Step 3: Otsu binarization on enhanced image
        _, binary = _cv2.threshold(big, 0, 255,
                                   _cv2.THRESH_BINARY + _cv2.THRESH_OTSU)

        # Step 4: Empty-cell detection on binarized image
        # After Otsu binarization, dark pixels (0) represent ink marks
        dark_pixels = _np.sum(binary == 0)
        total_pixels = binary.size
        dark_ratio = dark_pixels / total_pixels
        if dark_ratio < 0.005:  # lowered to 0.5% ink to capture faint thin '1's
            return None, 0.0

        # Step 5: Add white border for OCR
        bordered = _cv2.copyMakeBorder(
            binary, 20, 20, 20, 20,
            _cv2.BORDER_CONSTANT, value=255
        )
        rgb = _cv2.cvtColor(bordered, _cv2.COLOR_GRAY2RGB)

        # Step 6: EasyOCR digit recognition
        recognizer._initialize()
        results = recognizer.reader.readtext(
            rgb, allowlist='0123456789', paragraph=False
        )
        
        # We can return either the absolute best value (for single marks)
        # or ALL valid numbers (for checking the TOTAL cell which might be a fraction '14/20')
        valid_numbers = []
        if results:
            for r in results:
                text = r[1].strip()
                if text.isdigit():
                    v = int(text)
                    if 0 <= v <= max_val_allowed:
                        valid_numbers.append((v, float(r[2])))
            
            if valid_numbers:
                best = max(valid_numbers, key=lambda x: x[1])
                return best[0], best[1], [v[0] for v in valid_numbers]
                
        return None, 0.0, []
    except Exception:
        return None, 0.0, []


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
