import cv2
import numpy as np
import easyocr
import os
import argparse
import traceback

def extract_objective_marks_from_image(img, debug=True, output_dir="debug_output"):
    """
    Extracts marks from the top-left area of an objective sheet from an image array.
    Assumes marks are written in red ink and the max marks is 10.
    Returns (score, confidence) tuple.
    """
    if img is None:
        return None, 0.0

    H, W = img.shape[:2]
    
    # 1. Crop to top-right corner
    # Assuming marks are in the top 45% of height and right 50% of width
    crop_h, crop_w = int(H * 0.45), int(W * 0.5)
    roi = img[0:crop_h, W - crop_w:W]
    
    if debug:
        cv2.imwrite(os.path.join(output_dir, "obj_1_roi.png"), roi)
        print("[+] Saved ROI crop.")

    # 2. Extract Red Ink
    # Convert to HSV 
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    
    # We use very broad red boundaries to capture faded or dark red ink
    lower_red1 = np.array([0, 20, 20])
    upper_red1 = np.array([15, 255, 255])
    lower_red2 = np.array([160, 20, 20])
    upper_red2 = np.array([180, 255, 255])
    
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask1, mask2)
    
    # Also create an alternative mask using RGB space directly just in case HSV misses it
    # Red should be significantly higher than Blue and Green
    b, g, r = cv2.split(roi)
    rgb_red_mask = ((r.astype(np.int16) - g.astype(np.int16) > 30) & 
                    (r.astype(np.int16) - b.astype(np.int16) > 30))
    rgb_red_mask = (rgb_red_mask * 255).astype(np.uint8)
    
    # Combine both strategies for maximum robustness
    combined_mask = cv2.bitwise_or(red_mask, rgb_red_mask)

    if debug:
        cv2.imwrite(os.path.join(output_dir, "obj_2_red_mask.png"), combined_mask)
        print("[+] Saved red mask.")

    # Apply morphology to connect disjoint parts of the pen strokes
    kernel = np.ones((5,5), np.uint8)
    mask_cleaned = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    mask_cleaned = cv2.dilate(mask_cleaned, kernel, iterations=1)

    if debug:
        cv2.imwrite(os.path.join(output_dir, "obj_3_cleaned_mask.png"), mask_cleaned)

    # 3. Find bounding box of all red markings
    contours, _ = cv2.findContours(mask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        print("[-] No red ink found in the ROI.")
        print("[*] Falling back to analyzing the entire ROI with EasyOCR...")
        target_crop = roi
        target_name = "full_roi"
    else:
        # Filter contours by size to avoid noise
        valid_contours = [c for c in contours if cv2.contourArea(c) > 50]
        
        if not valid_contours:
            print("[-] Red areas found, but too small. Falling back to whole ROI.")
            target_crop = roi
            target_name = "full_roi"
        else:
            # Find cohesive bounding box (ignores stray marks far from the main circle)
            largest_c = max(valid_contours, key=cv2.contourArea)
            M = cv2.moments(largest_c)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                bx, by, bw, bh = cv2.boundingRect(largest_c)
                cx, cy = bx + bw // 2, by + bh // 2

            cohesive_contours = []
            for c in valid_contours:
                bx, by, bw, bh = cv2.boundingRect(c)
                bcx, bcy = bx + bw / 2, by + bh / 2
                dist = np.sqrt((bcx - cx)**2 + (bcy - cy)**2)
                if dist < 120:  # Threshold to group the score circle and ignore distant stray marks
                    cohesive_contours.append(c)

            if not cohesive_contours:
                cohesive_contours = [largest_c]

            all_points = np.vstack(cohesive_contours)
            x, y, w, h = cv2.boundingRect(all_points)

            pad = 20
            x_min = max(0, x - pad)
            y_min = max(0, y - pad)
            x_max = min(crop_w, x + w + pad)
            y_max = min(crop_h, y + h + pad)
            
            # We crop the original image but isolate just the red ink on white bg 
            # to prevent printed black text from confusing OCR
            red_ink_only = np.full(roi.shape, 255, dtype=np.uint8)
            red_ink_only[mask_cleaned == 255] = [0, 0, 0] # Make red ink black
            
            target_crop = red_ink_only[y_min:y_max, x_min:x_max]
            target_name = "red_ink_crop"
    
    if debug:
        cv2.imwrite(os.path.join(output_dir, f"obj_4_{target_name}.png"), target_crop)
        print(f"[+] Saved final extraction crop for OCR.")

    # 4. Use Recognizers to read the marks
    if target_name == "red_ink_crop":
        print("[*] Initializing custom Objective Marks Recognizer (ObjNet)...")
        import sys
        
        # Add obj_model to path to import ObjNet
        obj_model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "obj_model")
        if obj_model_path not in sys.path:
            sys.path.insert(0, obj_model_path)
            
        try:
            from obj_net import ObjMarksRecognizer
            # Using default path models/obj_marks_model.pth inside obj_model
            model_file = os.path.join(obj_model_path, "models", "obj_marks_model.pth")
            recognizer = ObjMarksRecognizer(model_file)
            
            if recognizer.model_loaded:
                # ObjNet expects the entire score crop (it handles the "7/10" whole)
                score, conf_val = recognizer.predict_from_image(target_crop)
                if score is not None:
                    print(f"   -> ObjNet detected score: '{score}' (conf: {conf_val:.2f})")
                    best_marks = score
                    max_conf = conf_val
                else:
                    best_marks = None
            else:
                best_marks = None
                max_conf = 0.0
        except Exception as e:
            print(f"[-] Failed to load ObjNet: {e}")
            best_marks = None
            max_conf = 0.0
            
    else:
        print("[*] Red ink failed. Loading EasyOCR for full ROI fallback...")
        import logging
        logging.getLogger("easyocr").setLevel(logging.ERROR)
        reader = easyocr.Reader(['en'], gpu=True, verbose=False)
        results = reader.readtext(target_crop)
        
        best_marks = None
        max_conf = 0.0
        
        for r in results:
            text, conf = str(r[1]).strip(), r[2]
            print(f"   -> OCR Raw text detected: '{text}' (conf: {conf:.2f})")
            
            if len(text) > 5:
                continue
                
            clean_text = text.split('/')[0].strip()
            clean_text = ''.join(c for c in clean_text if c.isdigit())
            
            if clean_text.isdigit():
                val = int(clean_text)
                if 0 <= val <= 10:
                    if conf > max_conf:
                        best_marks = val
                        max_conf = conf

    if best_marks is not None:
        print(f"=========================================")
        print(f"[SUCCESS] Final Objective Marks: {best_marks} / 10")
        print(f"=========================================")
        return best_marks, max_conf
    else:
        print("[-] Could not confidently recognize marks <= 10.")
        return None, 0.0


def extract_objective_marks(image_path, debug=True, output_dir="debug_output"):
    """
    Extracts marks from the top-left area of an objective sheet (file path version).
    Assumes marks are written in red ink and the max marks is 10.
    """
    if not os.path.exists(image_path):
        print(f"[-] File not found: {image_path}")
        return None

    if debug and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"[*] Processing {image_path}...")
    img = cv2.imread(image_path)
    
    score, conf = extract_objective_marks_from_image(img, debug, output_dir)
    return score


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract Objective Sheet Marks")
    parser.add_argument("--image", default="objective_sheet.jpeg", help="Path to objective sheet image")
    args = parser.parse_args()
    
    try:
        extract_objective_marks(args.image)
    except Exception as e:
        print("[-] Error during extraction:")
        traceback.print_exc()
