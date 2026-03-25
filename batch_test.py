import os
import cv2
import json
import glob
from table_detector import preprocess_image
from mark_extractor import extract_marks_grid_template, _debug_draw_table, extract_ht_number_boxes
from digit_recognizer import get_digit_recognizer

import sys

def process_batch(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    print(f"Scanning directory: {input_dir}")
    # Support various image formats
    image_paths = glob.glob(os.path.join(input_dir, "*.[jJ][pP][gG]")) + \
                  glob.glob(os.path.join(input_dir, "*.[jJ][pP][eE][gG]")) + \
                  glob.glob(os.path.join(input_dir, "*.[pP][nN][gG]"))
                  
    if not image_paths:
        print("No images found in the specified directory.")
        return
        
    print(f"Found {len(image_paths)} images. Processing...")
    
    # We will ONLY do detection of tables and drawing, to visibly confirm parsing.
    # The OCR process is slow and unneeded just to verify table boxes bounding! 
    # But since the user wants to check overall functionality, we can extract the bounding boxes and draw them.
    
    for idx, path in enumerate(image_paths):
        basename = os.path.basename(path)
        print(f"[{idx+1}/{len(image_paths)}] Processing: {basename}")
        
        try:
            with open(path, 'rb') as f:
                img_bytes = f.read()
                
            img_color, gray, thresh = preprocess_image(img_bytes)
            
            # Create a debug canvas to draw BOTH tables on
            canvas = img_color.copy()
            
            # --- Detect Marks Grid ---
            mark_cells, total_cell, table_info = extract_marks_grid_template(gray, thresh, img_color, debug_dir=None)
            
            if table_info:
                # Draw Marks Table
                rows = table_info.get('rows', [])
                for ri, row in enumerate(rows):
                    for ci, c in enumerate(row):
                        colour = (0, 0, 255) if ri == len(rows) - 1 else (0, 255, 0)
                        cv2.rectangle(canvas,
                                      (c['x'], c['y']),
                                      (c['x'] + c['w'], c['y'] + c['h']),
                                      colour, 2)
                        cv2.putText(canvas, f"R{ri}C{ci}",
                                    (c['x'] + 2, c['y'] + 15),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, colour, 1)
            
            # --- Detect HT Number ---
            ht_data, ht_boxes_crops = extract_ht_number_boxes(gray, thresh)
            if ht_data is not None:
                full_row_crop, cell_coords_rel = ht_data
                h_crop, w_crop = full_row_crop.shape
                
                # draw boxes on the HT row crop
                ht_canvas = cv2.cvtColor(full_row_crop, cv2.COLOR_GRAY2BGR)
                for bx, by, bw, bh in cell_coords_rel:
                    cv2.rectangle(ht_canvas, (bx, by), (bx + bw, by + bh), (255, 0, 0), 2)
                
                # Overlay it onto the main canvas (top left corner)
                ch, cw = canvas.shape[:2]
                if h_crop <= ch and w_crop <= cw:
                    canvas[10:10+h_crop, 10:10+w_crop] = ht_canvas
                    cv2.rectangle(canvas, (10, 10), (10+w_crop, 10+h_crop), (255, 0, 0), 4)
            
            # Save the annotated canvas
            out_path = os.path.join(output_dir, f"detected_{basename}")
            cv2.imwrite(out_path, canvas)
            
        except Exception as e:
            print(f"FAILED on {basename}: {e}")

if __name__ == '__main__':
    input_folder = r"c:\Users\ragha\OneDrive\Desktop\Descriptive Sheets"
    output_folder = r"c:\Users\ragha\OneDrive\Desktop\VS CODE\Projects\marks_detection\exports\batch_results"
    
    if len(sys.argv) > 1:
        input_folder = sys.argv[1]
    if len(sys.argv) > 2:
        output_folder = sys.argv[2]
        
    process_batch(input_folder, output_folder)
    print(f"\nDone. Results saved to: {output_folder}")
