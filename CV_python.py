"""
Billiard Project (part 1) python script

course: Computer Vision (M.IA005)
year: 2025/2026
authors: Angelina Moreira, Elske , Guilherme Gonçalves, Maria Shi

What this file consists of:
    - Reading from a JSON file
    - Pre-processing of the images
    - Ball detection
    - Classifying of the balls
    - Outputing the classifications and positions of the balls for each image
    - Warping the image to a top-view
    - Saving the warped image

This script does contain dependent files:

"""

from PIL import Image

from matplotlib.widgets import Slider, Button
from billard_detector import detect_billiard_balls, run_on_directory

import cv2
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

import os
import numpy as np
import math
import json

INPUT_DIR = Path("./development_set")
OUTPUT_DIR = Path("./processed_images")

# Reading from JSON file
def read_json():
    pass

# Pre-processing of the images
class Preprocessing():
    def __init__(self, height, width):
        self.height = height
        self.width = width

    def _verify_resolutions(self):
        data = []

        for img_path in INPUT_DIR.rglob('*.jpg'):
            try:
                with Image.open(img_path) as img:
                    width, height = img.size
                    data.append({
                        "Width": width,
                        "Height": height
                    })
            except Exception as e:
                pass

        return pd.DataFrame(data)
    
    def plot_resolutions(self):
        df = self._verify_resolutions()
        
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        sns.scatterplot(data=df, x="Width", y="Height", alpha=0.5, edgecolor=None)
        plt.title("Raw Image Dimensions")
        plt.xlabel("Width (pixels)")
        plt.ylabel("Height (pixels)")

        plt.tight_layout()
        plt.show()
    
    def _process_and_save_image(self, file_path, out_path):
        img = cv2.imread(str(file_path))

        if img is None:
            print(f"Warning: Could not read {file_path}")
            return

        img_resized = cv2.resize(img, dsize=(self.height, self.width)) 

        img_normalized = cv2.normalize(
            img_resized, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U
        )

        cv2.imwrite(str(out_path), img_normalized)
    
    def process_dir(self):
        for img_path in INPUT_DIR.rglob('*.jpg'):
            rel_path = img_path.relative_to(INPUT_DIR)
            out_path = OUTPUT_DIR / rel_path
            out_path.parent.mkdir(parents=True, exist_ok=True)

            self._process_and_save_image(img_path, out_path)
    

# Ball detection
class Ball_Detection():
    def __init__(self):
        pass

    def sample_felt_color_hsv(self, frame):
        
        h, w = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        cy, cx = h // 2, w // 2
        ph, pw = max(20, h // 10), max(20, w // 10)
        patch = hsv[cy - ph // 2: cy + ph // 2, cx - pw // 2: cx + pw // 2]

        median_hsv = np.median(patch.reshape(-1, 3), axis=0)
        felt_h, felt_s, felt_v = float(median_hsv[0]), float(median_hsv[1]), float(median_hsv[2])

        hue_tol = 18
        lower = np.array([max(0,   felt_h - hue_tol),
                          max(30,  felt_s - 60),
                          max(30,  felt_v - 80)], dtype=np.uint8)
        upper = np.array([min(180, felt_h + hue_tol),
                          255,
                          255], dtype=np.uint8)

        return lower, upper, median_hsv

    def find_table_mask(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower, upper, felt_hsv = self.sample_felt_color_hsv(frame)

        felt_mask = cv2.inRange(hsv, lower, upper)

        k_open  = np.ones((7,  7),  np.uint8)
        k_close = np.ones((21, 21), np.uint8)
        felt_mask = cv2.morphologyEx(felt_mask, cv2.MORPH_OPEN,  k_open)
        felt_mask = cv2.morphologyEx(felt_mask, cv2.MORPH_CLOSE, k_close)

        contours, _ = cv2.findContours(felt_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            mask = np.ones(frame.shape[:2], dtype=np.uint8) * 255
            return mask, felt_hsv

        largest = max(contours, key=cv2.contourArea)

        hull = cv2.convexHull(largest)
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [hull], -1, 255, -1)

        erode_px = max(5, frame.shape[1] // 60)
        k_erode  = np.ones((erode_px, erode_px), np.uint8)
        mask = cv2.erode(mask, k_erode, iterations=1)

        return mask, felt_hsv

    def estimate_ball_radius(self, table_mask):
        contours, _ = cv2.findContours(table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return 8, 28

        _, _, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
        ref = min(w, h)
        nominal_r = ref / 36
        min_r = max(6,  int(nominal_r * 0.55))
        max_r = max(20, int(nominal_r * 1.50))
        return min_r, max_r

    def multi_pass_hough(self,gray_table, table_mask, min_r, max_r):
        min_dist = max(int(min_r * 1.6), 12)
        candidates = []

        def _hough(self, img, dp, p1, p2):
            c = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, dp=dp, minDist=min_dist, param1=p1, param2=p2, minRadius=min_r, maxRadius=max_r)
            return [] if c is None else [tuple(np.round(x).astype(int)) for x in c[0]]

        blur_a = cv2.medianBlur(gray_table, 5)
        candidates += _hough(blur_a, 1.2, 55, 18)

        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        eq = clahe.apply(gray_table)
        blur_b = cv2.GaussianBlur(eq, (5, 5), 1.5)
        candidates += _hough(blur_b, 1.2, 65, 20)

        blur_c = cv2.GaussianBlur(gray_table, (9, 9), 2.5)
        candidates += _hough(blur_c, 1.3, 45, 13)

        return candidates

    def watershed_split_clusters(self, candidates, gray_table, table_mask, min_r, max_r):
        result = []
        oversized = [c for c in candidates if c[2] > max_r * 1.3]
        standard = [c for c in candidates if c[2] <= max_r * 1.3]
        result.extend(standard)

        for (x, y, r) in oversized:
            pad = r + min_r
            x0, y0 = max(0, x-pad), max(0, y-pad)
            x1 = min(gray_table.shape[1], x+pad)
            y1 = min(gray_table.shape[0], y+pad)
            roi = gray_table[y0:y1, x0:x1]
            mroi = table_mask[y0:y1, x0:x1]
            if roi.size == 0: continue

            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
            gradient = cv2.morphologyEx(roi, cv2.MORPH_GRADIENT, kernel)
            _, thresh = cv2.threshold(gradient, 15, 255, cv2.THRESH_BINARY)
            fg = cv2.bitwise_not(thresh)
            fg = cv2.bitwise_and(fg, mroi)
            dist = cv2.distanceTransform(fg, cv2.DIST_L2, 3)
            cv2.normalize(dist, dist, 0, 1.0, cv2.NORM_MINMAX)
            _, peaks = cv2.threshold(dist, 0.4, 1.0, cv2.THRESH_BINARY)
            peaks8 = (peaks * 255).astype(np.uint8)
            contours, _ = cv2.findContours(peaks8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cx_l = int(M["m10"] / M["m00"])
                    cy_l = int(M["m01"] / M["m00"])
                    result.append((int(x0 + cx_l), int(y0 + cy_l), min_r))
        return result

    def circle_iou(self, c1, c2):
        x1, y1, r1 = c1
        x2, y2, r2 = c2
        dist = math.hypot(x1 - x2, y1 - y2)

        if dist >= r1 + r2:
            return 0.0

        if dist <= abs(r1 - r2):
            return 1.0

        r1s, r2s, ds = r1 ** 2, r2 ** 2, dist ** 2
        alpha = math.acos((ds + r1s - r2s) / (2 * dist * r1))
        beta  = math.acos((ds + r2s - r1s) / (2 * dist * r2))
        intersection = r1s * alpha + r2s * beta - r1s * math.sin(2 * alpha) / 2 - r2s * math.sin(2 * beta) / 2
        smaller_area = math.pi * min(r1s, r2s)
        return intersection / smaller_area


    def verify_and_nms(self, candidates, frame, table_mask, felt_hsv, min_r):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        fh, fs, fv = float(felt_hsv[0]), float(felt_hsv[1]), float(felt_hsv[2])
        H, W = frame.shape[:2]

        passed = []
        for (x, y, r) in candidates:
            if not (0 <= y < H and 0 <= x < W): continue
            if table_mask[y, x] == 0:           continue

            probe = np.zeros((H, W), dtype=np.uint8)
            cv2.circle(probe, (x, y), max(1, int(r * 0.7)), 255, -1)
            probe = cv2.bitwise_and(probe, table_mask)
            if cv2.countNonZero(probe) < 8: continue

            mh, ms, mv = cv2.mean(hsv, mask=probe)[:3]
            hd = min(abs(mh-fh), 180-abs(mh-fh))

            if (hd < 14) and (abs(mv-fv) < 35) and (abs(ms-fs) < 40):
                continue 

            passed.append((x, y, r))

        passed.sort(key=lambda c: -c[2]) 

        verified = []
        for c in passed:
            x, y, r = c
            merged = False
            for i, vc in enumerate(verified):
                iou  = self.circle_iou(c, vc)
                dist = math.hypot(x-vc[0], y-vc[1])

                is_dup = (iou > 0.35) or (iou > 0.01 and dist < min_r * 0.8)
                if is_dup:
                    if r > vc[2]:
                        verified[i] = c
                    merged = True
                    break

            if not merged:
                verified.append(c)

        return verified

    def draw_detections(self, frame, balls):
        out = frame.copy()
        h, w = frame.shape[:2]
        for (x, y, r) in balls:
            pad = int(r * 0.30) + 3
            ar  = r + pad
            tl  = (max(0, x - ar), max(0, y - ar))
            br  = (min(w, x + ar), min(h, y + ar))
            cv2.rectangle(out, tl, br, (0, 255, 0), 2)
            cv2.circle(out, (x, y), 2, (0, 0, 255), -1)
        return out

    def detect_billiard_balls(self, frame, debug=False):
        table_mask, felt_hsv = self.find_table_mask(frame)

        min_r, max_r = self.estimate_ball_radius(table_mask)

        gray       = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_table = cv2.bitwise_and(gray, gray, mask=table_mask)

        candidates = self.multi_pass_hough(gray_table, table_mask, min_r, max_r)
        candidates = self.watershed_split_clusters(candidates, gray_table, table_mask, min_r, max_r)

        balls = self.erify_and_nms(candidates, frame, table_mask, felt_hsv, min_r)

        if debug:
            fig, axes = plt.subplots(1, 3, figsize=(20, 6))
            axes[0].imshow(table_mask,  cmap="gray");  axes[0].set_title("Table Mask (convex hull)")
            axes[1].imshow(gray_table,  cmap="gray");  axes[1].set_title("Gray (felt only)")
            axes[2].imshow(cv2.cvtColor(self.draw_detections(frame, balls), cv2.COLOR_BGR2RGB))
            axes[2].set_title(f"Detections ({len(balls)} balls)")
            for ax in axes: ax.axis("off")
            plt.tight_layout()
            plt.show()

        return self.draw_detections(frame, balls), balls

    def run_on_directory(self, image_dir, show = True):
        extensions = {".jpg", ".jpeg", ".png", ".bmp"}
        paths = [
            os.path.join(image_dir, f)
            for f in sorted(os.listdir(image_dir))
            if os.path.splitext(f)[1].lower() in extensions
        ]
        if not paths:
            print(f"No images found in {image_dir}")
            return
        for img_path in paths:
            frame = cv2.imread(img_path)
            if frame is None:
                print(f"  [skip] {img_path}")
                continue
            out, balls = detect_billiard_balls(frame, debug=False)
            print(f"{os.path.basename(img_path):60s}  →  {len(balls):2d} balls")
            if show:
                plt.figure(figsize=(10, 6))
                plt.imshow(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
                plt.title(f"{os.path.basename(img_path)} — {len(balls)} balls")
                plt.axis("off")
                plt.tight_layout()
                plt.show()
    
# Classifying of the balls
class Ball_Classification():

    def __init__(self):
        pass

# Outputing the classifications and positions of the balls for each image
def identify_balls(path_aux):
    img_path = os.path.join(OUTPUT_DIR, path_aux)
    frame = cv2.imread(img_path)
    
    if frame is None:
        print(f"Error: Could not load {img_path}")
        return 0, []
    
    img_height, img_width = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([100, 40, 160])
    upper_blue = np.array([135, 255, 255])
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # 2. Focus only on the table area
    contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0, []
    
    table_contour = max(contours, key=cv2.contourArea)
    table_surface = np.zeros_like(blue_mask)
    cv2.drawContours(table_surface, [cv2.convexHull(table_contour)], -1, 255, -1)

    # 3. Preprocess for Circle Detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_table = cv2.bitwise_and(gray, gray, mask=table_surface)
    gray_blur = cv2.medianBlur(gray_table, 5)

    circles = cv2.HoughCircles(
        gray_blur, 
        cv2.HOUGH_GRADIENT, 
        dp=1.1, 
        minDist=14,       
        param1=80,        # only looking at objects and not textures
        param2=18,        # looking for round shapes
        minRadius=6,      
        maxRadius=25
    )

    balls_data = []
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for i, (x, y, r) in enumerate(circles):
            padding = int(r * 0.2)
            # Normalize coordinates
            balls_data.append({
                "number": i + 1,
                "xmin": float(max(0, x - r - padding)) / img_width,
                "xmax": float(min(img_width, x + r + padding)) / img_width,
                "ymin": float(max(0, y - r - padding)) / img_height,
                "ymax": float(min(img_height, y + r + padding)) / img_height
            })

    return len(balls_data), balls_data


def write_json():
    # Define the directory
    dev_set_path = "development_set"
    all_results = []

    # 1. Cycle over all images in the folder
    for img_name in os.listdir(dev_set_path):
        if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):

            # 2. Get the relative path for the JSON "image_path" field
            relative_path = os.path.join(dev_set_path, img_name)

            # 3. Run the detection logic
            ball_count, balls_list = identify_balls(img_name)

            # 4. Create the dictionary for this specific image
            image_data = {
                "image_path": relative_path,
                "num_balls": ball_count,
                "balls": balls_list
            }

            all_results.append(image_data)

    # 6. Convert the entire list to a JSON string at the very end
    json_output = json.dumps(all_results, indent=4)

    # Save to a file
    with open("output.json", "w") as f:
        f.write(json_output)

# Warping the image to a top-view and saving the image
class Top_View():
    def a():
        pass
