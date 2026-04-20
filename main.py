"""
Billiard Project (part 1) python script

course: Computer Vision (M.IA005/ M.EIC029)
year: 2025/2026
authors: Angelina Moreira, Elske Zwol, Guilherme Gonçalves, Maria Shi

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

import cv2
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import math
import json
import sys

IMAGE_DIR = Path("./development_set")
PROCESSED_DIR = Path("./processed_images")
TOP_VIEW_DIR = Path("./top_view")

HEIGHT = 1920
WIDTH = 1080

# Pre-processing of the images
class Preprocessing():
    def __init__(self, height, width):
        self.height = height
        self.width = width

    def _verify_resolutions(self):
        data = []

        for img_path in IMAGE_DIR.rglob('*.jpg'):
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
    
    def process_and_save_image(self, file_path, out_path):
        img = cv2.imread(str(file_path))

        if img is None:
            print(f"Warning: Could not read {file_path}")
            return

        img_resized = cv2.resize(img, dsize=(self.height, self.width)) 

        img_normalized = cv2.normalize(
            img_resized, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U
        )

        cv2.imwrite(str(out_path), img_normalized)
    

# Ball detection
class Ball_Detection():
    def __init__(self):
        pass

    def _classify_view(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        H, W = frame.shape[:2]

        lower = np.array([85, 40, 60], dtype=np.uint8)
        upper = np.array([135, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)

        k = np.ones((11, 11), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return "front"

        cnt = max(contours, key=cv2.contourArea)
        cnt_area = cv2.contourArea(cnt)
        img_area = H * W

        fill_frac = cnt_area / img_area

        rect = cv2.minAreaRect(cnt)
        rect_area = rect[1][0] * rect[1][1]
        rectangularity = cnt_area / (rect_area + 1e-6)

        angle = rect[2]
        tilt = min(abs(angle), abs(90 - abs(angle)))

        x, y, bw, bh = cv2.boundingRect(cnt)
        center_x = (x + bw / 2) / W
        symmetry = 1.0 - abs(center_x - 0.5) * 2

        if fill_frac > 0.35 and rectangularity > 0.80:
            return "top"

        if rectangularity < 0.65 or tilt > 10 or symmetry < 0.75:
            return "diagonal"

        return "front"


    def _detect_felt_mask(self, frame, view_type = "diagonal"):
        H, W = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lower_broad = np.array([70, 40, 40], dtype=np.uint8)
        upper_broad = np.array([140, 255, 255], dtype=np.uint8)
        broad_mask = cv2.inRange(hsv, lower_broad, upper_broad)

        contours, _ = cv2.findContours(broad_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return np.ones((H, W), dtype=np.uint8) * 255, np.array([100, 150, 100])

        largest_broad = max(contours, key=cv2.contourArea)
        sample_mask = np.zeros((H, W), dtype=np.uint8)
        cv2.drawContours(sample_mask, [largest_broad], -1, 255, -1)

        sample_mask = cv2.erode(sample_mask, np.ones((30, 30), np.uint8))

        table_pixels = hsv[sample_mask > 0]
        if len(table_pixels) == 0:
            return np.ones((H, W), dtype=np.uint8) * 255, np.array([100, 150, 100])

        median_hsv = np.median(table_pixels, axis=0)
        felt_h, felt_s, felt_v = float(median_hsv[0]), float(median_hsv[1]), float(median_hsv[2])

        hue_tol = 15
        lower = np.array([max(0, felt_h - hue_tol), max(50, felt_s - 50), max(50, felt_v - 60)], dtype=np.uint8)
        upper = np.array([min(180, felt_h + hue_tol), 255, 255], dtype=np.uint8)

        felt_mask = cv2.inRange(hsv, lower, upper)

        b, g, r = cv2.split(frame)
        blue_dom = (
            (b.astype(np.int16) > g.astype(np.int16) + 5) &
            (b.astype(np.int16) > r.astype(np.int16) + 5)
        ).astype(np.uint8) * 255
        felt_mask = cv2.bitwise_and(felt_mask, blue_dom)

        k_open = np.ones((7, 7), np.uint8)
        k_close = np.ones((21, 21), np.uint8)
        felt_mask = cv2.morphologyEx(felt_mask, cv2.MORPH_OPEN, k_open)
        felt_mask = cv2.morphologyEx(felt_mask, cv2.MORPH_CLOSE, k_close)

        contours, _ = cv2.findContours(felt_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            mask = np.ones((H, W), dtype=np.uint8) * 255
            return mask, median_hsv

        largest = max(contours, key=cv2.contourArea)

        hull = cv2.convexHull(largest)
        mask = np.zeros((H, W), dtype=np.uint8)
        cv2.drawContours(mask, [hull], -1, 255, -1)

        dilated_felt = cv2.dilate(felt_mask, np.ones((15, 15), np.uint8))
        mask = cv2.bitwise_and(mask, dilated_felt)

        contours2, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours2:
            largest2 = max(contours2, key=cv2.contourArea)
            hull2 = cv2.convexHull(largest2)
            mask = np.zeros((H, W), dtype=np.uint8)
            cv2.drawContours(mask, [hull2], -1, 255, -1)

        if view_type == "top":
            erode_px = max(7, W // 50)
        elif view_type == "front":
            erode_px = max(6, W // 35)
        else:
            erode_px = max(5, W // 60)
        k_erode = np.ones((erode_px, erode_px), np.uint8)
        mask = cv2.erode(mask, k_erode, iterations=1)

        corners = cv2.goodFeaturesToTrack(mask, maxCorners=4, qualityLevel=0.01,
                                          minDistance=30, blockSize=15)
        if corners is not None and len(corners) >= 4:
            pts_corners = corners[:4].reshape(-1, 2).astype(np.float32)
            peri = cv2.arcLength(cv2.convexHull(pts_corners.astype(np.int32)), True)
            if view_type == "top":
                cut_radius = int(peri * 0.01)
            elif view_type == "front":
                cut_radius = int(peri * 0.01)
            else:
                cut_radius = 0 
            for x, y in pts_corners:
                cv2.circle(mask, (int(x), int(y)), cut_radius, 0, -1)

        if view_type == "diagonal":
            hull_contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if hull_contours:
                cnt = max(hull_contours, key=cv2.contourArea)
                peri = cv2.arcLength(cnt, True)

                for eps in np.linspace(0.01, 0.08, 20):
                    approx = cv2.approxPolyDP(cnt, eps * peri, True)
                    if len(approx) == 4:
                        break

                if len(approx) == 4:
                    pts = approx.reshape(-1, 2).astype(np.float32)

                    edges = []
                    for i in range(4):
                        p1 = pts[i]
                        p2 = pts[(i + 1) % 4]
                        length = float(np.linalg.norm(p2 - p1))
                        edges.append((i, length))
                    edges.sort(key=lambda x: -x[1])

                    mid_radius = int(peri * 0.005)
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    center = np.mean(pts, axis=0)

                    for idx, _ in edges[:2]:
                        p1 = pts[idx]
                        p2 = pts[(idx + 1) % 4]

                        edge_vec = p2 - p1
                        edge_len = np.linalg.norm(edge_vec)
                        edge_dir = edge_vec / (edge_len + 1e-6)
                        edge_normal = np.array([-edge_dir[1], edge_dir[0]])

                        if np.dot(edge_normal, center - p1) < 0:
                            edge_normal = -edge_normal

                        best_var = -1
                        best_point = (p1 + p2) / 2

                        for t in np.linspace(0.20, 0.80, 50):
                            pt = p1 + t * edge_vec
                            px, py = int(pt[0]), int(pt[1])

                            r = max(10, int(edge_len * 0.03))
                            y0 = max(0, py - r)
                            y1 = min(H, py + r)
                            x0 = max(0, px - r)
                            x1 = min(W, px + r)

                            if y1 - y0 < 3 or x1 - x0 < 3:
                                continue

                            patch = gray[y0:y1, x0:x1]
                            var = float(np.var(patch))

                            if var > best_var:
                                best_var = var
                                best_point = pt

                        cv2.circle(mask, (int(best_point[0]), int(best_point[1])), mid_radius, 0, -1)

        elif view_type == "front":
            hull_contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if hull_contours:
                cnt = max(hull_contours, key=cv2.contourArea)
                peri = cv2.arcLength(cnt, True)

                for eps in np.linspace(0.01, 0.08, 20):
                    approx = cv2.approxPolyDP(cnt, eps * peri, True)
                    if len(approx) == 4:
                        break

                if len(approx) == 4:
                    pts = approx.reshape(4, 2).astype(np.float32)

                    pts_sorted_y = pts[np.argsort(pts[:, 1])]
                    top_pts = pts_sorted_y[:2]
                    bottom_pts = pts_sorted_y[2:]

                    tl = top_pts[np.argmin(top_pts[:, 0])]
                    tr = top_pts[np.argmax(top_pts[:, 0])]
                    bl = bottom_pts[np.argmin(bottom_pts[:, 0])]
                    br = bottom_pts[np.argmax(bottom_pts[:, 0])]

                    edges_to_check = [(tl, bl), (tr, br)]

                    mid_radius = int(peri * 0.005)
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    center_x = (tl[0] + tr[0] + bl[0] + br[0]) / 4

                    for p1, p2 in edges_to_check:
                        edge_vec = p2 - p1
                        edge_len = np.linalg.norm(edge_vec)

                        best_var = -1
                        best_point = (p1 + p2) / 2

                        for t in np.linspace(0.30, 0.70, 40):
                            pt = p1 + t * edge_vec
                            px, py = int(pt[0]), int(pt[1])

                            r = max(10, int(edge_len * 0.03))
                            y0 = max(0, py - r)
                            y1 = min(H, py + r)
                            x0 = max(0, px - r)
                            x1 = min(W, px + r)

                            if y1 - y0 < 3 or x1 - x0 < 3:
                                continue

                            patch = gray[y0:y1, x0:x1]
                            var = float(np.var(patch))

                            if var > best_var:
                                best_var = var
                                best_point = pt

                        nudge_dir = -1 if best_point[0] < center_x else 1
                        nudge_amount = 5
                        best_point[0] += nudge_dir * nudge_amount

                        cv2.circle(mask, (int(best_point[0]), int(best_point[1])), mid_radius, 0, -1)

        return mask, median_hsv


    def _get_table_mask(self, frame):
        H, W = frame.shape[:2]
        view_type = self._classify_view(frame)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        mask, felt_hsv = self._detect_felt_mask(frame, view_type=view_type)

        table_pixels = hsv[mask > 0]
        if len(table_pixels) > 0:
            felt_hsv = np.median(table_pixels, axis=0)

        return mask, felt_hsv, view_type

    def _estimate_ball_radius(self, table_mask):
        contours, _ = cv2.findContours(table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return 8, 28

        _, _, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
        ref = min(w, h)
        nominal_r = ref / 36
        min_r = max(6,  int(nominal_r * 0.60))
        max_r = max(20, int(nominal_r * 1.50))
        return min_r, max_r

    def _multi_pass_hough(self, gray_table, min_r, max_r):
        min_dist = max(int(min_r * 2.2), 16)
        candidates = []

        base_p2 = int(10 + (min_r * 0.4))

        def _hough(img, dp, p1, p2_offset):
            final_p2 = base_p2 + p2_offset
            c = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, dp=dp, minDist=min_dist, param1=p1, param2=final_p2, minRadius=min_r, maxRadius=max_r)
            return [] if c is None else [tuple(np.round(x).astype(int)) for x in c[0]]

        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        eq = clahe.apply(gray_table)

        blur_a = cv2.medianBlur(gray_table, 5)
        candidates += _hough(blur_a, 1.1, 55, 2)

        blur_b = cv2.GaussianBlur(eq, (5, 5), 1.5)
        candidates += _hough(blur_b, 1.1, 65, 4)

        blur_c = cv2.GaussianBlur(gray_table, (9, 9), 2.5)
        candidates += _hough(blur_c, 1.1, 45, 3)

        blur_d = cv2.medianBlur(gray_table, 5)
        candidates += _hough(blur_d, 1.1, 80, 4)

        k_glare = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        no_glare = cv2.morphologyEx(eq, cv2.MORPH_OPEN, k_glare)
        blur_e = cv2.GaussianBlur(no_glare, (5, 5), 1.5)
        candidates += _hough(blur_e, 1.1, 30, 0)

        return candidates

    def _circle_iou(self, c1, c2):
        x1, y1, r1 = c1[:3]
        x2, y2, r2 = c2[:3]

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


    def _verify_and_nms(self, candidates, frame, table_mask, felt_hsv, min_r, debug= False):
        H, W = table_mask.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        fh, fs, fv = float(felt_hsv[0]), float(felt_hsv[1]), float(felt_hsv[2])

        if debug: print(f"\nTotal candidates: {len(candidates)}")

        on_table = []
        felt_rejected = 0
        off_table = 0
        for (x, y, r) in candidates:
            if not (0 <= y < H and 0 <= x < W): 
                off_table += 1
                continue
            if table_mask[y, x] == 0: 
                off_table += 1
                continue

            probe = np.zeros((H, W), dtype=np.uint8)
            cv2.circle(probe, (x, y), max(1, int(r * 0.7)), 255, -1)
            probe = cv2.bitwise_and(probe, table_mask)
            if cv2.countNonZero(probe) < 8: continue

            mh, ms, mv = cv2.mean(hsv, mask=probe)[:3]
            hd = min(abs(mh - fh), 180 - abs(mh - fh))
            if (hd < 18) and (abs(mv - fv) < 35) and (abs(ms - fs) < 40):
                felt_rejected += 1
                if debug: print(f"  Felt-rejected: [{x},{y}] r={r} hd={hd:.1f} dv={abs(mv-fv):.1f} ds={abs(ms-fs):.1f}")
                continue

            on_table.append((x, y, r))

        if debug: print(f"Off-table: {off_table}, Felt-rejected: {felt_rejected}, On table: {len(on_table)}")

        on_table.sort(key=lambda c: -c[2])

        verified = []
        for c in on_table:
            x, y, r = c
            merged = False
            for i, vc in enumerate(verified):
                dist = math.hypot(x - vc[0], y - vc[1])
                R_large = max(r, vc[2])
                R_small = min(r, vc[2])

                is_dup = False
                if dist < min_r * 1.50:
                    is_dup = True
                elif dist < R_large * 1.50 and R_small < R_large * 0.75:
                    is_dup = True
                elif self._circle_iou(c, vc) > 0.45:
                    is_dup = True
                elif dist < (R_large + R_small) * 0.65:
                    is_dup = True

                if is_dup:
                    if r > vc[2]:
                        verified[i] = c
                    merged = True
                    break

            if not merged:
                verified.append(c)

        if debug: print(f"After NMS: {len(verified)}")

        if len(verified) > 0:
            top_radii = [v[2] for v in verified[:3]]
            avg_r = sum(top_radii) / len(top_radii)

            final = []
            for v in verified:
                if v[2] >= avg_r * 0.55:
                    final.append(v)
                elif debug:
                    print(f"  Purged tiny: [{v[0]}, {v[1]}] r={v[2]} (<55% of avg {avg_r:.0f})")
            verified = final

        if len(verified) > 16:
            if debug: print(f"Truncating {len(verified)} to 16")
            verified = verified[:16]

        if debug: print(f"Final count: {len(verified)} balls\n")

        return verified

    def _draw_detections(self, frame, balls):
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

    def detect_billiard_balls(self, frame, debug = False):
        table_mask, felt_hsv, view_type = self._get_table_mask(frame)
        if debug: print(f"Detected view: {view_type}")

        min_r, max_r = self._estimate_ball_radius(table_mask)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        felt_gray_val = int(cv2.mean(gray, mask=table_mask)[0])
        gray_table = gray.copy()
        gray_table[table_mask == 0] = felt_gray_val

        candidates = self._multi_pass_hough(gray_table, min_r, max_r)

        balls = self._verify_and_nms(candidates, frame, table_mask, felt_hsv, min_r, debug=debug)

        if debug:
            fig, axes = plt.subplots(1, 3, figsize=(20, 6))
            axes[0].imshow(table_mask, cmap="gray"); axes[0].set_title("Table Mask")
            axes[1].imshow(gray_table, cmap="gray"); axes[1].set_title("Gray (felt only)")
            axes[2].imshow(cv2.cvtColor(self._draw_detections(frame, balls), cv2.COLOR_BGR2RGB))
            axes[2].set_title(f"Detections ({len(balls)} balls)")
            for ax in axes: ax.axis("off")
            plt.tight_layout()
            plt.show()

        return self._draw_detections(frame, balls), balls
    
# Classifying of the balls
class Ball_Classification():

    def __init__(self):
        self.colour_anchors = {
                                "yellow":  (17,  178, 235),   # 1 / 9
                                "blue":    (106, 203, 103),   # 2 / 10
                                "red":     (2,   184, 175),   # 3 / 11
                                "red2":    (178, 184, 175),   # wrap-around for red
                                "purple":  (124,  79, 106),   # 4 / 12
                                "orange":  (7,   184, 225),   # 5 / 13
                                "green":   (83,  153,  80),   # 6 / 14
                                "maroon":  (13,  162, 110),   # 7 / 15
                            }
        self.anchor_to_colour = {k: ("red" if k == "red2" else k) for k in self.colour_anchors}
        self.colour_to_number = {
                                 "yellow": 1, 
                                 "blue": 2, 
                                 "red": 3, 
                                 "purple": 4,  
                                 "orange": 5, 
                                 "green": 6, 
                                 "maroon": 7
                                }
        self.colour_hsv_ranges = {
                                    "yellow":  [((17, 150, 140), (25, 255, 255))],
                                    "blue": [((95, 140, 100), (118, 255, 220))],        # V_min 60 -> 100
                                    "red":     [((0, 170, 160),  (5, 255, 230)),        # V_min 120 -> 160
                                                ((173, 170, 160), (179, 255, 230))],
                                    "purple":  [((110, 30, 50),  (145, 200, 200))],
                                    "orange":  [((3, 140, 160),  (13, 255, 255))],
                                    "green":   [((70, 100, 50),  (95, 255, 160))],
                                    "maroon":  [((0, 70, 40),    (16, 220, 170)),       # S_min 90 -> 70
                                            ((170, 70, 40),  (179, 220, 170))],
                                 }
        self.colour_mean_anchors = {
                                        "yellow":  (17,  178, 235),
                                        "blue":    (106, 203, 103),
                                        "red":     (2,   184, 175),
                                        "purple":  (125,  90, 130),
                                        "orange":  (7,   184, 225),
                                        "green":   (83,  153,  80),
                                        "maroon":  (10,  162, 130),
                                    }
        self.white_hsv_range = ((0, 0, 180), (179, 60, 255))
        self.black_hsv_range = ((0, 0, 0),   (179, 255, 60))
        self.white_frac_for_stripe_or_cue = 0.07
        self.white_frac_for_cue = 0.45
        self.black_frac_for_eight = 0.45
        self.cue_max_colour_frac = 0.1
        self.eight_max_colour_frac = 0.15
        self.mean_dist_scale = 40
        self.all_ball_labels = (
                                {"cue", "eight"}
                                | {f"solid_{c}"  for c in self.colour_to_number}
                                | {f"stripe_{c}" for c in self.colour_to_number}
                               )
        
    def label_to_number(self, label):
        if label == "cue":   return 0
        if label == "eight": return 8
        if label.startswith("solid_"):  return self.colour_to_number.get(label[6:], -1)
        if label.startswith("stripe_"): return self.colour_to_number.get(label[7:], 0) + 8
        return -1
    
    def _crop_ball(self, frame, ball, img_w, img_h, shrink=0.85):
        x1 = int(ball["xmin"] * img_w); x2 = int(ball["xmax"] * img_w)
        y1 = int(ball["ymin"] * img_h); y2 = int(ball["ymax"] * img_h)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None, None
        h, w = crop.shape[:2]
        cy, cx = h // 2, w // 2
        r = int(min(h, w) * 0.5 * shrink)
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(mask, (cx, cy), r, 255, -1)
        return crop, mask
    
    def _mask_from_ranges(self, hsv, ranges):
        combined = None
        for lo, hi in ranges:
            m = cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
            combined = m if combined is None else cv2.bitwise_or(combined, m)
        return combined

    def _hue_distance(self, h1, h2):
        """Circular distance on the [0, 179] hue wheel."""
        d = abs(float(h1) - float(h2))
        return min(d, 180.0 - d)

    def _mean_distance_score(self, mean_h, mean_s, mean_v, anchor):
        ah, as_, av = anchor
        d = self._hue_distance(mean_h, ah) \
            + 0.4 * abs(float(mean_s) - float(as_)) \
            + 0.4 * abs(float(mean_v) - float(av))
        return 1.0 / (1.0 + d / self.mean_dist_scale)

    def _classify_ranked(self, frame, ball):
        img_h, img_w = frame.shape[:2]
        crop, mask = self._crop_ball(frame, ball, img_w, img_h, shrink=0.85)
        if crop is None:
            return [("unknown", 0.0)], 0.0, 0.0

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        total = max(int((mask > 0).sum()), 1)

        white_mask = cv2.bitwise_and(self._mask_from_ranges(hsv, [self.white_hsv_range]), mask)
        white_frac = (white_mask > 0).sum() / total

        ys = np.where(white_mask.any(axis=1))[0]
        white_vertical_span = (ys.max() - ys.min() + 1) / white_mask.shape[0] if len(ys) else 0.0

        xs = np.where(white_mask.any(axis=0))[0]
        white_horizontal_span = (xs.max() - xs.min() + 1) / white_mask.shape[1] if len(xs) else 0.0

        has_white_band = (
            white_frac > self.white_frac_for_stripe_or_cue
            or (white_frac > 0.02 and white_horizontal_span > 0.5)
            or (white_frac > 0.03 and white_vertical_span > 0.35)
        )

        black_mask = cv2.bitwise_and(self._mask_from_ranges(hsv, [self.black_hsv_range]), mask)
        black_frac = (black_mask > 0).sum() / total

        if white_frac > self.white_frac_for_cue:
            return [("cue", float(white_frac))], white_frac, black_frac
        if black_frac > self.black_frac_for_eight and white_frac < self.white_frac_for_stripe_or_cue:
            return [("eight", float(black_frac))], white_frac, black_frac

        prefix = "stripe_" if has_white_band else "solid_"

        color_region = cv2.bitwise_and(mask, cv2.bitwise_not(cv2.bitwise_or(white_mask, black_mask)))
        color_pixels = hsv[color_region > 0]
        if len(color_pixels) > 0:
            color_pixels = color_pixels[color_pixels[:, 1] >= 80]

        if len(color_pixels) == 0:
            color_pixels = hsv[mask > 0]

        mean_h = float(np.mean(color_pixels[:, 0]))
        mean_s = float(np.mean(color_pixels[:, 1]))
        mean_v = float(np.mean(color_pixels[:, 2]))

        combined_scores = {}
        for color, ranges in self.colour_hsv_ranges.items():
            color_mask = cv2.bitwise_and(self._mask_from_ranges(hsv, ranges), mask)
            mask_frac = (color_mask > 0).sum() / total

            dist_score = self._mean_distance_score(mean_h, mean_s, mean_v, self.colour_mean_anchors[color])

            combined_scores[color] = (mask_frac + 0.005) * dist_score

        ranked = sorted(
            ((prefix + color, float(score)) for color, score in combined_scores.items()),
            key=lambda x: -x[1],
        )

        max_mask_frac = max((cv2.bitwise_and(self._mask_from_ranges(hsv, rs), mask) > 0).sum() / total
                            for rs in self.colour_hsv_ranges.values())

        if max_mask_frac < self.cue_max_colour_frac:
            ranked.append(("cue", float(white_frac)))

        if max_mask_frac < self.eight_max_colour_frac and not has_white_band:
            ranked.append(("eight", float(black_frac)))

        return ranked, white_frac, black_frac
    
    def _resolve_unique_labels(self, ranked_per_ball, white_fracs, black_fracs):
        available = set(self.all_ball_labels)
        n = len(ranked_per_ball)
        result = [None] * n
        unassigned = set(range(n))

        if n > 0:
            cue_idx = max(range(n), key=lambda i: white_fracs[i])
            result[cue_idx] = "cue"
            available.discard("cue")
            unassigned.discard(cue_idx)

            remaining = [i for i in range(n) if i != cue_idx]
            if remaining:
                eight_idx = max(remaining, key=lambda i: black_fracs[i])
                result[eight_idx] = "eight"
                available.discard("eight")
                unassigned.discard(eight_idx)

        def top_label(i):
            for lbl, _ in ranked_per_ball[i]:
                if lbl not in ("cue", "eight", "unknown"):
                    return lbl
            return None

        solid_balls  = {i for i in unassigned
                        if top_label(i) and top_label(i).startswith("solid_")}
        stripe_balls = {i for i in unassigned
                        if top_label(i) and top_label(i).startswith("stripe_")}
        leftover = unassigned - solid_balls - stripe_balls
        stripe_balls |= leftover

        def assign_group(group):
            thresholds = [0.15, 0.08, 0.04, 0.02, 0.005, 0.0]
            for threshold in thresholds:
                if not group or not available:
                    break
                while group and available:
                    best_per_ball = []
                    for i in group:
                        for label, score in ranked_per_ball[i]:
                            if label in ("cue", "eight", "unknown"):
                                continue
                            if label in available:
                                best_per_ball.append((i, label, score))
                                break
                    best_per_ball = [x for x in best_per_ball if x[2] >= threshold]
                    if not best_per_ball:
                        break
                    best_per_ball.sort(key=lambda x: -x[2])
                    i, label, _ = best_per_ball[0]
                    result[i] = label
                    available.discard(label)
                    group.discard(i)

        assign_group(solid_balls)

        assign_group(stripe_balls)

        for i in range(n):
            if result[i] is None:
                result[i] = "unknown"

        return result

    def classify_image(self, frame, balls):
        ranked_all = []
        white_fracs = []
        black_fracs = []
        for b in balls:
            ranked, wf, bf = self._classify_ranked(frame, b)
            ranked_all.append(ranked)
            white_fracs.append(wf)
            black_fracs.append(bf)
        return self._resolve_unique_labels(ranked_all, white_fracs, black_fracs)
    
    def _draw_labels(self, frame, balls, labels):
        img_h, img_w = frame.shape[:2]
        vis = frame.copy()
        for b, label in zip(balls, labels):
            x1 = int(b["xmin"] * img_w); x2 = int(b["xmax"] * img_w)
            y1 = int(b["ymin"] * img_h); y2 = int(b["ymax"] * img_h)
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(vis, str(self.label_to_number(label)), (x1, max(0, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
        return cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
    
    def visualize_classification(self, frame, balls, labels):
        plt.figure(figsize=(12, 8))
        plt.imshow(self._draw_labels(frame, balls, labels))
        plt.title("Classification result")
        plt.axis("off")
        plt.tight_layout()
        plt.show()

class Top_View():
    def __init__(self):
        pass

    def _order_points_clockwise(self, pts):
        pts = np.asarray(pts, dtype=np.float32)
        center = np.mean(pts, axis=0)

        angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
        pts = pts[np.argsort(angles)]

        area2 = 0.0
        for i in range(len(pts)):
            x1, y1 = pts[i]
            x2, y2 = pts[(i + 1) % len(pts)]
            area2 += x1 * y2 - x2 * y1

        if area2 > 0:
            pts = pts[::-1]

        start_idx = np.argmin(pts[:, 0] + pts[:, 1])
        pts = np.roll(pts, -start_idx, axis=0)
        return pts.astype(np.float32)

    def _polygon_area(self, pts):
        pts = np.asarray(pts, dtype=np.float32)
        x = pts[:, 0]
        y = pts[:, 1]
        return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

    def _line_from_two_points(self, p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        a = y1 - y2
        b = x2 - x1
        c = x1 * y2 - x2 * y1
        return np.array([a, b, c], dtype=np.float32)

    def _intersect_lines(self, line1, line2):
        a1, b1, c1 = line1
        a2, b2, c2 = line2
        det = a1 * b2 - a2 * b1
        if abs(det) < 1e-8:
            return None
        x = (b1 * c2 - b2 * c1) / det
        y = (c1 * a2 - c2 * a1) / det
        return np.array([x, y], dtype=np.float32)

    def _clip_point_to_image(self, pt, w, h):
        x = float(np.clip(pt[0], 0, w - 1))
        y = float(np.clip(pt[1], 0, h - 1))
        return np.array([x, y], dtype=np.float32)

    def _normalize(self, v):
        v = np.asarray(v, dtype=np.float32)
        n = np.linalg.norm(v) + 1e-8
        return v / n

    def _point_to_line_distance(self, line, points):
        a, b, c = line
        points = np.asarray(points, dtype=np.float32)
        denom = np.sqrt(a * a + b * b) + 1e-8
        return np.abs(a * points[:, 0] + b * points[:, 1] + c) / denom

    def _point_to_segment_distance(self, points, a, b):
        points = np.asarray(points, dtype=np.float32)
        a = np.asarray(a, dtype=np.float32)
        b = np.asarray(b, dtype=np.float32)

        ab = b - a
        ab2 = np.dot(ab, ab) + 1e-8
        ap = points - a[None, :]
        t = np.sum(ap * ab[None, :], axis=1) / ab2
        t = np.clip(t, 0.0, 1.0)
        proj = a[None, :] + t[:, None] * ab[None, :]
        return np.linalg.norm(points - proj, axis=1)

    def _line_to_border_points(self, line, w, h):
        a, b, c = line
        pts = []

        if abs(b) > 1e-8:
            y = -(a * 0 + c) / b
            if 0 <= y < h:
                pts.append(np.array([0, y], dtype=np.float32))
            y = -(a * (w - 1) + c) / b
            if 0 <= y < h:
                pts.append(np.array([w - 1, y], dtype=np.float32))

        if abs(a) > 1e-8:
            x = -(b * 0 + c) / a
            if 0 <= x < w:
                pts.append(np.array([x, 0], dtype=np.float32))
            x = -(b * (h - 1) + c) / a
            if 0 <= x < w:
                pts.append(np.array([x, h - 1], dtype=np.float32))

        if len(pts) < 2:
            return None

        best_pair = None
        best_d = -1
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                d = np.linalg.norm(pts[i] - pts[j])
                if d > best_d:
                    best_d = d
                    best_pair = (pts[i], pts[j])
        return best_pair

    def _fit_line_to_points_huber(self, points):
        points = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
        vx, vy, x0, y0 = cv2.fitLine(points, cv2.DIST_HUBER, 0, 0.01, 0.01).flatten()

        p1 = np.array([x0 - 3000 * vx, y0 - 3000 * vy], dtype=np.float32)
        p2 = np.array([x0 + 3000 * vx, y0 + 3000 * vy], dtype=np.float32)
        return self._line_from_two_points(p1, p2)

    def _robust_fit_line_iterative(self, points, max_iters=8, min_points=20):
        pts = np.asarray(points, dtype=np.float32)
        if len(pts) < min_points:
            raise ValueError(f"Not enough points for robust line fitting: {len(pts)}")

        current = pts.copy()

        for _ in range(max_iters):
            if len(current) < min_points:
                break

            line = self._fit_line_to_points_huber(current)
            d = self._point_to_line_distance(line, current)

            med = np.median(d)
            mad = np.median(np.abs(d - med)) + 1e-6
            thresh = max(1.5, med + 2.5 * 1.4826 * mad)

            inliers = d <= thresh
            if np.all(inliers):
                break
            if np.count_nonzero(inliers) < min_points:
                break

            current = current[inliers]

        line = self._fit_line_to_points_huber(current)
        return line, current

    def _segment_table_blue(self, image_bgr):
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

        lower_blue = np.array([85, 70, 70], dtype=np.uint8)
        upper_blue = np.array([125, 255, 255], dtype=np.uint8)

        mask_hsv = cv2.inRange(hsv, lower_blue, upper_blue)

        b, g, r = cv2.split(image_bgr)
        blue_dom = (
            (b.astype(np.int16) > g.astype(np.int16) + 10) &
            (b.astype(np.int16) > r.astype(np.int16) + 10)
        ).astype(np.uint8) * 255

        return cv2.bitwise_and(mask_hsv, blue_dom)

    def _apply_spatial_prior(self, mask):
        h, w = mask.shape
        roi_mask = np.zeros_like(mask)
        y0 = int(0.17 * h)
        x0 = int(0.02 * w)
        x1 = int(0.98 * w)
        roi_mask[y0:h, x0:x1] = 255
        return cv2.bitwise_and(mask, roi_mask)

    def _component_score(self, stats_row, image_area):
        area = stats_row[cv2.CC_STAT_AREA]
        bw = stats_row[cv2.CC_STAT_WIDTH]
        bh = stats_row[cv2.CC_STAT_HEIGHT]

        if bw <= 0 or bh <= 0:
            return -1e9

        bbox_area = bw * bh
        extent = area / (bbox_area + 1e-8)
        aspect = bw / (bh + 1e-8)

        if area < 0.01 * image_area:
            return -1e9
        if not (0.8 <= aspect <= 5.5):
            return -1e9
        if not (0.25 <= extent <= 0.98):
            return -1e9

        return 2.0 * area + 50000.0 * extent

    def _suppress_top_text_components(self, mask, top_frac=0.28, max_area_frac=0.003, min_width_frac=0.25):
        h, w = mask.shape
        image_area = h * w

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        cleaned = np.zeros_like(mask)

        for label in range(1, num_labels):
            area = stats[label, cv2.CC_STAT_AREA]
            bw = stats[label, cv2.CC_STAT_WIDTH]
            _, cy = centroids[label]

            keep = True
            if cy < top_frac * h:
                if area < max_area_frac * image_area and bw < min_width_frac * w:
                    keep = False

            if keep:
                cleaned[labels == label] = 255

        return cleaned

    def _find_best_table_component(self, mask):
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

        if num_labels <= 1:
            raise ValueError("No connected components found.")

        h, w = mask.shape
        image_area = h * w

        best_label = None
        best_score = -1e18

        for label in range(1, num_labels):
            score = self._component_score(stats[label], image_area)
            if score > best_score:
                best_score = score
                best_label = label

        if best_label is None:
            raise ValueError("No valid blue-cloth component found.")

        component_mask = np.zeros_like(mask)
        component_mask[labels == best_label] = 255
        return component_mask

    def _fill_table_component(self, mask_table):
        contours, _ = cv2.findContours(mask_table, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            raise ValueError("No contour found in selected table component.")

        contour = max(contours, key=cv2.contourArea)

        filled = np.zeros_like(mask_table)
        cv2.drawContours(filled, [contour], -1, 255, -1)

        filled = cv2.morphologyEx(
            filled, cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
        )

        return filled

    def _get_largest_contour(self, mask):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            raise ValueError("No contour found.")
        return max(contours, key=cv2.contourArea)

    def _build_hull_mask(self, mask):
        contour = self._get_largest_contour(mask)
        hull = cv2.convexHull(contour, returnPoints=True)

        hull_mask = np.zeros_like(mask)
        cv2.drawContours(hull_mask, [hull], -1, 255, -1)
        return contour, hull, hull_mask

    def _detect_initial_corners_from_contour(self, contour):
        perimeter = cv2.arcLength(contour, True)
        contour_area = cv2.contourArea(contour)

        best_pts = None
        best_score = -np.inf

        for factor in np.linspace(0.002, 0.08, 100):
            epsilon = factor * perimeter
            approx = cv2.approxPolyDP(contour, epsilon, True)

            if len(approx) != 4:
                continue

            pts = approx.reshape(4, 2).astype(np.float32)
            pts = self._order_points_clockwise(pts)

            if not cv2.isContourConvex(np.round(pts).astype(np.int32)):
                continue

            quad_area = self._polygon_area(pts)
            if quad_area < 0.70 * contour_area:
                continue

            edges = np.array([
                np.linalg.norm(pts[1] - pts[0]),
                np.linalg.norm(pts[2] - pts[1]),
                np.linalg.norm(pts[3] - pts[2]),
                np.linalg.norm(pts[0] - pts[3]),
            ], dtype=np.float32)

            opp_similarity = (
                min(edges[0], edges[2]) / (max(edges[0], edges[2]) + 1e-6) +
                min(edges[1], edges[3]) / (max(edges[1], edges[3]) + 1e-6)
            )

            bbox_w = float(np.max(pts[:, 0]) - np.min(pts[:, 0]))
            bbox_h = float(np.max(pts[:, 1]) - np.min(pts[:, 1]))
            if bbox_w < 1 or bbox_h < 1:
                continue

            bbox_fill = quad_area / (bbox_w * bbox_h + 1e-6)
            score = (
                2.0 * (quad_area / (contour_area + 1e-6)) +
                1.5 * opp_similarity +
                0.8 * bbox_fill
            )

            if score > best_score:
                best_score = score
                best_pts = pts

        if best_pts is not None:
            return best_pts

        rect = cv2.minAreaRect(contour)
        pts = cv2.boxPoints(rect).astype(np.float32)
        return self._order_points_clockwise(pts)

    def _split_points_by_quad_sides(self, points, quad):
        quad = self._order_points_clockwise(quad)
        pts = np.asarray(points, dtype=np.float32)

        sides = [
            (quad[0], quad[1]),
            (quad[1], quad[2]),
            (quad[2], quad[3]),
            (quad[3], quad[0]),
        ]

        dist_stack = []
        for a, b in sides:
            dist_stack.append(self._point_to_segment_distance(pts, a, b))

        dist_stack = np.stack(dist_stack, axis=1)
        labels = np.argmin(dist_stack, axis=1)
        side_points = [pts[labels == i] for i in range(4)]
        return side_points, labels

    def _trim_points_along_side(self, points, a, b, keep_ratio=0.70):
        pts = np.asarray(points, dtype=np.float32)
        if len(pts) < 20:
            return pts

        d = self._normalize(b - a)
        t = pts @ d
        order = np.argsort(t)
        pts_sorted = pts[order]

        n = len(pts_sorted)
        margin = int(round((1.0 - keep_ratio) * 0.5 * n))
        if 2 * margin >= n:
            return pts_sorted
        return pts_sorted[margin:n - margin]

    def _fit_side_lines_from_dense_hull_contour(self, hull_contour, rough_quad):
        pts = hull_contour.reshape(-1, 2).astype(np.float32)
        rough_quad = self._order_points_clockwise(rough_quad)

        side_points, _ = self._split_points_by_quad_sides(pts, rough_quad)

        sides = [
            (rough_quad[0], rough_quad[1]),
            (rough_quad[1], rough_quad[2]),
            (rough_quad[2], rough_quad[3]),
            (rough_quad[3], rough_quad[0]),
        ]

        fitted_lines = []
        trimmed_sets = []
        inlier_sets = []

        for i, side_pts in enumerate(side_points):
            if len(side_pts) < 20:
                raise ValueError(f"Too few dense hull-contour points for side {i}: {len(side_pts)}")

            a, b = sides[i]
            pts_trimmed = self._trim_points_along_side(side_pts, a, b, keep_ratio=0.70)
            if len(pts_trimmed) < 20:
                pts_trimmed = side_pts

            line, inliers = self._robust_fit_line_iterative(
                pts_trimmed,
                max_iters=8,
                min_points=20
            )

            fitted_lines.append(line)
            trimmed_sets.append(pts_trimmed)
            inlier_sets.append(inliers)

        debug = {
            "side_points": side_points,
            "trimmed_sets": trimmed_sets,
            "inlier_sets": inlier_sets,
            "fitted_lines": fitted_lines,
        }
        return fitted_lines, debug

    def _refine_corners_from_lines(self, fitted_lines, image_shape):
        h, w = image_shape[:2]

        top, right, bottom, left = fitted_lines

        tl = self._intersect_lines(left, top)
        tr = self._intersect_lines(top, right)
        br = self._intersect_lines(right, bottom)
        bl = self._intersect_lines(bottom, left)

        refined = [tl, tr, br, bl]
        if any(p is None for p in refined):
            raise ValueError("Failed to intersect fitted side lines.")

        refined = np.array([self._clip_point_to_image(p, w, h) for p in refined], dtype=np.float32)
        refined = self._order_points_clockwise(refined)
        return refined

    def _quad_is_reasonable(self, quad, image_shape):
        quad = self._order_points_clockwise(quad)
        h, w = image_shape[:2]
        img_area = h * w
        area = self._polygon_area(quad)

        if area < 0.02 * img_area:
            return False

        edges = np.array([
            np.linalg.norm(quad[1] - quad[0]),
            np.linalg.norm(quad[2] - quad[1]),
            np.linalg.norm(quad[3] - quad[2]),
            np.linalg.norm(quad[0] - quad[3]),
        ], dtype=np.float32)

        return np.min(edges) >= 20

    def _warp_from_corners(self, image_bgr, corners, out_w=720, out_h=400, assume_ordered=False):
        if not assume_ordered:
            corners = self._order_points_clockwise(corners)
        else:
            corners = np.asarray(corners, dtype=np.float32)

        dst = np.array([
            [0, 0],
            [out_w - 1, 0],
            [out_w - 1, out_h - 1],
            [0, out_h - 1]
        ], dtype=np.float32)

        H = cv2.getPerspectiveTransform(corners.astype(np.float32), dst)
        warped = cv2.warpPerspective(image_bgr, H, (out_w, out_h))
        return warped, H

    def _warp_mask_from_corners(self, mask, corners, out_w=720, out_h=400, assume_ordered=False):
        if not assume_ordered:
            corners = self._order_points_clockwise(corners)
        else:
            corners = np.asarray(corners, dtype=np.float32)

        dst = np.array([
            [0, 0],
            [out_w - 1, 0],
            [out_w - 1, out_h - 1],
            [0, out_h - 1]
        ], dtype=np.float32)

        H = cv2.getPerspectiveTransform(corners.astype(np.float32), dst)
        warped = cv2.warpPerspective(mask, H, (out_w, out_h), flags=cv2.INTER_NEAREST)
        return warped, H

    def _warp_from_corners_auto(self, image_bgr, corners, assume_ordered=False):
        if not assume_ordered:
            corners = self._order_points_clockwise(corners)
        else:
            corners = np.asarray(corners, dtype=np.float32)

        w_top = np.linalg.norm(corners[1] - corners[0])
        w_bottom = np.linalg.norm(corners[2] - corners[3])
        h_left = np.linalg.norm(corners[3] - corners[0])
        h_right = np.linalg.norm(corners[2] - corners[1])

        out_w = int(round(max(w_top, w_bottom)))
        out_h = int(round(max(h_left, h_right)))
        out_w = max(out_w, 200)
        out_h = max(out_h, 120)

        return self._warp_from_corners(
            image_bgr,
            corners,
            out_w=out_w,
            out_h=out_h,
            assume_ordered=True
        )

    def _build_inner_side_strips(self, warped_table_mask, strip_frac=0.03):
        table = (warped_table_mask > 0).astype(np.uint8) * 255
        h, w = table.shape

        d = max(4, int(round(strip_frac * min(h, w))))

        top = np.zeros_like(table)
        bottom = np.zeros_like(table)
        left = np.zeros_like(table)
        right = np.zeros_like(table)

        top[:d, :] = 255
        bottom[h - d:, :] = 255
        left[:, :d] = 255
        right[:, w - d:] = 255

        return {
            "top": cv2.bitwise_and(table, top),
            "bottom": cv2.bitwise_and(table, bottom),
            "left": cv2.bitwise_and(table, left),
            "right": cv2.bitwise_and(table, right),
        }

    def _detect_middle_pocket_scores_in_warp(self, warped_bgr, warped_table_mask, dark_thresh=70):
        gray = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        dark = (blur < dark_thresh).astype(np.uint8) * 255
        dark = cv2.morphologyEx(
            dark, cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        )
        dark = cv2.morphologyEx(
            dark, cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        )

        strips = self._build_inner_side_strips(warped_table_mask, strip_frac=0.03)

        zones = {
            "top": (
                slice(0, int(0.045 * h)),
                slice(int(0.47 * w), int(0.53 * w))
            ),
            "bottom": (
                slice(int(0.955 * h), h),
                slice(int(0.47 * w), int(0.53 * w))
            ),
            "left": (
                slice(int(0.47 * h), int(0.53 * h)),
                slice(0, int(0.045 * w))
            ),
            "right": (
                slice(int(0.47 * h), int(0.53 * h)),
                slice(int(0.955 * w), w)
            ),
        }

        strengths = {"top": 0.0, "bottom": 0.0, "left": 0.0, "right": 0.0}
        accepted = []
        rejected = []

        candidate_vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        zone_vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        strip_vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        side_colors = {
            "top": (0, 255, 255),
            "bottom": (255, 255, 0),
            "left": (255, 0, 255),
            "right": (0, 255, 0),
        }

        for side, strip_mask in strips.items():
            color = side_colors[side]
            strip_vis[strip_mask > 0] = (
                0.6 * strip_vis[strip_mask > 0] + 0.4 * np.array(color)
            ).astype(np.uint8)

        img_area = h * w
        min_area = max(30, int(0.00008 * img_area))
        max_area = max(120, int(0.010 * img_area))

        for side, (ys, xs) in zones.items():
            cv2.rectangle(zone_vis, (xs.start, ys.start), (xs.stop - 1, ys.stop - 1), (255, 255, 255), 1)

            zone_mask = np.zeros_like(dark)
            zone_mask[ys, xs] = 255

            candidate_mask = cv2.bitwise_and(dark, zone_mask)

            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(candidate_mask, connectivity=8)

            for lab in range(1, num_labels):
                area = stats[lab, cv2.CC_STAT_AREA]

                comp = np.zeros_like(dark)
                comp[labels == lab] = 255

                contours, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if not contours:
                    continue
                cnt = max(contours, key=cv2.contourArea)

                x, y, bw, bh = cv2.boundingRect(cnt)

                M = cv2.moments(cnt)
                if M["m00"] <= 1e-6:
                    continue
                cx = M["m10"] / M["m00"]
                cy = M["m01"] / M["m00"]

                reason = None

                if area < min_area or area > max_area:
                    reason = "area"
                else:
                    touch = np.count_nonzero((comp > 0) & (strips[side] > 0))
                    min_touch = 10
                    if touch < min_touch:
                        reason = "low_strip_touch"
                    else:
                        if side in ("top", "bottom"):
                            middle_offset = abs(cx - 0.5 * w)
                            max_middle_offset = 0.05 * w
                            span = bw
                            min_span = 0.035 * w
                            depth = cy if side == "top" else (h - 1 - cy)
                            max_depth = 0.022 * h
                        else:
                            middle_offset = abs(cy - 0.5 * h)
                            max_middle_offset = 0.05 * h
                            span = bh
                            min_span = 0.035 * h
                            depth = cx if side == "left" else (w - 1 - cx)
                            max_depth = 0.022 * w

                        if middle_offset > max_middle_offset:
                            reason = "off_center"
                        elif span < min_span:
                            reason = "too_small_span"
                        elif depth > max_depth:
                            reason = "too_deep"
                        else:
                            center_score = max(0.0, 1.0 - middle_offset / (max_middle_offset + 1e-6))
                            span_score = min(1.0, span / (0.06 * (w if side in ("top", "bottom") else h) + 1e-6))
                            depth_score = max(0.0, 1.0 - depth / (max_depth + 1e-6))
                            touch_score = min(1.0, touch / 25.0)

                            score = (
                                2.5 * center_score +
                                1.5 * span_score +
                                1.5 * depth_score +
                                2.0 * touch_score
                            )

                            strengths[side] += float(score)
                            accepted.append({
                                "side": side,
                                "area": area,
                                "bbox": (x, y, bw, bh),
                                "centroid": (cx, cy),
                                "touch": int(touch),
                                "middle_offset": float(middle_offset),
                                "span": float(span),
                                "depth": float(depth),
                                "score": float(score),
                            })

                            color = side_colors[side]
                            cv2.drawContours(candidate_vis, [cnt], -1, color, 2)
                            cv2.circle(candidate_vis, (int(round(cx)), int(round(cy))), 4, color, -1)
                            cv2.putText(
                                candidate_vis,
                                f"{side[0].upper()}:{score:.1f}",
                                (int(round(cx)) + 4, int(round(cy)) - 4),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.4,
                                color,
                                1,
                                cv2.LINE_AA
                            )

                if reason is not None:
                    rejected.append({
                        "side": side,
                        "bbox": (x, y, bw, bh),
                        "centroid": (cx, cy),
                        "reason": reason,
                    })
                    cv2.drawContours(candidate_vis, [cnt], -1, (255, 0, 0), 1)

        debug = {
            "gray": gray,
            "dark_mask": dark,
            "zone_vis": zone_vis,
            "strip_vis": strip_vis,
            "candidate_vis": candidate_vis,
            "accepted": accepted,
            "rejected": rejected,
            "strengths": strengths,
            "strips": strips,
        }
        return strengths, debug

    def _detect_middle_pocket_strengths_from_mask_notches(self, warped_table_mask):
        table = (warped_table_mask > 0).astype(np.uint8)
        h, w = table.shape

        strengths = {"top": 0.0, "bottom": 0.0, "left": 0.0, "right": 0.0}

        x0, x1 = int(0.35 * w), int(0.65 * w)
        y0, y1 = int(0.35 * h), int(0.65 * h)

        top_profile = []
        for x in range(x0, x1):
            ys = np.where(table[:, x] > 0)[0]
            top_profile.append(h if len(ys) == 0 else int(ys[0]))
        top_profile = np.array(top_profile, dtype=np.float32)
        top_baseline = np.percentile(top_profile, 15)
        top_notch = np.maximum(0.0, top_profile - top_baseline)
        strengths["top"] = float(np.sum(top_notch))

        bottom_profile = []
        for x in range(x0, x1):
            ys = np.where(table[:, x] > 0)[0]
            bottom_profile.append(-1 if len(ys) == 0 else int(ys[-1]))
        bottom_profile = np.array(bottom_profile, dtype=np.float32)
        bottom_baseline = np.percentile(bottom_profile, 85)
        bottom_notch = np.maximum(0.0, bottom_baseline - bottom_profile)
        strengths["bottom"] = float(np.sum(bottom_notch))

        left_profile = []
        for y in range(y0, y1):
            xs = np.where(table[y, :] > 0)[0]
            left_profile.append(w if len(xs) == 0 else int(xs[0]))
        left_profile = np.array(left_profile, dtype=np.float32)
        left_baseline = np.percentile(left_profile, 15)
        left_notch = np.maximum(0.0, left_profile - left_baseline)
        strengths["left"] = float(np.sum(left_notch))

        right_profile = []
        for y in range(y0, y1):
            xs = np.where(table[y, :] > 0)[0]
            right_profile.append(-1 if len(xs) == 0 else int(xs[-1]))
        right_profile = np.array(right_profile, dtype=np.float32)
        right_baseline = np.percentile(right_profile, 85)
        right_notch = np.maximum(0.0, right_baseline - right_profile)
        strengths["right"] = float(np.sum(right_notch))

        debug = {
            "top_profile": top_profile,
            "bottom_profile": bottom_profile,
            "left_profile": left_profile,
            "right_profile": right_profile,
            "top_notch": top_notch,
            "bottom_notch": bottom_notch,
            "left_notch": left_notch,
            "right_notch": right_notch,
        }
        return strengths, debug

    def _visualize_mask_notches(self, warped_table_mask, notch_debug):
        table = (warped_table_mask > 0).astype(np.uint8) * 255
        vis = cv2.cvtColor(table, cv2.COLOR_GRAY2BGR)
        h, w = table.shape

        cv2.rectangle(vis, (int(0.47 * w), 0), (int(0.53 * w), int(0.045 * h)), (0, 255, 255), 1)
        cv2.rectangle(vis, (int(0.47 * w), int(0.955 * h)), (int(0.53 * w), h - 1), (255, 255, 0), 1)
        cv2.rectangle(vis, (0, int(0.47 * h)), (int(0.045 * w), int(0.53 * h)), (255, 0, 255), 1)
        cv2.rectangle(vis, (int(0.955 * w), int(0.47 * h)), (w - 1, int(0.53 * h)), (0, 255, 0), 1)

        if len(notch_debug["top_notch"]) > 0:
            top_idx = int(np.argmax(notch_debug["top_notch"]))
            x_top = int(0.35 * w) + top_idx
            cv2.circle(vis, (x_top, int(notch_debug["top_profile"][top_idx])), 4, (0, 255, 255), -1)

        if len(notch_debug["bottom_notch"]) > 0:
            bottom_idx = int(np.argmax(notch_debug["bottom_notch"]))
            x_bottom = int(0.35 * w) + bottom_idx
            cv2.circle(vis, (x_bottom, int(notch_debug["bottom_profile"][bottom_idx])), 4, (255, 255, 0), -1)

        if len(notch_debug["left_notch"]) > 0:
            left_idx = int(np.argmax(notch_debug["left_notch"]))
            y_left = int(0.35 * h) + left_idx
            cv2.circle(vis, (int(notch_debug["left_profile"][left_idx]), y_left), 4, (255, 0, 255), -1)

        if len(notch_debug["right_notch"]) > 0:
            right_idx = int(np.argmax(notch_debug["right_notch"]))
            y_right = int(0.35 * h) + right_idx
            cv2.circle(vis, (int(notch_debug["right_profile"][right_idx]), y_right), 4, (0, 255, 0), -1)

        return vis

    def _decide_corner_rotation_using_middle_pockets(self,image_bgr,raw_mask_table,corners,tmp_w=720,tmp_h=400,notch_weight=0.25,geom_weight=2.0):
        canonical = self._order_points_clockwise(corners)

        def side_lengths(c):
            w_top = np.linalg.norm(c[1] - c[0])
            w_bottom = np.linalg.norm(c[2] - c[3])
            h_left = np.linalg.norm(c[3] - c[0])
            h_right = np.linalg.norm(c[2] - c[1])
            width_est = 0.5 * (w_top + w_bottom)
            height_est = 0.5 * (h_left + h_right)
            return width_est, height_est

        def evaluate(c):
            warped_img, _ = self._warp_from_corners(
                image_bgr, c, out_w=tmp_w, out_h=tmp_h, assume_ordered=True
            )
            warped_mask, _ = self._warp_mask_from_corners(
                raw_mask_table, c, out_w=tmp_w, out_h=tmp_h, assume_ordered=True
            )

            rgb_strengths, rgb_debug = self._detect_middle_pocket_scores_in_warp(
                warped_img, warped_mask, dark_thresh=70
            )
            notch_strengths, notch_debug = self._detect_middle_pocket_strengths_from_mask_notches(
                warped_mask
            )
            notch_vis = self._visualize_mask_notches(warped_mask, notch_debug)

            strengths = {
                "top": rgb_strengths["top"] + notch_weight * notch_strengths["top"],
                "bottom": rgb_strengths["bottom"] + notch_weight * notch_strengths["bottom"],
                "left": rgb_strengths["left"] + notch_weight * notch_strengths["left"],
                "right": rgb_strengths["right"] + notch_weight * notch_strengths["right"],
            }
            horizontal_score = np.sqrt(max(strengths["top"], 0.0) * max(strengths["bottom"], 0.0))
            vertical_score = np.sqrt(max(strengths["left"], 0.0) * max(strengths["right"], 0.0))

            width_est, height_est = side_lengths(c)

            geom_score = (width_est - height_est) / (max(width_est, height_est) + 1e-6)

            total_score = (horizontal_score - vertical_score) + geom_weight * geom_score

            return {
                "corners": c,
                "warped_tmp": warped_img,
                "warped_tmp_mask": warped_mask,
                "dark_mask": rgb_debug["dark_mask"],
                "zone_vis": rgb_debug["zone_vis"],
                "strip_vis": rgb_debug["strip_vis"],
                "candidate_vis": rgb_debug["candidate_vis"],
                "accepted": rgb_debug["accepted"],
                "rejected": rgb_debug["rejected"],
                "rgb_strengths": rgb_strengths,
                "notch_strengths": notch_strengths,
                "strengths": strengths,
                "horizontal_score": horizontal_score,
                "vertical_score": vertical_score,
                "geom_score": geom_score,
                "total_score": total_score,
                "notch_vis": notch_vis,
                "width_est": width_est,
                "height_est": height_est,
            }

        cand0 = evaluate(np.roll(canonical, 0, axis=0).astype(np.float32))
        cand1 = evaluate(np.roll(canonical, -1, axis=0).astype(np.float32))

        chosen = cand1 if cand1["total_score"] > cand0["total_score"] else cand0

        debug = {
            "candidate0_horizontal_score": cand0["horizontal_score"],
            "candidate0_vertical_score": cand0["vertical_score"],
            "candidate0_geom_score": cand0["geom_score"],
            "candidate0_total_score": cand0["total_score"],
            "candidate0_strengths": cand0["strengths"],
            "candidate0_width_est": cand0["width_est"],
            "candidate0_height_est": cand0["height_est"],

            "candidate1_horizontal_score": cand1["horizontal_score"],
            "candidate1_vertical_score": cand1["vertical_score"],
            "candidate1_geom_score": cand1["geom_score"],
            "candidate1_total_score": cand1["total_score"],
            "candidate1_strengths": cand1["strengths"],
            "candidate1_width_est": cand1["width_est"],
            "candidate1_height_est": cand1["height_est"],

            "rotation_k": 1 if chosen is cand1 else 0,
            "rotated_90": chosen is cand1,

            "warped_tmp": chosen["warped_tmp"],
            "warped_tmp_mask": chosen["warped_tmp_mask"],
            "dark_mask": chosen["dark_mask"],
            "zone_vis": chosen["zone_vis"],
            "strip_vis": chosen["strip_vis"],
            "candidate_vis": chosen["candidate_vis"],
            "accepted": chosen["accepted"],
            "rejected": chosen["rejected"],
            "rgb_strengths": chosen["rgb_strengths"],
            "notch_strengths": chosen["notch_strengths"],
            "strengths": chosen["strengths"],
            "horizontal_score": chosen["horizontal_score"],
            "vertical_score": chosen["vertical_score"],
            "geom_score": chosen["geom_score"],
            "total_score": chosen["total_score"],
            "notch_vis": chosen["notch_vis"],
            "notch_weight": notch_weight,
            "geom_weight": geom_weight,
        }

        return chosen["corners"], debug

    def _draw_debug_overlay(self, image_bgr, contour, hull, hull_contour, rough_corners, refined_corners, debug):
        vis = image_bgr.copy()
        h, w = vis.shape[:2]

        cv2.drawContours(vis, [contour], -1, (0, 255, 0), 2)
        cv2.drawContours(vis, [hull], -1, (255, 0, 0), 2)
        cv2.drawContours(vis, [hull_contour], -1, (0, 200, 255), 2)

        rc = np.round(rough_corners).astype(np.int32).reshape(-1, 1, 2)
        cv2.drawContours(vis, [rc], -1, (0, 255, 255), 2)

        side_colors = [
            (255, 255, 0),
            (255, 0, 255),
            (0, 165, 255),
            (180, 255, 180),
        ]

        for i, pts in enumerate(debug["trimmed_sets"]):
            color = side_colors[i % len(side_colors)]
            for p in pts:
                x, y = int(p[0]), int(p[1])
                cv2.circle(vis, (x, y), 1, color, -1)

        for line in debug["fitted_lines"]:
            pair = self._line_to_border_points(line, w, h)
            if pair is None:
                continue
            p1, p2 = pair
            p1 = tuple(np.round(p1).astype(int))
            p2 = tuple(np.round(p2).astype(int))
            cv2.line(vis, p1, p2, (255, 255, 255), 2)

        for i, p in enumerate(rough_corners):
            x, y = int(p[0]), int(p[1])
            cv2.circle(vis, (x, y), 7, (0, 0, 255), -1)
            cv2.putText(vis, f"R{i}", (x + 8, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        for i, p in enumerate(refined_corners):
            x, y = int(p[0]), int(p[1])
            cv2.circle(vis, (x, y), 7, (255, 255, 255), -1)
            cv2.circle(vis, (x, y), 4, (0, 0, 0), -1)
            cv2.putText(vis, f"F{i}", (x + 8, y + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return vis

    def _show_debug(self,image_bgr,mask_blue,mask_roi,mask_table_raw,mask_table_filled,overlay,warped,pocket_dark_mask=None,pocket_strengths=None,candidate_vis=None,notch_vis=None):
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
        warped_rgb = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)

        plt.figure(figsize=(20, 11))

        plt.subplot(2, 4, 1)
        plt.imshow(image_rgb)
        plt.title("Original")
        plt.axis("off")

        plt.subplot(2, 4, 2)
        plt.imshow(mask_blue, cmap="gray")
        plt.title("Blue mask")
        plt.axis("off")

        plt.subplot(2, 4, 3)
        plt.imshow(mask_roi, cmap="gray")
        plt.title("After spatial prior")
        plt.axis("off")

        plt.subplot(2, 4, 4)
        plt.imshow(mask_table_raw, cmap="gray")
        plt.title("Selected blue cloth (raw)")
        plt.axis("off")

        plt.subplot(2, 4, 5)
        plt.imshow(overlay_rgb)
        plt.title("Contour + hull + fitted edges")
        plt.axis("off")

        if candidate_vis is not None:
            plt.subplot(2, 4, 6)
            plt.imshow(cv2.cvtColor(candidate_vis, cv2.COLOR_BGR2RGB))
            title = "Pocket candidates"
            if pocket_strengths is not None:
                title += (
                    f"\nT={pocket_strengths['top']:.2f} "
                    f"B={pocket_strengths['bottom']:.2f} "
                    f"L={pocket_strengths['left']:.2f} "
                    f"R={pocket_strengths['right']:.2f}"
                )
            plt.title(title)
            plt.axis("off")

        if notch_vis is not None:
            plt.subplot(2, 4, 7)
            plt.imshow(cv2.cvtColor(notch_vis, cv2.COLOR_BGR2RGB))
            plt.title("Mask-notch evidence")
            plt.axis("off")

        plt.subplot(2, 4, 8)
        plt.imshow(warped_rgb)
        plt.title("Final warped cloth")
        plt.axis("off")

        plt.tight_layout()
        plt.show()

    def _detect_blue_cloth_quad(self, image_bgr):
        mask_blue = self._segment_table_blue(image_bgr)
        mask_roi = self._apply_spatial_prior(mask_blue)
        mask_roi = self._suppress_top_text_components(mask_roi)

        mask_table_raw = self._find_best_table_component(mask_roi)

        mask_table_filled = self._fill_table_component(mask_table_raw)

        contour, hull, hull_mask = self._build_hull_mask(mask_table_filled)
        hull_contour = self._get_largest_contour(hull_mask)

        rough_corners = self._detect_initial_corners_from_contour(hull_contour)
        fitted_lines, debug = self._fit_side_lines_from_dense_hull_contour(hull_contour, rough_corners)
        refined_corners = self._refine_corners_from_lines(fitted_lines, image_bgr.shape)

        if not self._quad_is_reasonable(refined_corners, image_bgr.shape):
            raise ValueError("Detected cloth quadrilateral is not reasonable.")

        return {
            "mask_blue": mask_blue,
            "mask_roi": mask_roi,
            "mask_table_raw": mask_table_raw,
            "mask_table_filled": mask_table_filled,
            "contour": contour,
            "hull": hull,
            "hull_mask": hull_mask,
            "hull_contour": hull_contour,
            "rough_corners": rough_corners,
            "refined_corners": refined_corners,
            "debug": debug,
        }

    def debug_blue_cloth_pipeline(self, image_path, out_w=720, out_h=400, auto_size=False):
        image_bgr = cv2.imread(image_path)
        if image_bgr is None:
            raise FileNotFoundError(f"Could not load image: {image_path}")

        result = self._detect_blue_cloth_quad(image_bgr)

        overlay = self._draw_debug_overlay(
            image_bgr=image_bgr,
            contour=result["contour"],
            hull=result["hull"],
            hull_contour=result["hull_contour"],
            rough_corners=result["rough_corners"],
            refined_corners=result["refined_corners"],
            debug=result["debug"]
        )

        oriented_corners, pocket_debug = self._decide_corner_rotation_using_middle_pockets(
            image_bgr=image_bgr,
            raw_mask_table=result["mask_table_raw"],
            corners=result["refined_corners"],
            tmp_w=720,
            tmp_h=400,
            notch_weight=0.25,
            geom_weight=2.0
        )

        if auto_size:
            warped, H = self._warp_from_corners_auto(
                image_bgr,
                oriented_corners,
                assume_ordered=True
            )
        else:
            warped, H = self._warp_from_corners(
                image_bgr,
                oriented_corners,
                out_w=out_w,
                out_h=out_h,
                assume_ordered=True
            )

        self._show_debug(
            image_bgr=image_bgr,
            mask_blue=result["mask_blue"],
            mask_roi=result["mask_roi"],
            mask_table_raw=result["mask_table_raw"],
            mask_table_filled=result["mask_table_filled"],
            overlay=overlay,
            warped=warped,
            pocket_dark_mask=pocket_debug["dark_mask"],
            pocket_strengths=pocket_debug["strengths"],
            candidate_vis=pocket_debug["candidate_vis"],
            notch_vis=pocket_debug["notch_vis"]
        )

        print("\nRough corners:")
        print(result["rough_corners"])

        print("\nRefined cloth corners:")
        print(result["refined_corners"])

        print("\nOriented corners:")
        print(oriented_corners)

        print("\nRGB strengths:")
        print(pocket_debug["rgb_strengths"])

        print("\nMask-notch strengths:")
        print(pocket_debug["notch_strengths"])

        print("\nCombined strengths:")
        print(pocket_debug["strengths"])

        print("\nHorizontal score:", pocket_debug["horizontal_score"])
        print("Vertical score:", pocket_debug["vertical_score"])
        print("Rotation k:", pocket_debug["rotation_k"])
        print("Notch weight:", pocket_debug["notch_weight"])

        print("\nAccepted RGB middle-pocket candidates:")
        for a in pocket_debug["accepted"]:
            print(a)

        print("\nRejected RGB middle-pocket candidates:")
        for r in pocket_debug["rejected"]:
            print(r)

        print("\nHomography:")
        print(H)

        result["overlay"] = overlay
        result["warped"] = warped
        result["H"] = H
        result["oriented_corners"] = oriented_corners
        result["pocket_debug"] = pocket_debug
        return result

    def process_single_image(self, image_path, save_path=None, out_w=720, out_h=400, auto_size=False, notch_weight=0.25, geom_weight=2.0):
        image_bgr = cv2.imread(image_path)
        if image_bgr is None:
            raise FileNotFoundError(f"Could not load image: {image_path}")

        result = self._detect_blue_cloth_quad(image_bgr)

        oriented_corners, pocket_debug = self._decide_corner_rotation_using_middle_pockets(
            image_bgr=image_bgr,
            raw_mask_table=result["mask_table_raw"],
            corners=result["refined_corners"],
            tmp_w=720,
            tmp_h=400,
            notch_weight=notch_weight,
            geom_weight=geom_weight
        )

        if auto_size:
            warped, H = self._warp_from_corners_auto(
                image_bgr,
                oriented_corners,
                assume_ordered=True
            )
        else:
            warped, H = self._warp_from_corners(
                image_bgr=image_bgr,
                corners=oriented_corners,
                out_w=out_w,
                out_h=out_h,
                assume_ordered=True
            )

        if save_path is not None:
            save_dir = os.path.dirname(save_path)
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)

            ok = cv2.imwrite(save_path, warped)
            if not ok:
                raise IOError(f"Failed to save warped image to: {save_path}")

        return {
            "warped": warped,
            "homography": H,
            "rough_corners": result["rough_corners"],
            "refined_corners": result["refined_corners"],
            "oriented_corners": oriented_corners,

            "rotation_k": pocket_debug["rotation_k"],
            "rotated_90": pocket_debug["rotated_90"],

            "rgb_strengths": pocket_debug["rgb_strengths"],
            "notch_strengths": pocket_debug["notch_strengths"],
            "combined_strengths": pocket_debug["strengths"],

            "horizontal_score": pocket_debug["horizontal_score"],
            "vertical_score": pocket_debug["vertical_score"],
            "geom_score": pocket_debug["geom_score"],
            "total_score": pocket_debug["total_score"],

            "candidate0_total_score": pocket_debug["candidate0_total_score"],
            "candidate1_total_score": pocket_debug["candidate1_total_score"],

            "candidate0_horizontal_score": pocket_debug["candidate0_horizontal_score"],
            "candidate0_vertical_score": pocket_debug["candidate0_vertical_score"],
            "candidate1_horizontal_score": pocket_debug["candidate1_horizontal_score"],
            "candidate1_vertical_score": pocket_debug["candidate1_vertical_score"],
        }
    
# Reading from JSON file
def read_json(input_json):
    with open(input_json) as f:
        data = json.load(f)

    if isinstance(data, dict) and "image_path" in data:
        image_paths = data["image_path"]
    elif isinstance(data, list):
        image_paths = data
    else:
        sys.exit(1)

    os.makedirs(TOP_VIEW_DIR, exist_ok=True)
    return image_paths

# Outputing the classifications and positions of the balls for each image
def write_json(output_json, all_results):
    with open(output_json, "w") as f:
        json.dump(all_results, f, indent=4)

def detect_and_format_balls(frame, debug):
    if frame is None:
        return 0, []

    bd = Ball_Detection()
    _, balls_raw = bd.detect_billiard_balls(frame, debug=debug)

    img_h, img_w = frame.shape[:2]
    balls_data = []
    for (x, y, r) in balls_raw:
        padding = int(r * 0.3) + 3
        balls_data.append({
            "xmin": float(max(0, x - r - padding)) / img_w,
            "xmax": float(min(img_w, x + r + padding)) / img_w,
            "ymin": float(max(0, y - r - padding)) / img_h,
            "ymax": float(min(img_h, y + r + padding)) / img_h,
        })

    return len(balls_data), balls_data

def main():

    debug_mode = "--debug" in sys.argv
    if debug_mode:
        sys.argv.remove("--debug")

    input_json_path = sys.argv[1] if len(sys.argv) > 1 else "input.json"
    output_json_path = sys.argv[2] if len(sys.argv) > 2 else "output.json"

    image_paths = read_json(input_json=input_json_path)

    p = Preprocessing(height=HEIGHT, width=WIDTH)
    bc = Ball_Classification()
    tp = Top_View()

    if debug_mode:
        p.plot_resolutions()

    all_results = []

    for img_path_str in image_paths:
        img_path = Path(img_path_str)
        img_name = img_path.name

        if not img_path.exists():
            all_results.append({
                "image_path": img_path_str,
                "num_balls": 0,
                "balls": []
            })
            continue

        processed_path = PROCESSED_DIR / img_name
        
        p.process_and_save_image(str(img_path), str(processed_path))

        frame = cv2.imread(str(processed_path))
        ball_count, balls_list = detect_and_format_balls(frame, debug_mode)

        cleaned_balls = []
        if frame is not None and balls_list:
            labels = bc.classify_image(frame, balls_list)
            for b, label in zip(balls_list, labels):
                cleaned_balls.append({
                    "number": bc.label_to_number(label),
                    "xmin": b["xmin"],
                    "xmax": b["xmax"],
                    "ymin": b["ymin"],
                    "ymax": b["ymax"],
                })
        all_results.append({
            "image_path": img_path_str,
            "num_balls": len(cleaned_balls),
            "balls": cleaned_balls,
        })
        if debug_mode:
            bc.visualize_classification(frame, cleaned_balls, labels)

        name_no_ext = os.path.splitext(img_name)[0]
        topview_path = os.path.join(TOP_VIEW_DIR, f"{name_no_ext}.jpg")
        try:
            tp.process_single_image(
                image_path=str(processed_path),
                save_path=topview_path,
                out_w=720,
                out_h=400,
                auto_size=False,
            )
            if debug_mode:
                tp.debug_blue_cloth_pipeline(image_path=str(processed_path))
        except Exception:
            pass
    
    write_json(output_json=output_json_path, all_results=all_results)

if __name__ == "__main__":
    main()
