"""
Billiard Ball Detector v4 — Pure OpenCV, No ML
================================================
v4 fixes:
  - Pocket masking: dark-blob geometry + strict border proximity erases
    ALL pocket types (corner, side, decorative oval reliefs) before Hough
  - Touching balls: watershed segmentation splits dense clusters that
    Hough collapses into one oversized circle
  - NMS overhaul: two-threshold strategy — heavy IoU = duplicate (collapse),
    light IoU + far centres = separate balls (keep both)
"""

import cv2
import numpy as np
import math
import os
import matplotlib.pyplot as plt


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: Sample felt colour from centre patch
# ─────────────────────────────────────────────────────────────────────────────
def sample_felt_color_hsv(frame: np.ndarray):
    h, w  = frame.shape[:2]
    hsv   = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    cy, cx = h // 2, w // 2
    ph, pw = max(20, h // 10), max(20, w // 10)
    patch  = hsv[cy - ph//2 : cy + ph//2, cx - pw//2 : cx + pw//2]
    median = np.median(patch.reshape(-1, 3), axis=0)
    fh, fs, fv = float(median[0]), float(median[1]), float(median[2])
    hue_tol = 18
    lower = np.array([max(0,   fh - hue_tol), max(30, fs - 60), max(30, fv - 80)], dtype=np.uint8)
    upper = np.array([min(180, fh + hue_tol), 255,              255             ], dtype=np.uint8)
    return lower, upper, median


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: Convex-hull table mask
# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
# GEOMETRIC POCKET ERADICATION
# ─────────────────────────────────────────────────────────────────────────────
def find_table_mask(frame: np.ndarray):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower, upper, felt_hsv = sample_felt_color_hsv(frame)
    felt_mask = cv2.inRange(hsv, lower, upper)

    felt_mask = cv2.morphologyEx(felt_mask, cv2.MORPH_OPEN,  np.ones((5,  5),  np.uint8))
    felt_mask = cv2.morphologyEx(felt_mask, cv2.MORPH_CLOSE, np.ones((21, 21), np.uint8))

    contours, _ = cv2.findContours(felt_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.ones(frame.shape[:2], dtype=np.uint8) * 255, felt_hsv

    raw_contour = max(contours, key=cv2.contourArea)
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    hull = cv2.convexHull(raw_contour)
    cv2.drawContours(mask, [hull], -1, 255, -1)

    # Derive pocket erase radius from table size (pockets ~ 1.8–2.5× ball radius)
    _, _, tw, th = cv2.boundingRect(hull)
    nominal_ball_r = min(tw, th) / 36
    pocket_erase_r = int(nominal_ball_r * 2.2)   # generous enough to cover full pocket

    # Erase corners via approxPolyDP — these are always pocket locations
    epsilon = 0.02 * cv2.arcLength(hull, True)
    corners = cv2.approxPolyDP(hull, epsilon, True)
    for pt in corners:
        cv2.circle(mask, tuple(pt[0]), pocket_erase_r, 0, -1)

    # Also erase mid-points on the long sides (side pockets)
    # Pool tables always have 6 pockets: 4 corners + 2 side-centres
    if len(corners) >= 4:
        pts = corners.reshape(-1, 2)
        # Find the two longest edges — their midpoints are the side pockets
        edges = [(pts[i], pts[(i+1) % len(pts)]) for i in range(len(pts))]
        edges_sorted = sorted(edges, key=lambda e: np.linalg.norm(e[1]-e[0]), reverse=True)
        for p1, p2 in edges_sorted[:2]:
            mid = ((p1 + p2) / 2).astype(int)
            cv2.circle(mask, tuple(mid), pocket_erase_r, 0, -1)

    # Minimal erosion — only enough to remove the rail lip (not edge balls)
    ep = max(3, int(nominal_ball_r * 0.3))
    mask = cv2.erode(mask, np.ones((ep, ep), np.uint8), iterations=1)

    return mask, felt_hsv


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: Estimate ball radius from table geometry
# ─────────────────────────────────────────────────────────────────────────────
def estimate_ball_radius(table_mask):
    # SAFETY NET: If a copy/paste error turned the mask into a tuple, fix it!
    if isinstance(table_mask, tuple):
        table_mask = table_mask[0]
        
    contours, _ = cv2.findContours(table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return 8, 28
        
    _, _, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    nominal_r  = min(w, h) / 36
    
    return max(6, int(nominal_r * 0.55)), max(20, int(nominal_r * 1.50))

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5: Multi-pass Hough
# ─────────────────────────────────────────────────────────────────────────────
def multi_pass_hough(gray_table: np.ndarray, table_mask: np.ndarray,
                     min_r: int, max_r: int) -> list:
    """
    Pass A–C: minDist = 1.8 × min_r  — safe separation, avoids false splits
    Pass D:   minDist = 1.05 × min_r — tight, catches touching balls
              uses sharpened image so each ball's edge ring is distinct
    """
    candidates = []

    def _hough(img, dp, p1, p2, md):
        c = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT,
                             dp=dp, minDist=md,
                             param1=p1, param2=p2,
                             minRadius=min_r, maxRadius=max_r)
        return [] if c is None else [tuple(np.round(x).astype(int)) for x in c[0]]

    md_safe  = max(int(min_r * 1.8), 10)
    md_tight = max(int(min_r * 1.05), 8)

    candidates += _hough(cv2.medianBlur(gray_table, 5), 1.2, 55, 18, md_safe)

    clahe   = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    blur_b  = cv2.GaussianBlur(clahe.apply(gray_table), (5, 5), 1.5)
    candidates += _hough(blur_b, 1.2, 65, 20, md_safe)

    candidates += _hough(cv2.GaussianBlur(gray_table, (9, 9), 2.5), 1.3, 45, 13, md_safe)

    blurred  = cv2.GaussianBlur(gray_table, (3, 3), 0.8)
    sharp    = cv2.addWeighted(gray_table, 1.6, blurred, -0.6, 0)
    candidates += _hough(sharp, 1.1, 60, 16, md_tight)

    return candidates


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6: Edge-Based Cluster Splitting
# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
# GRADIENT-BASED WATERSHED
# ─────────────────────────────────────────────────────────────────────────────
def watershed_split_clusters(candidates: list, gray_table: np.ndarray,
                              table_mask: np.ndarray,
                              min_r: int, max_r: int) -> list:
    result = []
    oversized = [c for c in candidates if c[2] > max_r * 1.3]
    standard = [c for c in candidates if c[2] <= max_r * 1.3]
    
    result.extend(standard)

    for (x, y, r) in oversized:
        pad = r + min_r
        x0, y0 = max(0, x-pad), max(0, y-pad)
        x1  = min(gray_table.shape[1], x+pad)
        y1  = min(gray_table.shape[0], y+pad)
        
        roi = gray_table[y0:y1, x0:x1]
        mroi = table_mask[y0:y1, x0:x1] 
        
        if roi.size == 0: continue

        # THE UPGRADE: Morphological Gradient
        # This acts like a spotlight on the boundaries between touching objects, 
        # even if they are the exact same shade of dark gray.
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        gradient = cv2.morphologyEx(roi, cv2.MORPH_GRADIENT, kernel)
        
        # Binarize the gradient (edges become white, balls become black)
        _, thresh = cv2.threshold(gradient, 15, 255, cv2.THRESH_BINARY)
        
        # Invert so balls are white pools
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

def box_iou(c1, c2):
    """Calculates the Intersection over Union (IoU) of two bounding boxes."""
    x1, y1, r1 = c1
    x2, y2, r2 = c2
    
    b1_x1, b1_y1, b1_x2, b1_y2 = x1-r1, y1-r1, x1+r1, y1+r1
    b2_x1, b2_y1, b2_x2, b2_y2 = x2-r2, y2-r2, x2+r2, y2+r2

    x_left = max(b1_x1, b2_x1)
    y_top = max(b1_y1, b2_y1)
    x_right = min(b1_x2, b2_x2)
    y_bottom = min(b1_y2, b2_y2)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    inter = (x_right - x_left) * (y_bottom - y_top)
    area1 = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    area2 = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    
    return inter / float(area1 + area2 - inter)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 7: Shine/Shadow Filter & Glare-Killing NMS
# ─────────────────────────────────────────────────────────────────────────────
def verify_and_nms(candidates: list, frame: np.ndarray, table_mask: np.ndarray,
                   felt_hsv: np.ndarray, min_r: int) -> list:
    """
    Stage 1 — Rejection filters (per-candidate):
      • Centre must be inside table mask
      • Must have some contrast (not a flat patch of felt)
      • Must not look like bare felt in all three HSV dimensions

    Stage 2 — NMS (duplicate suppression):
      • box_iou > 0.35  → definite same ball → keep larger
      • box_iou 0.10–0.35 AND centre dist < 1.3×min_r → near-dup → keep larger
      • Otherwise → independent balls, keep both

    The glare-killer: if a small circle's centre falls INSIDE a larger verified
    circle (dist < larger_r * 0.6), it is a glare/shadow sub-detection → drop it.
    """
    hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    fh, fs, fv = float(felt_hsv[0]), float(felt_hsv[1]), float(felt_hsv[2])
    H, W = frame.shape[:2]

    # ── Stage 1: per-candidate filters ───────────────────────────────────────
    passed = []
    for (x, y, r) in candidates:
        if not (0 <= y < H and 0 <= x < W): continue
        if table_mask[y, x] == 0:           continue

        # Contrast check on the ball crop — flat felt has very low contrast
        pad = max(1, int(r * 0.85))
        crop = gray[max(0, y-pad):min(H, y+pad), max(0, x-pad):min(W, x+pad)]
        if crop.size == 0: continue
        mn, mx = float(crop.min()), float(crop.max())

        # Pockets are near-black voids: no bright pixels at all
        if mx < 80:
            continue   # too dark — pocket or shadow void, not a ball

        # Flat textureless patch — not a 3D sphere
        if (mx - mn) < 30:
            continue

        # Felt colour rejection: only reject if ALL THREE channels match felt
        probe = np.zeros((H, W), dtype=np.uint8)
        cv2.circle(probe, (x, y), max(1, int(r * 0.7)), 255, -1)
        probe = cv2.bitwise_and(probe, table_mask)
        if cv2.countNonZero(probe) < 8: continue

        mh, ms, mv = cv2.mean(hsv, mask=probe)[:3]
        hd = min(abs(mh - fh), 180 - abs(mh - fh))
        if (hd < 14) and (abs(mv - fv) < 35) and (abs(ms - fs) < 40):
            continue   # bare felt patch

        passed.append({'box': (x, y, r), 'hue': mh})

    # ── Stage 2: NMS ─────────────────────────────────────────────────────────
    # Sort largest radius first (most confident Hough vote wins)
    passed.sort(key=lambda c: -c['box'][2])

    verified = []
    for current in passed:
        cx, cy, cr = current['box']
        c_hue      = current['hue']
        is_dup     = False

        for i, v in enumerate(verified):
            vx, vy, vr = v['box']
            dist = math.hypot(cx - vx, cy - vy)
            iou  = box_iou(current['box'], v['box'])

            # Glare / shadow sub-detection: small circle centre is inside larger circle
            if dist < vr * 1.75 and cr < vr:
                is_dup = True
                break

            # Heavy overlap → same ball
            if iou > 0.35:
                if cr > vr:
                    verified[i] = current
                is_dup = True
                break

            # Moderate overlap AND very close centres → near-duplicate
            if iou > 0.10 and dist < min_r * 1:
                if cr > vr:
                    verified[i] = current
                is_dup = True
                break

        if not is_dup:
            verified.append(current)

    return [item['box'] for item in verified]


# ─────────────────────────────────────────────────────────────────────────────
# STEP 8: Draw results
# ─────────────────────────────────────────────────────────────────────────────
def draw_detections(frame: np.ndarray, balls: list) -> np.ndarray:
    out  = frame.copy()
    H, W = frame.shape[:2]
    for (x, y, r) in balls:
        ar = r + int(r * 0.30) + 3
        cv2.rectangle(out,
                      (max(0, x-ar), max(0, y-ar)),
                      (min(W, x+ar), min(H, y+ar)),
                      (0, 255, 0), 2)
        cv2.circle(out, (x, y), 2, (0, 0, 255), -1)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────
def detect_billiard_balls(frame: np.ndarray, debug: bool = False):
    """
    Detect billiard balls in a BGR frame.
    Returns (annotated_frame, list_of_(x, y, radius)).
    """

    # Ensure correct variable unpacking
    table_mask, felt_hsv = find_table_mask(frame)
    min_r, max_r = estimate_ball_radius(table_mask)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_table = cv2.bitwise_and(gray, gray, mask=table_mask)

    candidates = multi_pass_hough(gray_table, table_mask, min_r, max_r)
    candidates = watershed_split_clusters(candidates, gray_table, table_mask, min_r, max_r)
    
    # Note: Use the IoU & Dark Edge version of verify_and_nms here for data collection!
    balls = verify_and_nms(candidates, frame, table_mask, felt_hsv, min_r)

    if debug:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        axes[0].imshow(table_mask, cmap="gray"); axes[0].set_title("Table Mask")
        axes[1].imshow(gray_table, cmap="gray"); axes[1].set_title("Gray (felt only)")
        axes[2].imshow(cv2.cvtColor(draw_detections(frame, balls), cv2.COLOR_BGR2RGB))
        axes[2].set_title(f"Detections ({len(balls)} balls)")
        for ax in axes: ax.axis("off")
        plt.tight_layout(); plt.show()

    return draw_detections(frame, balls), balls


# ─────────────────────────────────────────────────────────────────────────────
# BATCH RUNNER
# ─────────────────────────────────────────────────────────────────────────────
def run_on_directory(image_dir: str, show: bool = True):
    exts  = {".jpg", ".jpeg", ".png", ".bmp"}
    paths = [os.path.join(image_dir, f)
             for f in sorted(os.listdir(image_dir))
             if os.path.splitext(f)[1].lower() in exts]
    if not paths:
        print(f"No images found in {image_dir}"); return
    for p in paths:
        frame = cv2.imread(p)
        if frame is None: print(f"  [skip] {p}"); continue
        out, balls = detect_billiard_balls(frame)
        print(f"{os.path.basename(p):60s}  →  {len(balls):2d} balls")
        if show:
            plt.figure(figsize=(10, 6))
            plt.imshow(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
            plt.title(f"{os.path.basename(p)} — {len(balls)} balls")
            plt.axis("off"); plt.tight_layout(); plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    if len(sys.argv) == 2:
        frame = cv2.imread(sys.argv[1])
        out, balls = detect_billiard_balls(frame, debug=True)
        print(f"Detected {len(balls)} balls.")
    elif len(sys.argv) == 3 and sys.argv[1] == "--dir":
        run_on_directory(sys.argv[2], show=True)
    else:
        output_dir = "."
        img_path   = os.path.join(output_dir,
                        "4a_png.rf.a6bb5c5706fd8628eb53d34a122cf441.jpg")
        frame = cv2.imread(img_path)
        if frame is None:
            print("Pass an image path as argv[1]")
        else:
            output_frame, balls = detect_billiard_balls(frame, debug=True)
            print(f"Detected {len(balls)} validated balls.")
            plt.imshow(cv2.cvtColor(output_frame, cv2.COLOR_BGR2RGB))
            plt.axis("off"); plt.show()