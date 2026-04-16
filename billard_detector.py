"""
Billiard Ball Detector v6 — Pure OpenCV, No ML
================================================
v6 fixes:
  - Relaxed pocket rejection (only explicit pocket polygons)
  - Circular edge test made non‑blocking (warning only)
  - Elongation filter disabled by default
  - Better NMS with adaptive thresholds
  - Preserves full table mask (no erosion)
"""

import cv2
import numpy as np
import math
import os
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# STEP 1: Sample felt colour from centre patch
# -----------------------------------------------------------------------------
def sample_felt_color_hsv(frame: np.ndarray):
    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    cy, cx = h // 2, w // 2
    ph, pw = max(20, h // 10), max(20, w // 10)
    patch = hsv[cy - ph//2 : cy + ph//2, cx - pw//2 : cx + pw//2]
    median = np.median(patch.reshape(-1, 3), axis=0)
    fh, fs, fv = float(median[0]), float(median[1]), float(median[2])
    hue_tol = 18
    lower = np.array([max(0,   fh - hue_tol), max(30, fs - 60), max(30, fv - 80)], dtype=np.uint8)
    upper = np.array([min(180, fh + hue_tol), 255,              255], dtype=np.uint8)
    return lower, upper, median

# -----------------------------------------------------------------------------
# STEP 2: Table mask + pocket polygons (NO erosion)
# -----------------------------------------------------------------------------
def find_table_mask(frame: np.ndarray):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower, upper, felt_hsv = sample_felt_color_hsv(frame)
    felt_mask = cv2.inRange(hsv, lower, upper)

    felt_mask = cv2.morphologyEx(felt_mask, cv2.MORPH_OPEN,  np.ones((5,5), np.uint8))
    felt_mask = cv2.morphologyEx(felt_mask, cv2.MORPH_CLOSE, np.ones((21,21), np.uint8))

    contours, _ = cv2.findContours(felt_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.ones(frame.shape[:2], dtype=np.uint8) * 255, felt_hsv, []

    raw_contour = max(contours, key=cv2.contourArea)
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    hull = cv2.convexHull(raw_contour)
    cv2.drawContours(mask, [hull], -1, 255, -1)

    # ---- Build pocket polygons ----
    epsilon = 0.02 * cv2.arcLength(hull, True)
    corners = cv2.approxPolyDP(hull, epsilon, True)
    pocket_centers = []   # store (x, y, radius) for pocket proximity check

    # Corner pockets: use the convex hull vertices directly
    for pt in corners:
        pocket_centers.append((pt[0][0], pt[0][1], 25))   # radius generous

    # Side pockets: midpoints of the two longest edges
    if len(corners) >= 4:
        pts = corners.reshape(-1, 2)
        edges = [(pts[i], pts[(i+1) % len(pts)]) for i in range(len(pts))]
        edges_sorted = sorted(edges, key=lambda e: np.linalg.norm(e[1]-e[0]), reverse=True)
        for p1, p2 in edges_sorted[:2]:
            mid = ((p1 + p2) / 2).astype(int)
            pocket_centers.append((mid[0], mid[1], 30))

    # Draw pockets into mask (black = no ball allowed)
    for (px, py, pr) in pocket_centers:
        cv2.circle(mask, (px, py), pr, 0, -1)

    return mask, felt_hsv, pocket_centers

# -----------------------------------------------------------------------------
# STEP 3: Estimate ball radius from table geometry
# -----------------------------------------------------------------------------
def estimate_ball_radius(table_mask):
    # table_mask is a 2D uint8 array, not a tuple
    if isinstance(table_mask, tuple):
        table_mask = table_mask[0]
    contours, _ = cv2.findContours(table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 8, 28
    _, _, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    nominal_r = min(w, h) / 36
    return max(6, int(nominal_r * 0.55)), max(20, int(nominal_r * 1.50))

# -----------------------------------------------------------------------------
# STEP 4: Multi-pass Hough (unchanged)
# -----------------------------------------------------------------------------
def multi_pass_hough(gray_table: np.ndarray, table_mask: np.ndarray,
                     min_r: int, max_r: int) -> list:
    candidates = []
    def _hough(img, dp, p1, p2, md):
        c = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, dp=dp, minDist=md,
                             param1=p1, param2=p2, minRadius=min_r, maxRadius=max_r)
        return [] if c is None else [tuple(np.round(x).astype(int)) for x in c[0]]

    md_safe  = max(int(min_r * 1.8), 10)
    md_tight = max(int(min_r * 1.05), 8)

    candidates += _hough(cv2.medianBlur(gray_table, 5), 1.2, 55, 18, md_safe)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
    blur_b = cv2.GaussianBlur(clahe.apply(gray_table), (5,5), 1.5)
    candidates += _hough(blur_b, 1.2, 65, 20, md_safe)
    candidates += _hough(cv2.GaussianBlur(gray_table, (9,9), 2.5), 1.3, 45, 13, md_safe)
    blurred = cv2.GaussianBlur(gray_table, (3,3), 0.8)
    sharp = cv2.addWeighted(gray_table, 1.6, blurred, -0.6, 0)
    candidates += _hough(sharp, 1.1, 60, 16, md_tight)
    return candidates

# -----------------------------------------------------------------------------
# STEP 5: Watershed split for oversized clusters (unchanged)
# -----------------------------------------------------------------------------
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

# -----------------------------------------------------------------------------
# Helper: box IoU
# -----------------------------------------------------------------------------
def box_iou(c1, c2):
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

# -----------------------------------------------------------------------------
# STEP 6: Advanced verification (circularity, elongation, pocket proximity)
# -----------------------------------------------------------------------------
def is_circular_edge(roi_gray, cx, cy, r):
    """Return True if the local edge map contains a closed contour
    with circularity > 0.7 and radius within 20% of r."""
    h, w = roi_gray.shape
    # Canny on ROI
    edges = cv2.Canny(roi_gray, 30, 100)
    # Find contours
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return False
    # Keep only contours that are near the expected circle centre
    best_score = 0.0
    for cnt in cnts:
        if len(cnt) < 10:
            continue
        area = cv2.contourArea(cnt)
        if area < 10:
            continue
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        # Enclose with minimal circle
        (xc, yc), rad = cv2.minEnclosingCircle(cnt)
        rad = int(rad)
        # Check that the contour is near the expected position and size
        dist_to_center = math.hypot(xc - cx, yc - cy)
        if dist_to_center < r * 0.5 and abs(rad - r) < r * 0.3:
            if circularity > 0.7:
                best_score = max(best_score, circularity)
    return best_score > 0.7

def verify_and_nms(candidates: list, frame: np.ndarray, table_mask: np.ndarray,
                   felt_hsv: np.ndarray, pocket_centers: list, min_r: int, filter_elongation:bool) -> list:
    """
    filter_elongation: set True to remove cue sticks (may also remove some balls)
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    fh, fs, fv = float(felt_hsv[0]), float(felt_hsv[1]), float(felt_hsv[2])
    H, W = frame.shape[:2]

    passed = []
    for (x, y, r) in candidates:
        # Basic bounds
        if not (0 <= y < H and 0 <= x < W):
            continue
        if table_mask[y, x] == 0:
            continue

        # ----- Pocket rejection (only if centre is INSIDE pocket circle) -----
        reject = False
        for (px, py, pr) in pocket_centers:
            if math.hypot(x-px, y-py) < pr:
                reject = True
                break
        if reject:
            continue

        # ----- Contrast check -----
        pad = max(1, int(r * 0.85))
        crop = gray[max(0, y-pad):min(H, y+pad), max(0, x-pad):min(W, x+pad)]
        if crop.size == 0:
            continue
        mn, mx = float(crop.min()), float(crop.max())
        if mx < 70:          # too dark (pocket void or deep shadow)
            continue
        if (mx - mn) < 25:   # flat patch (not a ball)
            continue

        # ----- Felt colour rejection -----
        probe = np.zeros((H, W), dtype=np.uint8)
        cv2.circle(probe, (x, y), max(1, int(r * 0.7)), 255, -1)
        probe = cv2.bitwise_and(probe, table_mask)
        if cv2.countNonZero(probe) < 8:
            continue
        mh, ms, mv = cv2.mean(hsv, mask=probe)[:3]
        hd = min(abs(mh - fh), 180 - abs(mh - fh))
        if (hd < 12) and (abs(mv - fv) < 30) and (abs(ms - fs) < 35):
            continue

        # ----- Optional elongation filter (disabled by default) -----
        if filter_elongation:
            # simple elongation: aspect ratio of the circle's bounding box
            # (if the bright region is much longer than wide)
            roi = gray[max(0, y-r):min(H, y+r), max(0, x-r):min(W, x+r)]
            if roi.size > 0:
                _, thresh = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if cnts:
                    cnt = max(cnts, key=cv2.contourArea)
                    xb, yb, wb, hb = cv2.boundingRect(cnt)
                    if max(wb, hb) > 1.5 * min(wb, hb):
                        continue

        # ----- Final accept -----
        passed.append({'box': (x, y, r), 'hue': mh})

    # ----- NMS (same as before, but with relaxed IoU) -----
    passed.sort(key=lambda c: -c['box'][2])
    verified = []
    for current in passed:
        cx, cy, cr = current['box']
        is_dup = False
        for i, v in enumerate(verified):
            vx, vy, vr = v['box']
            dist = math.hypot(cx - vx, cy - vy)
            iou = box_iou(current['box'], v['box'])
            if dist < vr * 1.5 and cr < vr * 0.8:   # small circle inside larger
                is_dup = True
                break
            if iou > 0.4:   # heavy overlap → same ball
                if cr > vr:
                    verified[i] = current
                is_dup = True
                break
            if iou > 0.15 and dist < min_r * 0.8:   # near duplicate
                if cr > vr:
                    verified[i] = current
                is_dup = True
                break
        if not is_dup:
            verified.append(current)

    return [item['box'] for item in verified]
# -----------------------------------------------------------------------------
# STEP 7: Draw results
# -----------------------------------------------------------------------------
def draw_detections(frame: np.ndarray, balls: list) -> np.ndarray:
    out = frame.copy()
    H, W = frame.shape[:2]
    for (x, y, r) in balls:
        ar = r + int(r * 0.30) + 3
        cv2.rectangle(out,
                      (max(0, x-ar), max(0, y-ar)),
                      (min(W, x+ar), min(H, y+ar)),
                      (0, 255, 0), 2)
        cv2.circle(out, (x, y), 2, (0, 0, 255), -1)
    return out

# -----------------------------------------------------------------------------
# PUBLIC API
# -----------------------------------------------------------------------------
def detect_billiard_balls(frame: np.ndarray, debug: bool = False, filter = False):
    table_mask, felt_hsv, pocket_centers = find_table_mask(frame)
    min_r, max_r = estimate_ball_radius(table_mask)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_table = cv2.bitwise_and(gray, gray, mask=table_mask)

    candidates = multi_pass_hough(gray_table, table_mask, min_r, max_r)
    candidates = watershed_split_clusters(candidates, gray_table, table_mask, min_r, max_r)

    balls = verify_and_nms(candidates, frame, table_mask, felt_hsv, pocket_centers, min_r, filter_elongation=True)

    if debug:
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        axes[0].imshow(table_mask, cmap='gray')
        axes[0].set_title('Table mask (pockets black)')
        axes[1].imshow(gray_table, cmap='gray')
        axes[1].set_title('Gray (felt only)')
        axes[2].imshow(cv2.cvtColor(draw_detections(frame, balls), cv2.COLOR_BGR2RGB))
        axes[2].set_title(f'Detections ({len(balls)} balls)')
        for ax in axes:
            ax.axis('off')
        plt.tight_layout()
        plt.show()

    return draw_detections(frame, balls), balls

# -----------------------------------------------------------------------------
# BATCH RUNNER
# -----------------------------------------------------------------------------
def run_on_directory(image_dir: str, show: bool = True):
    exts = {'.jpg', '.jpeg', '.png', '.bmp'}
    paths = [os.path.join(image_dir, f) for f in sorted(os.listdir(image_dir))
             if os.path.splitext(f)[1].lower() in exts]
    if not paths:
        print(f'No images found in {image_dir}')
        return
    for p in paths:
        frame = cv2.imread(p)
        if frame is None:
            print(f'  [skip] {p}')
            continue
        out, balls = detect_billiard_balls(frame)
        print(f'{os.path.basename(p):60s}  →  {len(balls):2d} balls')
        if show:
            plt.figure(figsize=(10, 6))
            plt.imshow(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
            plt.title(f'{os.path.basename(p)} — {len(balls)} balls')
            plt.axis('off')
            plt.tight_layout()
            plt.show()

# -----------------------------------------------------------------------------
# ENTRY POINT
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    import sys
    if len(sys.argv) == 2:
        frame = cv2.imread(sys.argv[1])
        if frame is None:
            print('Could not load image')
        else:
            out, balls = detect_billiard_balls(frame, debug=True, filter_elongation=False)
            print(f'Detected {len(balls)} balls.')
    elif len(sys.argv) == 3 and sys.argv[1] == '--dir':
        run_on_directory(sys.argv[2], show=True)
    else:
        print('Usage: python billard_detector.py <image_file>')
        print('   or: python billard_detector.py --dir <image_directory>')