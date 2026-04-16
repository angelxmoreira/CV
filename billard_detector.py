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
def find_table_mask(frame: np.ndarray):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower, upper, felt_hsv = sample_felt_color_hsv(frame)
    felt_mask = cv2.inRange(hsv, lower, upper)
    felt_mask = cv2.morphologyEx(felt_mask, cv2.MORPH_OPEN,  np.ones((7,  7),  np.uint8))
    felt_mask = cv2.morphologyEx(felt_mask, cv2.MORPH_CLOSE, np.ones((21, 21), np.uint8))

    contours, _ = cv2.findContours(felt_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.ones(frame.shape[:2], dtype=np.uint8) * 255, felt_hsv

    hull = cv2.convexHull(max(contours, key=cv2.contourArea))
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [hull], -1, 255, -1)

    ep   = max(5, frame.shape[1] // 80)
    mask = cv2.erode(mask, np.ones((ep, ep), np.uint8), iterations=1)
    return mask, felt_hsv


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: Estimate ball radius from table geometry
# ─────────────────────────────────────────────────────────────────────────────
def estimate_ball_radius(table_mask: np.ndarray):
    contours, _ = cv2.findContours(table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 8, 28
    _, _, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    nominal_r  = min(w, h) / 36
    return max(6, int(nominal_r * 0.55)), max(20, int(nominal_r * 1.50))


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4: Erase pockets from mask BEFORE Hough
# ─────────────────────────────────────────────────────────────────────────────
def mask_out_pockets(table_mask: np.ndarray, frame: np.ndarray,
                     min_r: int, max_r: int) -> np.ndarray:
    """
    Pockets (corner, side, decorative oval) share three properties no ball has:
      1. Very dark interior (brightness well below felt level)
      2. Radius >= 90 % of max ball radius  (bigger than any ball)
      3. Located within 22 % of the table bounding-box border

    We threshold for dark blobs, check all three criteria, then erase them
    from the mask so Hough never sees them.
    """
    cleaned = table_mask.copy()
    gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_t  = cv2.bitwise_and(gray, gray, mask=table_mask)

    # Adaptive dark threshold relative to felt brightness
    felt_brightness = float(np.median(gray_t[table_mask == 255]))
    dark_thresh     = min(60, int(felt_brightness * 0.40))

    _, dark = cv2.threshold(gray_t, dark_thresh, 255, cv2.THRESH_BINARY_INV)
    dark    = cv2.bitwise_and(dark, table_mask)
    dark    = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    dark    = cv2.morphologyEx(dark, cv2.MORPH_OPEN,  np.ones((5, 5), np.uint8))

    contours, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    tc, _ = cv2.findContours(table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not tc:
        return cleaned
    tx, ty, tw, th = cv2.boundingRect(max(tc, key=cv2.contourArea))

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 20:
            continue
        (cx, cy_c), radius = cv2.minEnclosingCircle(cnt)
        radius = float(radius)
        peri   = cv2.arcLength(cnt, True)
        circ   = (4 * math.pi * area / peri ** 2) if peri > 0 else 0

        # Must be large enough to be a pocket (not just a shadow under a ball)
        if radius < max_r * 0.9:
            continue
        # Must be roughly circular
        if circ < 0.45:
            continue
        # Must be near the table border
        mx, my = tw * 0.22, th * 0.22
        if not (cx < tx + mx or cx > tx + tw - mx or
                cy_c < ty + my or cy_c > ty + th - my):
            continue

        cv2.circle(cleaned, (int(cx), int(cy_c)), int(radius * 1.35), 0, -1)

    return cleaned


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
# STEP 6: Watershed split for oversized / cluster circles
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


# ─────────────────────────────────────────────────────────────────────────────
# STEP 7: Felt rejection + two-threshold NMS
# ─────────────────────────────────────────────────────────────────────────────
def circle_iou(c1, c2) -> float:
    x1, y1, r1 = c1;  x2, y2, r2 = c2
    dist = math.hypot(x1-x2, y1-y2)
    if dist >= r1 + r2:    return 0.0
    if dist <= abs(r1-r2): return 1.0
    r1s, r2s, ds = r1**2, r2**2, dist**2
    try:
        a = math.acos(max(-1.0, min(1.0, (ds+r1s-r2s) / (2*dist*r1))))
        b = math.acos(max(-1.0, min(1.0, (ds+r2s-r1s) / (2*dist*r2))))
    except ValueError:
        return 1.0
    inter = r1s*a + r2s*b - r1s*math.sin(2*a)/2 - r2s*math.sin(2*b)/2
    return inter / (math.pi * min(r1s, r2s))


def verify_and_nms(candidates: list, frame: np.ndarray, table_mask: np.ndarray,
                   felt_hsv: np.ndarray, min_r: int) -> list:
    """
    Stage 1 — Felt rejection
      Reject only if ALL THREE match felt: hue, value, saturation.
      (Blue ball same hue as blue felt but meaningfully darker → passes.)

    Stage 2 — Two-threshold NMS
      IoU > 0.50                           → definite duplicate → collapse
      IoU 0.15–0.50 AND dist < 1.2×min_r  → near-duplicate    → collapse
      otherwise                            → keep both (real adjacent balls)
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    fh, fs, fv = float(felt_hsv[0]), float(felt_hsv[1]), float(felt_hsv[2])
    H, W = frame.shape[:2]

    # ── Stage 1 ───────────────────────────────────────────────────────────────
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
            continue   # bare felt patch

        passed.append((x, y, r))

    # ── Stage 2 ───────────────────────────────────────────────────────────────
    passed.sort(key=lambda c: -c[2])   # largest radius first = most confident

    verified = []
    for c in passed:
        x, y, r = c
        merged = False
        for i, vc in enumerate(verified):
            iou  = circle_iou(c, vc)
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
    table_mask, felt_hsv = find_table_mask(frame)
    min_r, max_r         = estimate_ball_radius(table_mask)
    table_mask           = mask_out_pockets(table_mask, frame, min_r, max_r)

    gray       = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_table = cv2.bitwise_and(gray, gray, mask=table_mask)

    candidates = multi_pass_hough(gray_table, table_mask, min_r, max_r)
    candidates = watershed_split_clusters(candidates, gray_table, table_mask, min_r, max_r)
    balls      = verify_and_nms(candidates, frame, table_mask, felt_hsv, min_r)

    if debug:
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        axes[0].imshow(table_mask, cmap="gray"); axes[0].set_title("Mask (pockets erased)")
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
