"""
Billiard Ball Detector v2 — Pure OpenCV, No ML
================================================
Fixes over v1:
  - Felt-colour is sampled from image centre (table is always roughly centred)
  - Table mask = CONVEX HULL of the dominant felt blob, not raw morphology
  - Tighter colour tolerance rejects wooden rails, banners, crowd
  - Hough only runs inside the shrunk convex-hull felt region
  - Radius bounds derived from convex-hull dimensions, not bounding rect
"""

import cv2
import numpy as np
import math
import os
import matplotlib.pyplot as plt


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: Sample the felt colour from the image centre
# ─────────────────────────────────────────────────────────────────────────────
def sample_felt_color_hsv(frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    The pool table felt occupies the centre of the frame in broadcast footage.
    Sample a small central patch to learn the felt hue/saturation, then build
    tight HSV bounds around it.  This replaces ALL hard-coded colour ranges.

    Returns (lower_bound, upper_bound) as uint8 numpy arrays for cv2.inRange.
    """
    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Sample a 10 % × 10 % central patch — almost certainly felt
    cy, cx = h // 2, w // 2
    ph, pw = max(20, h // 10), max(20, w // 10)
    patch = hsv[cy - ph // 2: cy + ph // 2, cx - pw // 2: cx + pw // 2]

    # Median is robust to balls sitting in the centre of the patch
    median_hsv = np.median(patch.reshape(-1, 3), axis=0)
    felt_h, felt_s, felt_v = float(median_hsv[0]), float(median_hsv[1]), float(median_hsv[2])

    # Tight hue window (±18), moderate saturation floor, generous value range
    hue_tol = 18
    lower = np.array([max(0,   felt_h - hue_tol),
                      max(30,  felt_s - 60),
                      max(30,  felt_v - 80)], dtype=np.uint8)
    upper = np.array([min(180, felt_h + hue_tol),
                      255,
                      255], dtype=np.uint8)

    return lower, upper, median_hsv


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: Build a CONVEX-HULL table mask from the felt colour
# ─────────────────────────────────────────────────────────────────────────────
def find_table_mask(frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    1. Threshold on learned felt colour.
    2. Keep only the LARGEST blob (the felt).
    3. Take its CONVEX HULL — this fills holes left by balls and logos.
    4. Erode inward to exclude cushion/rail edges.

    Returns (table_mask, felt_hsv_median).
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower, upper, felt_hsv = sample_felt_color_hsv(frame)

    felt_mask = cv2.inRange(hsv, lower, upper)

    # Clean up specks and small gaps
    k_open  = np.ones((7,  7),  np.uint8)
    k_close = np.ones((21, 21), np.uint8)
    felt_mask = cv2.morphologyEx(felt_mask, cv2.MORPH_OPEN,  k_open)
    felt_mask = cv2.morphologyEx(felt_mask, cv2.MORPH_CLOSE, k_close)

    contours, _ = cv2.findContours(felt_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        # Fallback: full frame
        mask = np.ones(frame.shape[:2], dtype=np.uint8) * 255
        return mask, felt_hsv

    # Largest blob = felt
    largest = max(contours, key=cv2.contourArea)

    # Convex hull fills holes caused by dark balls / rack logos
    hull = cv2.convexHull(largest)
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [hull], -1, 255, -1)

    # Erode away rails/cushions — kept small so near-rail balls aren't lost
    erode_px = max(5, frame.shape[1] // 60)
    k_erode  = np.ones((erode_px, erode_px), np.uint8)
    mask = cv2.erode(mask, k_erode, iterations=1)

    return mask, felt_hsv


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: Scale-aware radius estimation from convex hull
# ─────────────────────────────────────────────────────────────────────────────
def estimate_ball_radius(table_mask: np.ndarray) -> tuple[int, int]:
    """
    Use the table mask to estimate ball radius.
    Standard pool table width ≈ 36 ball-diameters.
    We use the MINIMUM side of the bounding rect of the convex hull.
    """
    contours, _ = cv2.findContours(table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 8, 28

    _, _, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    ref = min(w, h)
    nominal_r = ref / 36
    min_r = max(6,  int(nominal_r * 0.55))
    max_r = max(20, int(nominal_r * 1.50))
    return min_r, max_r


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4: Multi-pass Hough (only on the felt region)
# ─────────────────────────────────────────────────────────────────────────────
def multi_pass_hough(
    gray_table: np.ndarray,
    table_mask: np.ndarray,
    min_r: int,
    max_r: int,
) -> list[tuple[int, int, int]]:
    """
    Three complementary Hough passes to catch all ball types:
      A — median blur, balanced params          (most balls)
      B — CLAHE + Gaussian, higher sensitivity  (dark / stripe balls)
      C — heavy blur, low thresholds            (white / cue ball)
    """
    min_dist = max(int(min_r * 1.6), 12)
    candidates = []

    def _hough(img, dp, p1, p2):
        c = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT,
                             dp=dp, minDist=min_dist,
                             param1=p1, param2=p2,
                             minRadius=min_r, maxRadius=max_r)
        return [] if c is None else [tuple(np.round(x).astype(int)) for x in c[0]]

    # Pass A
    blur_a = cv2.medianBlur(gray_table, 5)
    candidates += _hough(blur_a, 1.2, 55, 18)

    # Pass B — CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    eq = clahe.apply(gray_table)
    blur_b = cv2.GaussianBlur(eq, (5, 5), 1.5)
    candidates += _hough(blur_b, 1.2, 65, 20)

    # Pass C — soft edges (cue ball)
    blur_c = cv2.GaussianBlur(gray_table, (9, 9), 2.5)
    candidates += _hough(blur_c, 1.3, 45, 13)

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
# STEP 5: Felt-aware verification + NMS
# ─────────────────────────────────────────────────────────────────────────────
def circle_iou(c1: tuple, c2: tuple) -> float:
    """
    Approximate IoU for two circles using the ratio of intersection area
    to the smaller circle's area.  Fast and scale-invariant.
    """
    x1, y1, r1 = c1
    x2, y2, r2 = c2
    dist = math.hypot(x1 - x2, y1 - y2)
    # No overlap
    if dist >= r1 + r2:
        return 0.0
    # One fully inside the other
    if dist <= abs(r1 - r2):
        return 1.0
    # Partial overlap — lens area formula
    r1s, r2s, ds = r1 ** 2, r2 ** 2, dist ** 2
    alpha = math.acos((ds + r1s - r2s) / (2 * dist * r1))
    beta  = math.acos((ds + r2s - r1s) / (2 * dist * r2))
    intersection = r1s * alpha + r2s * beta - r1s * math.sin(2 * alpha) / 2 - r2s * math.sin(2 * beta) / 2
    smaller_area = math.pi * min(r1s, r2s)
    return intersection / smaller_area


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
# STEP 6: Draw results
# ─────────────────────────────────────────────────────────────────────────────
def draw_detections(frame: np.ndarray, balls: list[tuple[int, int, int]]) -> np.ndarray:
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


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────
def detect_billiard_balls(
    frame: np.ndarray,
    debug: bool = False,
) -> tuple[np.ndarray, list[tuple[int, int, int]]]:
    """
    Detect billiard balls in a BGR frame.

    Returns
    -------
    output_frame : BGR image with bounding boxes
    balls        : list of (x, y, radius)
    """
    # 1. Adaptive felt segmentation
    table_mask, felt_hsv = find_table_mask(frame)

    # 2. Scale-aware radius bounds
    min_r, max_r = estimate_ball_radius(table_mask)

    # 3. Grey image restricted to felt only
    gray       = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_table = cv2.bitwise_and(gray, gray, mask=table_mask)

    # 4. Multi-pass Hough
    candidates = multi_pass_hough(gray_table, table_mask, min_r, max_r)
    candidates = watershed_split_clusters(candidates, gray_table, table_mask, min_r, max_r)

    # 5. Verify + NMS
    balls = verify_and_nms(candidates, frame, table_mask, felt_hsv, min_r)

    if debug:
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        axes[0].imshow(table_mask,  cmap="gray");  axes[0].set_title("Table Mask (convex hull)")
        axes[1].imshow(gray_table,  cmap="gray");  axes[1].set_title("Gray (felt only)")
        axes[2].imshow(cv2.cvtColor(draw_detections(frame, balls), cv2.COLOR_BGR2RGB))
        axes[2].set_title(f"Detections ({len(balls)} balls)")
        for ax in axes: ax.axis("off")
        plt.tight_layout()
        plt.show()

    return draw_detections(frame, balls), balls


# ─────────────────────────────────────────────────────────────────────────────
# BATCH RUNNER
# ─────────────────────────────────────────────────────────────────────────────
def run_on_directory(image_dir: str, show: bool = True):
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
        # Notebook-style usage
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
            plt.axis("off")
            plt.show()
