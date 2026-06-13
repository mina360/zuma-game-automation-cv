import cv2
import numpy as np
import math


COLORS = {
    "RED": [
        (np.array([0,   150, 120]), np.array([10,  255, 255])),
        (np.array([170, 150, 120]), np.array([180, 255, 255]))
    ],
    "YELLOW": [(np.array([18, 120, 120]), np.array([35, 255, 255]))],
    "GREEN":  [(np.array([40,  80,  80]), np.array([80, 255, 255]))],
    "BLUE":   [(np.array([90,  80,  80]), np.array([130,255, 255]))],
    "PURPLE": [(np.array([125, 80, 80]), np.array([155,255,255]))],
}


def get_mouse_pos():
    try:
        import pyautogui
        p = pyautogui.position()
        return int(p.x), int(p.y)
    except Exception:
        try:
            import ctypes
            class POINT(ctypes.Structure):
                _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]
            pt = POINT()
            ctypes.windll.user32.GetCursorPos(ctypes.byref(pt))
            return int(pt.x), int(pt.y)
        except Exception:
            return None


def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def dist2(a, b):
    return (a[0] - b[0])**2 + (a[1] - b[1])**2


def _circle_mask(h, w, cx, cy, r):
    m = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(m, (cx, cy), r, 255, -1)
    return m


def classify_by_ranges(hsv_roi):
    counts = {}
    best_color = None
    best_count = 0

    for name, ranges in COLORS.items():
        mask = np.zeros(hsv_roi.shape[:2], dtype=np.uint8)
        for lo, hi in ranges:
            mask |= cv2.inRange(hsv_roi, lo, hi)
        cnt = int(cv2.countNonZero(mask))
        counts[name] = cnt
        if cnt > best_count:
            best_count = cnt
            best_color = name

    if best_count == 0:
        return None, counts

    return best_color, counts


def pick_color_at_point(frame_bgr, center_xy, radius):
    H, W = frame_bgr.shape[:2]
    cx, cy = int(round(center_xy[0])), int(round(center_xy[1]))
    if cx < 0 or cy < 0 or cx >= W or cy >= H:
        return None

    x0 = clamp(cx - radius, 0, W - 1)
    y0 = clamp(cy - radius, 0, H - 1)
    x1 = clamp(cx + radius, 0, W - 1)
    y1 = clamp(cy + radius, 0, H - 1)

    roi = frame_bgr[y0:y1 + 1, x0:x1 + 1]
    if roi.size == 0:
        return None

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    cm = _circle_mask(hsv.shape[0], hsv.shape[1], cx - x0, cy - y0, radius)

    hsv2 = hsv.copy()
    hsv2[cm == 0] = (0, 0, 0)

    c, counts = classify_by_ranges(hsv2)
    if c is None:
        return None

    return {
        "color": c,
        "center": (float(cx), float(cy)),
        "radius": int(radius),
        "counts": counts
    }


def detect_two_nearest_balls(frame_bgr, frog_box, expand_ratio=0.55):
    x, y, w, h = frog_box
    H, W = frame_bgr.shape[:2]

    ex = int(w * expand_ratio)
    ey = int(h * expand_ratio)

    rx0 = clamp(x - ex, 0, W - 1)
    ry0 = clamp(y - ey, 0, H - 1)
    rx1 = clamp(x + w + ex, 0, W - 1)
    ry1 = clamp(y + h + ey, 0, H - 1)

    roi = frame_bgr[ry0:ry1 + 1, rx0:rx1 + 1]
    if roi.size == 0:
        return []

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    any_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for name, ranges in COLORS.items():
        m = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lo, hi in ranges:
            m |= cv2.inRange(hsv, lo, hi)
        any_mask |= m

    kernel = np.ones((3, 3), np.uint8)
    any_mask = cv2.morphologyEx(any_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    any_mask = cv2.morphologyEx(any_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    num, labels, stats, cents = cv2.connectedComponentsWithStats(any_mask, connectivity=8)

    frog_center = (x + w / 2.0, y + h / 2.0)
    candidates = []

    for i in range(1, num):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < 60:
            continue

        cx, cy = cents[i]
        fx, fy = (rx0 + float(cx), ry0 + float(cy))

        r = int(max(6, min(22, math.sqrt(area / math.pi))))

        if dist2((fx, fy), frog_center) > (max(w, h) * 2.2) ** 2:
            continue

        info = pick_color_at_point(frame_bgr, (fx, fy), radius=max(6, r))
        if info is None:
            continue

        info["area"] = area
        info["d2frog"] = dist2((fx, fy), frog_center)
        candidates.append(info)

    candidates.sort(key=lambda c: (c["d2frog"], -c["area"]))

    picked = []
    for c in candidates:
        ok = True
        for p in picked:
            if dist2(c["center"], p["center"]) < (max(c["radius"], p["radius"]) * 1.6) ** 2:
                ok = False
                break
        if ok:
            picked.append(c)
        if len(picked) == 2:
            break

    return picked


def assign_current_next_by_mouse(balls, mouse_xy_global, window_bbox):
    if not balls:
        return None, None

    if mouse_xy_global is None or window_bbox is None:
        if len(balls) == 1:
            return balls[0], None
        return balls[0], balls[1]

    left, top, ww, hh = window_bbox
    mx, my = mouse_xy_global
    m = (float(mx - left), float(my - top))

    balls2 = sorted(balls, key=lambda b: dist2(b["center"], m))
    cur = balls2[0]
    nxt = balls2[1] if len(balls2) > 1 else None
    return cur, nxt

def draw_ball_info(debug_img, ball, label, color_bgr=(255, 255, 255)):
    if ball is None:
        return

    cx, cy = int(ball["center"][0]), int(ball["center"][1])
    r = int(ball["radius"])
    txt = f"{label}:{ball['color']}"

    cv2.circle(debug_img, (cx, cy), r, color_bgr, 2)
    cv2.circle(debug_img, (cx, cy), 2, color_bgr, -1)
    cv2.putText(
        debug_img, txt, (cx + 8, cy - 8),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_bgr, 2, cv2.LINE_AA
    )
