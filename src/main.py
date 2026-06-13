from balls_detector import default_cfg as default_balls_cfg, detect_chain_balls, order_chain
from ball_color_detector import detect_two_nearest_balls, get_mouse_pos, draw_ball_info
import os
import time
import cv2
import numpy as np
import keyboard
import math
from dataclasses import dataclass
from collections import deque

from window_detector import init_zuma_window, capture_zuma_frame

HAVE_INPUT = True
try:
    import pyautogui
    pyautogui.FAILSAFE = False
except Exception:
    HAVE_INPUT = False

HAVE_END = True
try:
    from end_detector import detect_ends
except Exception:
    HAVE_END = False

HAVE_FROG = True
try:
    from frog_detector import load_template_gray, load_template_edges
    from frog_fast import FrogFastDetector, FrogFastConfig, render_frog_roi_debug_image
except Exception:
    HAVE_FROG = False


HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "play_results", "trail")
ROOT_DIR = os.path.normpath(os.path.join(HERE, ".."))
TEMPLATES_DIR = os.path.join(ROOT_DIR, "templates")

WINDOW_TEMPLATE_PATH = os.path.join(TEMPLATES_DIR, "zuma_window_head.png")
WINDOW_MATCH_THRESHOLD = 0.62
WINDOW_FIXED_W = 808
WINDOW_FIXED_H = 636
LEVEL_TEMPLATE_PATH = os.path.join(TEMPLATES_DIR, "level.png")
LEVEL_MATCH_THRESHOLD = 0.65
LEVEL_DOWNSCALE = 0.70

POST_LEVEL_DELAY_SEC = 0.0

ROI_Y1_RATIO = 0.72
ROI_Y2_RATIO = 0.92
ROI_X1_RATIO = 0.38
ROI_X2_RATIO = 0.78
LEVEL_BOX_PAD = 10

BASELINE_SEC = 0.8
WAIT_TRAIL_TIMEOUT = 6.0
TRAIL_START_THR_PIXELS = 14
TRAIL_START_STREAK = 3
CAPTURE_MAX_SEC = 4.0

Y_H_LO, Y_S_LO, Y_V_LO = 18, 80, 170
Y_H_HI, Y_S_HI, Y_V_HI = 40, 255, 255

G_H_LO, G_S_LO, G_V_LO = 20, 100, 100
G_H_HI, G_S_HI, G_V_HI = 35, 255, 255

MOTION_THR = 18

USE_SPARK_FILTER = True
SPARK_MIN_AREA = 4
SPARK_MAX_AREA = 320
SPARK_MAX_WH = 60

PER_FRAME_DILATE = 1
PER_FRAME_CLOSE = 1

CROP_TOP = 35
CROP_BOTTOM = 0
CROP_LEFT = 8
CROP_RIGHT = 8

STOP_NEAR_END_ENABLED = True
STOP_BEFORE_END_MARGIN_PX = 15
END_NEAR_STREAK = 3
END_NEAR_MIN_NEWPIX = 2
EXTRA_AFTER_NEAR_END_SEC = 0.10

SKULL_MASK_PATH = os.path.join(TEMPLATES_DIR, "skull_mask.png")

FROG_TEMPLATE_GRAY = os.path.join(TEMPLATES_DIR, "gray.png")
FROG_MIN_SCORE = 0.60

DEBUG_WINDOW = "ZumaBot Debug"
DEBUG_W, DEBUG_H = 1200, 800
DEBUG_ENABLE = True
DEBUG_MAX_FPS = 12
DEBUG_SCALE = 0.90
DEBUG_DRAW_TOPK = 40
DETECT_CHAIN_EVERY = 2

PATH_FILTER_ENABLE = True
PATH_DIST_TOL_PX = 18.0
PATH_FALLBACK_USE_FIELD = True
PATH_FALLBACK_MULT = 1.6

GLOW_ENABLE = True
GLOW_INTERVAL_SEC = 0.30
GLOW_TTL_SEC = 1.20
GLOW_MATCH_DIST_PX = 28.0

GLOW_V_TH = 235
GLOW_S_TH = 70
GLOW_WHITE_RATIO_TH = 0.09
GLOW_VP90_TH = 245
GLOW_GLOBAL_FLASH_SKIP = 35.0

POST_SHOT_EFFECTS_ENABLE = True
POST_SHOT_REQUIRE_GLOW_SHOT = True
POST_SHOT_SCAN_WINDOW_SEC = 1.10
POST_SHOT_SCAN_INTERVAL_SEC = 0.12

EFFECT_ROI_X1_RATIO = 0.00
EFFECT_ROI_X2_RATIO = 0.70
EFFECT_ROI_Y1_RATIO = 0.00
EFFECT_ROI_Y2_RATIO = 0.28

EFFECT_ROI_DOWNSCALE = 0.80

EFFECT_TPL_BACK_PATH = os.path.join(TEMPLATES_DIR, "backwards_ball.png")
EFFECT_TPL_SLOW_PATH = os.path.join(TEMPLATES_DIR, "slowdown_ball.png")
EFFECT_TPL_ACCU_PATH = os.path.join(TEMPLATES_DIR, "accuracy_ball.png")

EFFECT_MATCH_THR = 0.68

EFFECT_EXTRA_WAIT_BACK_SEC = 1.10
EFFECT_EXTRA_WAIT_SLOW_SEC = 1.10
EFFECT_EXTRA_WAIT_ACCU_SEC = 0.60

EXPLOSION_FLASH_ENABLE = True
EXPLOSION_ROI_X1_RATIO = 0.18
EXPLOSION_ROI_X2_RATIO = 0.82
EXPLOSION_ROI_Y1_RATIO = 0.36
EXPLOSION_ROI_Y2_RATIO = 0.98
EXPLOSION_V_TH = 245
EXPLOSION_WHITE_RATIO_TH = 0.06
EXPLOSION_MEAN_V_DELTA_TH = 18.0
EFFECT_EXTRA_WAIT_EXPLOSION_SEC = 1.35

RISK_FIELD_R = 26
RISK_FALLBACK_DIST = 520.0

DANGER_HIGH_THR = 70.0
RUN_LEN_GOOD = 2

AUTO_ENABLED = True
SHOOT_COOLDOWN_SEC = 1
SWAP_COOLDOWN_SEC = 0.20
AFTER_SWAP_VERIFY_SEC = 0.06
AIM_OFFSET_PX = 0

RAYCAST_ENABLED = True
RAYCAST_RADIUS_PAD = 2.0
RAYCAST_EPS = 1e-6
RAYCAST_MAX_BALLS = 220
RAYCAST_DRAW = True


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def crop_client_area(frame_bgr):
    h, w = frame_bgr.shape[:2]
    y1 = clamp(CROP_TOP, 0, h - 1)
    y2 = clamp(h - CROP_BOTTOM, 1, h)
    x1 = clamp(CROP_LEFT, 0, w - 1)
    x2 = clamp(w - CROP_RIGHT, 1, w)
    return frame_bgr[y1:y2, x1:x2]


def build_gold_mask(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lo = np.array([G_H_LO, G_S_LO, G_V_LO], dtype=np.uint8)
    hi = np.array([G_H_HI, G_S_HI, G_V_HI], dtype=np.uint8)
    m = cv2.inRange(hsv, lo, hi)
    k = np.ones((3, 3), np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=1)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k, iterations=1)
    return m


def yellow_mask_hsv(client_bgr):
    hsv = cv2.cvtColor(client_bgr, cv2.COLOR_BGR2HSV)
    lo = np.array([Y_H_LO, Y_S_LO, Y_V_LO], dtype=np.uint8)
    hi = np.array([Y_H_HI, Y_S_HI, Y_V_HI], dtype=np.uint8)
    m = cv2.inRange(hsv, lo, hi)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones(
        (3, 3), np.uint8), iterations=1)
    return m


def motion_mask(prev_gray, cur_gray):
    diff = cv2.absdiff(cur_gray, prev_gray)
    _, m = cv2.threshold(diff, MOTION_THR, 255, cv2.THRESH_BINARY)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones(
        (3, 3), np.uint8), iterations=1)
    return m


def filter_small_components(mask_u8):
    num, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask_u8, connectivity=8)
    out = np.zeros_like(mask_u8)
    for i in range(1, num):
        area = int(stats[i, cv2.CC_STAT_AREA])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])
        if area < SPARK_MIN_AREA or area > SPARK_MAX_AREA:
            continue
        if w > SPARK_MAX_WH or h > SPARK_MAX_WH:
            continue
        out[labels == i] = 255
    return out


def extract_level_score_and_box(frame_bgr, template_mask_ds):
    h, w = frame_bgr.shape[:2]
    ry1 = clamp(int(h * ROI_Y1_RATIO), 0, h - 1)
    ry2 = clamp(int(h * ROI_Y2_RATIO), 1, h)
    rx1 = clamp(int(w * ROI_X1_RATIO), 0, w - 1)
    rx2 = clamp(int(w * ROI_X2_RATIO), 1, w)

    roi = frame_bgr[ry1:ry2, rx1:rx2]
    if roi.size == 0:
        return 0.0, None

    roi_mask = build_gold_mask(roi)
    ds = float(LEVEL_DOWNSCALE)
    roi_mask_ds = cv2.resize(roi_mask, None, fx=ds,
                             fy=ds, interpolation=cv2.INTER_NEAREST)

    th, tw = template_mask_ds.shape[:2]
    if roi_mask_ds.shape[0] < th or roi_mask_ds.shape[1] < tw:
        return 0.0, None

    res = cv2.matchTemplate(roi_mask_ds, template_mask_ds, cv2.TM_CCORR_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)

    score = float(max_val)
    if score < LEVEL_MATCH_THRESHOLD:
        return score, None

    inv = 1.0 / ds
    x_roi = int(round(max_loc[0] * inv))
    y_roi = int(round(max_loc[1] * inv))
    w_roi = int(round(tw * inv))
    h_roi = int(round(th * inv))

    x = rx1 + x_roi
    y = ry1 + y_roi
    w_box = w_roi
    h_box = h_roi

    pad = int(LEVEL_BOX_PAD)
    x -= pad
    y -= pad
    w_box += 2 * pad
    h_box += 2 * pad

    x = clamp(x, 0, w - 1)
    y = clamp(y, 0, h - 1)
    w_box = clamp(w_box, 1, w - x)
    h_box = clamp(h_box, 1, h - y)
    return score, (x, y, w_box, h_box)


def detect_end_fast(client_bgr):
    if not HAVE_END:
        return None, None
    try:
        boxes, scores, _fallback = detect_ends(client_bgr, SKULL_MASK_PATH)
        if not boxes or not scores:
            return None, None
        items = list(zip(boxes, scores))
        items.sort(key=lambda t: float(t[1]))
        (x, y, w, h), _s = items[0]
        cx = int(x + w / 2)
        cy = int(y + h / 2)
        r = int(0.55 * max(w, h))
        return (cx, cy), r
    except Exception:
        return None, None


def make_circle_mask(H, W, center_xy, radius):
    m = np.zeros((H, W), dtype=np.uint8)
    if center_xy is None or radius is None:
        return m
    cv2.circle(m, (int(center_xy[0]), int(center_xy[1])), int(radius), 255, -1)
    return m


def draw_box(img, box, color=(0, 255, 0), thickness=2, label=None):
    if box is None:
        return
    x, y, w, h = [int(v) for v in box]
    cv2.rectangle(img, (x, y), (x + w, y + h), color, thickness)
    if label:
        cv2.putText(img, label, (x, max(0, y - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)


def client_to_screen(ctx, pt_client):
    if pt_client is None:
        return None
    bbox = ctx.get("bbox") if isinstance(ctx, dict) else None
    if bbox is None:
        return None
    left, top, ww, hh = bbox
    sx = int(left + CROP_LEFT + float(pt_client[0]))
    sy = int(top + CROP_TOP + float(pt_client[1]))
    return (sx, sy)


def focus_window_once(ctx):
    if not HAVE_INPUT:
        return
    bbox = ctx.get("bbox") if isinstance(ctx, dict) else None
    if bbox is None:
        return
    left, top, ww, hh = bbox
    cx = int(left + ww * 0.5)
    cy = int(top + hh * 0.5)
    try:
        pyautogui.click(cx, cy, button="left")
        time.sleep(0.03)
    except Exception:
        pass


def right_click_swap(ctx, frog_center_client):
    if not HAVE_INPUT:
        return False
    pt = client_to_screen(ctx, frog_center_client)
    if pt is None:
        return False
    try:
        pyautogui.click(pt[0], pt[1], button="right")
        return True
    except Exception:
        return False


def aim_and_shoot(ctx, frog_center_client, target_client):
    if not HAVE_INPUT:
        return False
    pt = client_to_screen(ctx, target_client)
    if pt is None:
        return False
    try:
        pyautogui.moveTo(pt[0], pt[1])
        pyautogui.click(pt[0], pt[1], button="left")
        return True
    except Exception:
        return False


def build_risk_field(time_map_f32, capture_dur, radius=26):
    if capture_dur <= 1e-6:
        capture_dur = 1e-6
    risk = np.clip(100.0 * (time_map_f32 / float(capture_dur)),
                   0.0, 100.0).astype(np.uint8)
    k = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    field = cv2.dilate(risk, k, iterations=1)
    return risk, field


def safe_color_eq(a, b):
    if a is None or b is None:
        return False
    return str(a).lower() == str(b).lower()


def ball_risk_from_field(ball_center, risk_field_u8):
    x, y = int(ball_center[0]), int(ball_center[1])
    H, W = risk_field_u8.shape[:2]
    if x < 0 or x >= W or y < 0 or y >= H:
        return None
    v = int(risk_field_u8[y, x])
    if v <= 0:
        return None
    return float(v)


def fallback_risk_by_end(ball_center, end_center):
    if end_center is None:
        return 0.0
    dx = float(ball_center[0] - end_center[0])
    dy = float(ball_center[1] - end_center[1])
    d = (dx * dx + dy * dy) ** 0.5
    r = 100.0 * (1.0 - (d / float(RISK_FALLBACK_DIST)))
    return float(np.clip(r, 0.0, 100.0))


def _safe_get_dt(path_dt, x, y):
    H, W = path_dt.shape[:2]
    if x < 0 or x >= W or y < 0 or y >= H:
        return 1e9
    return float(path_dt[y, x])


def filter_balls_on_path(balls, path_dt, risk_field_u8=None, tol_base_px=18.0, fallback_use_field=True, fallback_mult=1.6):
    if not balls:
        return balls
    rs = [int(b.get("r", 0)) for b in balls if b.get("r", 0) > 0]
    med_r = float(np.median(rs)) if rs else 0.0
    tol_px = float(max(tol_base_px, 1.15 * med_r))
    out = []
    for b in balls:
        cx, cy = int(b["center"][0]), int(b["center"][1])
        d = _safe_get_dt(path_dt, cx, cy)
        ok = (d <= tol_px)
        if (not ok) and fallback_use_field and (risk_field_u8 is not None):
            if int(risk_field_u8[cy, cx]) > 0 and d <= (tol_px * float(fallback_mult)):
                ok = True
        if ok:
            out.append(b)
    return out


class GlowDetector:
    def __init__(self, interval_sec=0.30, ttl_sec=1.20, match_dist_px=28.0):
        self.interval_sec = float(interval_sec)
        self.ttl_sec = float(ttl_sec)
        self.match_dist_px = float(match_dist_px)
        self.last_check_t = 0.0
        self.prev_global_v = None
        self.prev_on_centers = []
        self.active = []

    @staticmethod
    def _dist2(a, b):
        dx = float(a[0] - b[0])
        dy = float(a[1] - b[1])
        return dx * dx + dy * dy

    def _upsert_active(self, cand, now):
        for ex in self.active:
            if self._dist2(ex["center"], cand["center"]) <= (self.match_dist_px ** 2):
                ex["center"] = cand["center"]
                ex["r"] = cand["r"]
                ex["score"] = cand["score"]
                ex["white_ratio"] = cand["white_ratio"]
                ex["vp90"] = cand["vp90"]
                ex["expires"] = now + self.ttl_sec
                return
        c2 = dict(cand)
        c2["expires"] = now + self.ttl_sec
        self.active.append(c2)

    def get_active(self, now):
        self.active = [c for c in self.active if c.get("expires", 0.0) > now]
        return list(self.active)

    def update(self, client_bgr, chain, now, dbg=None):
        self.get_active(now)
        if (now - self.last_check_t) < self.interval_sec:
            return self.active
        self.last_check_t = now
        if client_bgr is None or client_bgr.size == 0 or not chain:
            return self.active

        hsv = cv2.cvtColor(client_bgr, cv2.COLOR_BGR2HSV)
        global_v = float(np.mean(hsv[:, :, 2]))
        if self.prev_global_v is not None:
            if abs(global_v - float(self.prev_global_v)) > float(GLOW_GLOBAL_FLASH_SKIP):
                self.prev_global_v = global_v
                return self.active
        self.prev_global_v = global_v

        H, W = hsv.shape[:2]
        cand_now = []

        for b in chain:
            cx, cy = int(b["center"][0]), int(b["center"][1])
            r = int(b.get("r", 14))
            rad = max(8, int(0.95 * r))

            x1 = max(0, cx - rad)
            y1 = max(0, cy - rad)
            x2 = min(W, cx + rad + 1)
            y2 = min(H, cy + rad + 1)
            if (x2 - x1) < 8 or (y2 - y1) < 8:
                continue

            roi = hsv[y1:y2, x1:x2]
            hh, ww = roi.shape[:2]

            mask = np.zeros((hh, ww), dtype=np.uint8)
            cv2.circle(mask, (cx - x1, cy - y1),
                       max(6, int(0.85 * rad)), 255, -1)

            v = roi[:, :, 2][mask > 0]
            s = roi[:, :, 1][mask > 0]
            if v.size < 30:
                continue

            vp90 = float(np.percentile(v, 90))
            white_ratio = float(
                np.mean((v >= int(GLOW_V_TH)) & (s <= int(GLOW_S_TH))))
            score = 0.55 * (vp90 / 255.0) + 0.45 * white_ratio

            is_glow = (white_ratio >= float(GLOW_WHITE_RATIO_TH)) or (vp90 >= float(
                GLOW_VP90_TH) and white_ratio >= (0.5 * float(GLOW_WHITE_RATIO_TH)))
            if is_glow:
                cand_now.append({
                    "center": (cx, cy),
                    "r": r,
                    "score": float(score),
                    "white_ratio": float(white_ratio),
                    "vp90": float(vp90),
                    "color": b.get("color", None),
                })

        new = []
        for c in cand_now:
            if not any(self._dist2(c["center"], p) <= (self.match_dist_px ** 2) for p in self.prev_on_centers):
                new.append(c)

        if new:
            preview = ", ".join(
                [f"{c['center']} wr={c['white_ratio']:.2f}" for c in new[:6]])
            print(f"[GLOW] Found {len(new)} glowing balls: {preview}")

        self.prev_on_centers = [c["center"] for c in cand_now]

        for c in cand_now:
            self._upsert_active(c, now)
            if dbg is not None and DEBUG_ENABLE:
                cv2.circle(dbg, c["center"], int(
                    1.20 * c["r"]), (255, 0, 255), 2)
                cv2.putText(dbg, "GLOW", (c["center"][0] + 6, c["center"][1] - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 255), 1, cv2.LINE_AA)

        return self.active


def build_runs(chain, risks):
    runs = []
    if not chain:
        return runs
    i = 0
    n = len(chain)
    while i < n:
        c = chain[i].get("color", None)
        j = i
        while j < n and safe_color_eq(chain[j].get("color", None), c):
            j += 1
        seg = chain[i:j]
        seg_risks = risks[i:j]
        mid = seg[len(seg) // 2]
        run = {
            "color": c,
            "i0": i,
            "i1": j - 1,
            "len": (j - i),
            "center": mid["center"],
            "risk_max": float(np.max(seg_risks)) if seg_risks else 0.0,
            "risk_min": float(np.min(seg_risks)) if seg_risks else 0.0,
            "risk_mean": float(np.mean(seg_risks)) if seg_risks else 0.0,
        }
        runs.append(run)
        i = j
    return runs


def _ray_circle_first_t(origin_xy, dir_unit_xy, center_xy, radius):
    ox, oy = float(origin_xy[0]), float(origin_xy[1])
    dx, dy = float(dir_unit_xy[0]), float(dir_unit_xy[1])
    cx, cy = float(center_xy[0]), float(center_xy[1])
    R = float(radius)

    ocx = ox - cx
    ocy = oy - cy

    b = 2.0 * (dx * ocx + dy * ocy)
    c = (ocx * ocx + ocy * ocy) - (R * R)
    disc = b * b - 4.0 * c
    if disc < 0.0:
        return None

    s = float(disc ** 0.5)
    t1 = (-b - s) * 0.5
    t2 = (-b + s) * 0.5

    if t1 >= 0.0:
        return t1
    if t2 >= 0.0:
        return t2
    return None


def raycast_first_hit_index(frog_center, target_pt, chain):
    if frog_center is None or target_pt is None or not chain:
        return None, None, None

    ox, oy = float(frog_center[0]), float(frog_center[1])
    tx, ty = float(target_pt[0]), float(target_pt[1])

    vx = tx - ox
    vy = ty - oy
    dist2 = vx * vx + vy * vy
    distv = dist2 ** 0.5
    if distv <= 1e-6:
        return None, None, None

    inv = 1.0 / distv
    dx = vx * inv
    dy = vy * inv

    best_t = None
    best_i = None

    n = min(len(chain), int(RAYCAST_MAX_BALLS))
    for i in range(n):
        b = chain[i]
        cxy = b.get("center", None)
        if cxy is None:
            continue
        r = float(b.get("r", 18))
        r2 = r + float(RAYCAST_RADIUS_PAD)

        t = _ray_circle_first_t((ox, oy), (dx, dy), cxy, r2)
        if t is None:
            continue
        if t > distv + 0.5:
            continue

        if best_t is None or t < best_t:
            best_t = t
            best_i = i

    if best_i is None:
        return None, None, None

    hx = ox + dx * float(best_t)
    hy = oy + dy * float(best_t)
    return int(best_i), float(best_t), (hx, hy)


def candidate_visible_by_raycast(frog_center, chain, target_pt, span_i0_i1):
    idx, t, hit_pt = raycast_first_hit_index(frog_center, target_pt, chain)
    if idx is None:
        return True, None, None
    i0, i1 = span_i0_i1
    ok = (i0 <= idx <= i1)
    return bool(ok), idx, hit_pt


def enumerate_candidates(runs, chain, risks, cur_color, next_color):
    cands = []
    if not chain:
        return cands

    max_risk = float(np.max(risks)) if risks else 0.0
    defend = max_risk >= DANGER_HIGH_THR

    def score_run(tag, r):
        base = r["risk_max"] if defend else (100.0 - r["risk_min"])
        bonus = 30.0
        if r["len"] >= 3:
            bonus += 10.0
        if tag == "cur":
            bonus += 3.0
        return float(base + bonus)

    def score_single(tag, i):
        rr = float(risks[i]) if risks else 0.0
        base = rr if defend else (100.0 - rr)
        bonus = 2.0 if tag == "cur" else 0.0
        return float(base + bonus)

    good_runs = [r for r in runs if r["len"] >= RUN_LEN_GOOD]
    for r in good_runs:
        if safe_color_eq(r["color"], cur_color):
            cands.append({"action": "shoot", "target_pt": r["center"], "target_color": r["color"], "span": (
                int(r["i0"]), int(r["i1"])), "why": "run_cur", "score": score_run("cur", r)})
        elif safe_color_eq(r["color"], next_color):
            cands.append({"action": "swap_shoot", "target_pt": r["center"], "target_color": r["color"], "span": (
                int(r["i0"]), int(r["i1"])), "why": "run_next", "score": score_run("next", r)})

    for i, b in enumerate(chain):
        c = b.get("color", None)
        if safe_color_eq(c, cur_color):
            cands.append({"action": "shoot", "target_pt": b["center"], "target_color": c, "span": (
                int(i), int(i)), "why": "single_cur", "score": score_single("cur", i)})
        elif safe_color_eq(c, next_color):
            cands.append({"action": "swap_shoot", "target_pt": b["center"], "target_color": c, "span": (
                int(i), int(i)), "why": "single_next", "score": score_single("next", i)})

    idx = int(np.argmax(risks)) if (risks and defend) else (
        int(np.argmin(risks)) if risks else 0)
    cands.append({"action": "shoot", "target_pt": chain[idx]["center"], "target_color": chain[idx].get(
        "color", None), "span": (int(idx), int(idx)), "why": "fallback_any", "score": -9999.0})

    cands.sort(key=lambda d: float(d.get("score", 0.0)), reverse=True)
    return cands


def choose_target_with_raycast(frog_center, runs, chain, risks, cur_color, next_color):
    if not chain:
        return None, None, "no_chain", None, (0, -1), False, None, None

    cands = enumerate_candidates(runs, chain, risks, cur_color, next_color)
    if not cands:
        return None, None, "no_candidates", None, (0, -1), False, None, None

    if (not RAYCAST_ENABLED) or (frog_center is None):
        best = cands[0]
        return best["action"], best["target_pt"], best["why"], best["target_color"], best["span"], False, None, None

    for cand in cands:
        visible, hit_idx, hit_pt = candidate_visible_by_raycast(
            frog_center, chain, cand["target_pt"], cand["span"])
        if visible:
            return cand["action"], cand["target_pt"], cand["why"], cand["target_color"], cand["span"], False, hit_idx, hit_pt

    best = cands[0]
    visible, hit_idx, hit_pt = candidate_visible_by_raycast(
        frog_center, chain, best["target_pt"], best["span"])
    return best["action"], best["target_pt"], best["why"] + "_OCCLUDED", best["target_color"], best["span"], True, hit_idx, hit_pt


@dataclass
class BotStrategyConfig:
    HIT_MARGIN: float = 2.0
    EXPECT_HIT_TOL: int = 2
    MIN_CLEARANCE_PX: float = 3.0
    MIN_HIT_DEPTH: float = 0.35
    REQUIRE_FEASIBLE_SHOT: bool = True
    FEASIBLE_MIN_N: int = 10
    RETRY_EXPECTED_AIM: bool = True
    DANGER_MED_MULT: float = 8.5
    DANGER_HIGH_MULT: float = 6.0
    DANGER_CRIT_MULT: float = 4.4
    HOLE_SHOT_FORBID_PAD: int = 24
    W_REMOVE: float = 14.0
    W_CASCADE: float = 11.0
    W_GROUP: float = 2.4
    W_EARLY: float = 9.0
    W_SAFETY: float = 14.0
    W_SETUP2: float = 8.0
    W_AIM_EASE: float = 8.0
    W_MIX_PEN: float = 7.0
    PEN_NO_REMOVE: float = 18.0
    MIN_SHOT_SCORE: float = 8.0
    MIN_SHOT_SCORE_SMALL: float = 4.0


BOT_CFG = BotStrategyConfig()


def dist(p1, p2):
    dx = float(p1[0] - p2[0])
    dy = float(p1[1] - p2[1])
    return float((dx * dx + dy * dy) ** 0.5)


def simulate_insert(colors, pos, c):
    arr = colors[:pos] + [c] + colors[pos:]
    removed = 0
    casc = 0
    idx = pos

    while True:
        if not arr:
            break
        idx = clamp(idx, 0, len(arr) - 1)
        L = idx
        R = idx
        while L - 1 >= 0 and arr[L - 1] == arr[idx]:
            L -= 1
        while R + 1 < len(arr) and arr[R + 1] == arr[idx]:
            R += 1

        run = R - L + 1
        if run < 3:
            break

        removed += run
        casc += 1
        arr = arr[:L] + arr[R + 1:]
        idx = L - 1

    return removed, casc


def run_length_after_insert(colors, pos, shot_color):
    arr = colors[:pos] + [shot_color] + colors[pos:]
    idx = pos
    L = idx
    R = idx
    while L - 1 >= 0 and arr[L - 1] == shot_color:
        L -= 1
    while R + 1 < len(arr) and arr[R + 1] == shot_color:
        R += 1
    return (R - L + 1)


def mixedness(colors, pos, win):
    n = len(colors)
    if n == 0:
        return 0.0
    half = win // 2
    L = clamp(pos - half, 0, n - 1)
    R = clamp(pos + half, 0, n - 1)
    window = colors[L:R + 1]
    if not window:
        return 0.0
    distinct = len(set(window))
    transitions = sum(1 for i in range(1, len(window))
                      if window[i] != window[i - 1])
    d_norm = (distinct - 1) / max(1, (win - 1))
    t_norm = transitions / max(1, (len(window) - 1))
    return 0.6 * d_norm + 0.4 * t_norm


def target_point_from_pos(chain, pos):
    n = len(chain)
    if n == 0:
        return None
    if pos <= 0:
        return chain[0]["center"]
    if pos >= n:
        return chain[-1]["center"]
    x1, y1 = chain[pos - 1]["center"]
    x2, y2 = chain[pos]["center"]
    return (int((x1 + x2) / 2), int((y1 + y2) / 2))


def safety_score_from_pos(cumdist, pos):
    if not cumdist:
        return 0.5
    idx = clamp(pos, 0, len(cumdist) - 1)
    maxd = max(1e-6, float(cumdist[-1]))
    return float(1.0 - clamp(float(cumdist[idx]) / maxd, 0.0, 1.0))


def early_score_from_pos(cumdist, pos):
    if not cumdist:
        return 0.5
    idx = clamp(pos, 0, len(cumdist) - 1)
    maxd = max(1e-6, float(cumdist[-1]))
    return float(clamp(float(cumdist[idx]) / maxd, 0.0, 1.0))


def estimate_ball_radius(chain):
    if not chain:
        return None
    rs = [float(b.get("r", 0.0))
          for b in chain if float(b.get("r", 0.0)) > 1.0]
    if not rs:
        return None
    rs.sort()
    return float(rs[len(rs) // 2])


def danger_metrics(chain, hole_center, hole_r):
    if not chain or hole_center is None or hole_r is None:
        return 0.0, 0

    br = estimate_ball_radius(chain)
    if br is None:
        br = 18.0

    head = chain[0]["center"]
    d = dist(head, hole_center) - float(hole_r)

    if d <= BOT_CFG.DANGER_CRIT_MULT * br:
        level = 3
    elif d <= BOT_CFG.DANGER_HIGH_MULT * br:
        level = 2
    elif d <= BOT_CFG.DANGER_MED_MULT * br:
        level = 1
    else:
        level = 0

    scale = BOT_CFG.DANGER_MED_MULT * br
    danger = float(clamp(1.0 - (d / (scale + 1e-6)), 0.0, 1.0))
    return danger, level


def _ray_circle_intersection_t(ox, oy, dx, dy, cx, cy, r):
    vx = ox - cx
    vy = oy - cy
    b = 2.0 * (dx * vx + dy * vy)
    c = vx * vx + vy * vy - r * r
    disc = b * b - 4.0 * c
    if disc < 0.0:
        return None
    s = float(disc ** 0.5)
    t1 = (-b - s) * 0.5
    t2 = (-b + s) * 0.5
    if t1 >= 0.0:
        return t1
    if t2 >= 0.0:
        return t2
    return None


def raycast_first_hit_with_depth(frog, aim_point, chain):
    fx, fy = float(frog[0]), float(frog[1])
    ax, ay = float(aim_point[0]), float(aim_point[1])
    dx, dy = ax - fx, ay - fy
    norm = math.hypot(dx, dy)
    if norm < 1.0:
        return None, None, None, None
    dx, dy = dx / norm, dy / norm

    best_t = 1e18
    best_i = None
    best_depth = None

    for i, b in enumerate(chain):
        cx, cy = b["center"]
        r = float(b.get("r", 0.0)) + BOT_CFG.HIT_MARGIN
        t = _ray_circle_intersection_t(fx, fy, dx, dy, float(cx), float(cy), r)
        if t is None or t <= 0:
            continue

        px = fx + dx * float(t)
        py = fy + dy * float(t)
        d = math.hypot(float(cx) - px, float(cy) - py)
        if d <= r and t < best_t:
            best_t = float(t)
            best_i = int(i)
            best_depth = float(clamp(1.0 - (d / (r + 1e-6)), 0.0, 1.0))

    return best_i, best_t if best_i is not None else None, best_depth, (dx, dy)


def ray_clearance_before_hit(frog, dir_xy, chain, hit_t, ignore_i):
    fx, fy = float(frog[0]), float(frog[1])
    dx, dy = float(dir_xy[0]), float(dir_xy[1])
    best = 1e9
    for i, b in enumerate(chain):
        if i == ignore_i:
            continue
        cx, cy = b["center"]
        vx, vy = float(cx) - fx, float(cy) - fy
        t = vx * dx + vy * dy
        if t <= 0 or t >= float(hit_t):
            continue
        px = fx + dx * float(t)
        py = fy + dy * float(t)
        d = math.hypot(float(cx) - px, float(cy) - py) - float(b.get("r", 0.0))
        if d < best:
            best = d
    return float(best if best < 1e9 else 999.0)


def _feasible_shot_check(frog_center, chain, aim_point, expected_i):
    hit_i, hit_t, hit_depth, dir_xy = raycast_first_hit_with_depth(
        frog_center, aim_point, chain)
    if hit_i is None or hit_t is None or hit_depth is None or dir_xy is None:
        return False, None, None, None, None

    if abs(int(hit_i) - int(expected_i)) > BOT_CFG.EXPECT_HIT_TOL:
        return False, hit_i, hit_t, hit_depth, dir_xy

    clearance = ray_clearance_before_hit(
        frog_center, dir_xy, chain, hit_t, hit_i)
    if clearance < BOT_CFG.MIN_CLEARANCE_PX or float(hit_depth) < BOT_CFG.MIN_HIT_DEPTH:
        return False, hit_i, hit_t, hit_depth, dir_xy

    return True, hit_i, hit_t, hit_depth, dir_xy


def evaluate_best_move(chain, cumdist, shot_color, hole_center, hole_forbid_r, frog_center, hole_r):
    colors = [b.get("color", None) for b in chain]
    n = len(colors)
    if n == 0 or shot_color is None:
        return None, None, None

    danger, danger_level_id = danger_metrics(chain, hole_center, hole_r)
    w_safety = BOT_CFG.W_SAFETY * (0.6 + 1.6 * danger)
    w_early = BOT_CFG.W_EARLY * (1.0 - 0.85 * danger)

    best_score = -1e18
    best_pos = None
    best_target = None
    best_dbg = None

    require_feasible = BOT_CFG.REQUIRE_FEASIBLE_SHOT and (
        n >= BOT_CFG.FEASIBLE_MIN_N)

    for pos in range(0, n + 1):
        target = target_point_from_pos(chain, pos)
        if target is None:
            continue

        if hole_center is not None and hole_forbid_r is not None:
            if dist(target, hole_center) <= float(hole_forbid_r):
                continue

        expected = int(clamp(pos, 0, n - 1))

        if require_feasible and frog_center is not None:
            ok, hit_i, hit_t, hit_depth, dir_xy = _feasible_shot_check(
                frog_center, chain, target, expected)
            if (not ok) and BOT_CFG.RETRY_EXPECTED_AIM:
                target2 = chain[expected]["center"]
                ok, hit_i, hit_t, hit_depth, dir_xy = _feasible_shot_check(
                    frog_center, chain, target2, expected)
                if not ok:
                    continue
                target = target2
        else:
            hit_i, hit_t, hit_depth, dir_xy = (None, None, None, None)

        removed, casc = simulate_insert(colors, pos, shot_color)
        run_after = run_length_after_insert(colors, pos, shot_color)
        safety = safety_score_from_pos(cumdist, pos)
        early = early_score_from_pos(cumdist, pos)
        mix = mixedness(colors, pos, win=7)

        if hit_i is None or hit_t is None or dir_xy is None or hit_depth is None:
            aim_ease = 0.0
            clearance_norm = 0.0
        else:
            clearance = ray_clearance_before_hit(
                frog_center, dir_xy, chain, hit_t, hit_i)
            clearance_norm = float(clamp(clearance / 18.0, 0.0, 1.0))
            aim_ease = 0.65 * float(hit_depth) + 0.35 * clearance_norm

        if danger_level_id >= 2 and removed < 3:
            if not (run_after >= 2 and safety >= 0.60):
                continue

        score = 0.0
        score += BOT_CFG.W_REMOVE * float(removed)
        score += BOT_CFG.W_CASCADE * float(max(0, casc - 1))
        score += BOT_CFG.W_GROUP * float(run_after)

        if removed >= 3:
            score += w_early * float(early)
            score += w_safety * float(safety)
        else:
            if run_after >= 2:
                score += BOT_CFG.W_SETUP2 * float(safety)

        score += BOT_CFG.W_AIM_EASE * float(aim_ease)
        score -= BOT_CFG.W_MIX_PEN * float(mix)

        if removed < 3 and run_after < 2:
            score -= BOT_CFG.PEN_NO_REMOVE

        if (run_after % 2) == 1:
            score += 0.35

        if score > best_score:
            best_score = float(score)
            best_pos = int(pos)
            best_target = target
            best_dbg = {
                "danger": float(danger),
                "dlevel": int(danger_level_id),
                "safety": float(safety),
                "early": float(early),
                "removed": int(removed),
                "casc": int(casc),
                "run": int(run_after),
                "mix": float(mix),
                "aim": float(aim_ease),
                "clr": float(clearance_norm),
                "score": float(score),
            }

    min_score = BOT_CFG.MIN_SHOT_SCORE if n >= 10 else BOT_CFG.MIN_SHOT_SCORE_SMALL
    if best_pos is None or best_score < float(min_score):
        return None, None, best_dbg

    return best_pos, best_target, best_dbg


@dataclass
class TargetPlan:
    source: str
    action: str
    why: str
    score: float
    target_color: str | None
    span: tuple
    decide_t: float
    aim_kind: str = "ball"
    ref_idx: int | None = None
    pos: int | None = None


class EffectWordDetector:
    def __init__(self, back_path, slow_path, accu_path, match_thr=0.68, scales=(0.90, 1.00, 1.10)):
        self.match_thr = float(match_thr)
        self.scales = tuple(float(s) for s in scales)
        self.templates = {}
        self._load_one("BACK", back_path)
        self._load_one("SLOW", slow_path)
        self._load_one("ACCU", accu_path)

    @staticmethod
    def _resolve_path(p):
        if p is None:
            return ""
        here = os.path.dirname(os.path.abspath(__file__))
        cand = os.path.join(here, p)
        if os.path.exists(cand):
            return cand
        return p

    @staticmethod
    def _prep_gray(img_bgr):
        g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        m = (g > 20).astype(np.uint8)
        if int(m.max()) == 0:
            return g
        ys, xs = np.where(m > 0)
        y0, y1 = int(ys.min()), int(ys.max())
        x0, x1 = int(xs.min()), int(xs.max())
        return g[y0:y1 + 1, x0:x1 + 1]

    @staticmethod
    def _to_edge(g):
        g2 = cv2.GaussianBlur(g, (3, 3), 0)
        e = cv2.Canny(g2, 50, 150)
        e = cv2.dilate(e, np.ones((2, 2), np.uint8), iterations=1)
        return e

    def _load_one(self, name, path):
        path2 = self._resolve_path(path)
        img = cv2.imread(path2)
        if img is None:
            print(f"[EFFECT] Failed to load template '{name}' from: {path2}")
            self.templates[name] = []
            return
        g = self._prep_gray(img)
        tpl_list = []
        for s in self.scales:
            if abs(s - 1.0) < 1e-6:
                gs = g
            else:
                gs = cv2.resize(g, None, fx=s, fy=s,
                                interpolation=cv2.INTER_AREA)
            e = self._to_edge(gs)
            h, w = e.shape[:2]
            if h >= 4 and w >= 4:
                tpl_list.append((e, w, h, s))
        self.templates[name] = tpl_list

    def detect(self, roi_bgr):
        if roi_bgr is None or roi_bgr.size == 0:
            return []
        g = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        e = self._to_edge(g)
        H, W = e.shape[:2]
        dets = []

        for name, tpls in self.templates.items():
            best_score = -1.0
            best_loc = None
            best_size = None
            for (tpl_e, tw, th, _s) in tpls:
                if H < th or W < tw:
                    continue
                res = cv2.matchTemplate(e, tpl_e, cv2.TM_CCOEFF_NORMED)
                _, mx, _, loc = cv2.minMaxLoc(res)
                mx = float(mx)
                if mx > best_score:
                    best_score = mx
                    best_loc = loc
                    best_size = (tw, th)
            if best_score >= self.match_thr and best_loc is not None and best_size is not None:
                dets.append({"name": name, "score": best_score,
                            "loc": best_loc, "size": best_size})
        return dets


class PostShotEffectMonitor:
    def __init__(self):
        self.detector = EffectWordDetector(
            EFFECT_TPL_BACK_PATH, EFFECT_TPL_SLOW_PATH, EFFECT_TPL_ACCU_PATH, match_thr=EFFECT_MATCH_THR)
        self.active_until = 0.0
        self.last_scan_t = 0.0
        self.armed = False
        self.found = set()
        self.shot_t = 0.0
        self.flash_ema_v = None
        self.last_events = []

    @staticmethod
    def _roi_from_ratios(img_shape, x1r, x2r, y1r, y2r):
        H, W = img_shape[:2]
        x1 = int(clamp(int(W * float(x1r)), 0, W - 1))
        x2 = int(clamp(int(W * float(x2r)), x1 + 1, W))
        y1 = int(clamp(int(H * float(y1r)), 0, H - 1))
        y2 = int(clamp(int(H * float(y2r)), y1 + 1, H))
        return x1, y1, x2, y2

    @staticmethod
    def _near_glow(shot_pt, special_candidates):
        if shot_pt is None or not special_candidates:
            return False
        sx, sy = float(shot_pt[0]), float(shot_pt[1])
        for c in special_candidates:
            cc = c.get("center", None)
            if cc is None:
                continue
            cx, cy = float(cc[0]), float(cc[1])
            rr = float(c.get("r", 14))
            d = ((sx - cx) ** 2 + (sy - cy) ** 2) ** 0.5
            if d <= (rr + 14.0):
                return True
        return False

    def notify_shot(self, now, shot_pt, special_candidates):
        if not POST_SHOT_EFFECTS_ENABLE:
            return
        arm = True
        if POST_SHOT_REQUIRE_GLOW_SHOT:
            arm = self._near_glow(shot_pt, special_candidates)
        if not arm:
            self.armed = False
            self.active_until = 0.0
            self.found.clear()
            self.last_events = []
            return

        self.armed = True
        self.shot_t = float(now)
        self.active_until = float(now) + float(POST_SHOT_SCAN_WINDOW_SEC)
        self.last_scan_t = 0.0
        self.found.clear()
        self.last_events = []
        self.flash_ema_v = None

    def _event_extra_wait(self, name):
        if name == "BACK":
            return float(EFFECT_EXTRA_WAIT_BACK_SEC)
        if name == "SLOW":
            return float(EFFECT_EXTRA_WAIT_SLOW_SEC)
        if name == "ACCU":
            return float(EFFECT_EXTRA_WAIT_ACCU_SEC)
        if name == "EXPLOSION":
            return float(EFFECT_EXTRA_WAIT_EXPLOSION_SEC)
        return 0.0

    def _detect_explosion_flash(self, client_bgr):
        if not EXPLOSION_FLASH_ENABLE:
            return None
        x1, y1, x2, y2 = self._roi_from_ratios(
            client_bgr.shape, EXPLOSION_ROI_X1_RATIO, EXPLOSION_ROI_X2_RATIO, EXPLOSION_ROI_Y1_RATIO, EXPLOSION_ROI_Y2_RATIO)
        roi = client_bgr[y1:y2, x1:x2]
        if roi.size == 0:
            return None
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        v = hsv[:, :, 2].astype(np.float32)
        mean_v = float(v.mean())
        white_ratio = float((v >= float(EXPLOSION_V_TH)).mean())

        if self.flash_ema_v is None:
            self.flash_ema_v = mean_v
            return None

        self.flash_ema_v = 0.90 * float(self.flash_ema_v) + 0.10 * mean_v
        surge = mean_v - float(self.flash_ema_v)

        if white_ratio >= float(EXPLOSION_WHITE_RATIO_TH) and surge >= float(EXPLOSION_MEAN_V_DELTA_TH):
            return {"name": "EXPLOSION", "extra": self._event_extra_wait("EXPLOSION"), "roi": (x1, y1, x2, y2), "white_ratio": white_ratio, "mean_v": mean_v, "surge": surge}
        return None

    def update(self, client_bgr, now, dbg=None):
        self.last_events = []
        if (not POST_SHOT_EFFECTS_ENABLE) or (not self.armed) or (float(now) > float(self.active_until)):
            return []
        if (float(now) - float(self.last_scan_t)) < float(POST_SHOT_SCAN_INTERVAL_SEC):
            return []
        self.last_scan_t = float(now)

        events = []

        x1, y1, x2, y2 = self._roi_from_ratios(
            client_bgr.shape, EFFECT_ROI_X1_RATIO, EFFECT_ROI_X2_RATIO, EFFECT_ROI_Y1_RATIO, EFFECT_ROI_Y2_RATIO)
        roi = client_bgr[y1:y2, x1:x2]
        if roi.size != 0:
            roi2 = roi
            if EFFECT_ROI_DOWNSCALE != 1.0:
                roi2 = cv2.resize(roi, None, fx=float(EFFECT_ROI_DOWNSCALE), fy=float(
                    EFFECT_ROI_DOWNSCALE), interpolation=cv2.INTER_AREA)
            dets = self.detector.detect(roi2)
            for d in dets:
                name = d["name"]
                if name in self.found:
                    continue
                extra = self._event_extra_wait(name)
                if extra <= 0:
                    continue

                (lx, ly) = d["loc"]
                (tw, th) = d["size"]
                inv = 1.0 / float(EFFECT_ROI_DOWNSCALE)
                cx = int(x1 + lx * inv)
                cy = int(y1 + ly * inv)
                cw = int(tw * inv)
                ch = int(th * inv)

                ev = {"name": name, "score": float(d.get("score", 0.0)), "box": (
                    cx, cy, cw, ch), "extra": float(extra)}
                events.append(ev)
                self.found.add(name)

        flash_ev = self._detect_explosion_flash(client_bgr)
        if flash_ev is not None and "EXPLOSION" not in self.found:
            events.append({"name": "EXPLOSION", "score": 0.0, "box": None, "extra": float(
                flash_ev.get("extra", 0.0)), "flash": flash_ev})
            self.found.add("EXPLOSION")

        if dbg is not None and events:
            for ev in events:
                nm = ev["name"]
                if ev.get("box") is not None:
                    x, y, w, h = ev["box"]
                    cv2.rectangle(dbg, (int(x), int(y)), (int(
                        x + w), int(y + h)), (0, 140, 255), 2)
                    cv2.putText(dbg, f"{nm} +{ev['extra']:.2f}s", (int(x), max(
                        0, int(y - 6))), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 140, 255), 2, cv2.LINE_AA)
                else:
                    cv2.putText(dbg, f"EXPLOSION +{ev['extra']:.2f}s", (10, 58),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 140, 255), 2, cv2.LINE_AA)

        self.last_events = events
        return events


class TargetTracker:
    def __init__(self):
        self.prev_pt = None
        self.prev_t = None
        self.vel = np.array([0.0, 0.0], dtype=np.float32)

    def update(self, pt, t_now):
        if pt is None:
            return
        p = np.array([float(pt[0]), float(pt[1])], dtype=np.float32)
        if self.prev_pt is not None and self.prev_t is not None:
            dt = float(max(1e-3, t_now - self.prev_t))
            v = (p - self.prev_pt) / dt
            self.vel = 0.65 * self.vel + 0.35 * v
        self.prev_pt = p
        self.prev_t = float(t_now)

    def predict(self, lead_sec):
        if self.prev_pt is None:
            return None
        p = self.prev_pt + self.vel * float(lead_sec)
        return (float(p[0]), float(p[1]))


class StrategyEngine:
    def __init__(self):
        self.plan = None
        self.plan_dbg = None
        self.last_eval_t = 0.0
        self.force_recompute = True
        self.tracker = TargetTracker()
        self.last_targets = deque(maxlen=3)
        self.next_shot_allowed_t = 0.0

    def _plan_target_point(self, chain):
        if not self.plan or not chain:
            return None
        n = len(chain)
        if self.plan.aim_kind == "pos" and self.plan.pos is not None:
            return target_point_from_pos(chain, int(self.plan.pos))
        idx = self.plan.ref_idx
        if idx is None:
            i0, i1 = self.plan.span
            idx = int((int(i0) + int(i1)) // 2)
        idx = int(clamp(int(idx), 0, n - 1))
        return chain[idx]["center"]

    def _update_tracking(self, chain, now):
        pt = self._plan_target_point(chain)
        if pt is not None:
            self.tracker.update(pt, now)
            self.last_targets.append(pt)
        return pt

    def _is_stable(self, max_dist_px=18.0, need=2):
        if len(self.last_targets) < need:
            return False
        for i in range(1, len(self.last_targets)):
            if dist(self.last_targets[i], self.last_targets[i - 1]) > float(max_dist_px):
                return False
        return True

    def recompute_plan(self, frog_center, chain, runs, risks, cumdist, cur_color, next_color, hole_center, hole_r, now):
        if not chain:
            self.plan = None
            self.plan_dbg = None
            return None

        max_risk = float(np.max(risks)) if risks else 0.0
        defend_mode = (max_risk >= DANGER_HIGH_THR)

        plans = []

        try:
            action, tgt, why, tgt_color, span, occluded, first_hit_idx, first_hit_pt = choose_target_with_raycast(
                frog_center, runs, chain, risks, cur_color, next_color)
            if tgt is not None:
                score = 0.0
                try:
                    cands = enumerate_candidates(
                        runs, chain, risks, cur_color, next_color)
                    for c in cands:
                        if c.get("action") == action and c.get("why") == why and tuple(c.get("span")) == tuple(span):
                            score = float(c.get("score", 0.0))
                            break
                except Exception:
                    pass

                ref_idx = int(
                    (int(span[0]) + int(span[1])) // 2) if span else 0
                plans.append((
                    0 if defend_mode else 1,
                    TargetPlan(
                        source="risk",
                        action=action,
                        why=why,
                        score=float(score),
                        target_color=tgt_color,
                        span=tuple(span),
                        decide_t=float(now),
                        aim_kind="ball",
                        ref_idx=ref_idx,
                        pos=None
                    ),
                    {"occluded": bool(
                        occluded), "first_hit_idx": first_hit_idx, "first_hit_pt": first_hit_pt}
                ))
        except Exception:
            pass

        hole_forbid_r = None
        if hole_center is not None and hole_r is not None:
            hole_forbid_r = int(
                float(hole_r) + float(BOT_CFG.HOLE_SHOT_FORBID_PAD))

        def add_bot_plan(shot_color, action_name):
            if shot_color is None:
                return
            pos, tgt, dbg = evaluate_best_move(
                chain, cumdist, shot_color, hole_center, hole_forbid_r, frog_center, hole_r)
            if pos is None or tgt is None or dbg is None:
                return
            expected = int(clamp(int(pos), 0, len(chain) - 1))
            span = (int(clamp(expected - BOT_CFG.EXPECT_HIT_TOL, 0, len(chain) - 1)),
                    int(clamp(expected + BOT_CFG.EXPECT_HIT_TOL, 0, len(chain) - 1)))
            removed = int(dbg.get("removed", 0))
            runlen = int(dbg.get("run", 0))
            if defend_mode:
                pr = 1 if removed >= 3 else 2
            else:
                pr = 0 if (removed >= 3 or runlen >= 3) else 1
            plans.append((
                pr,
                TargetPlan(
                    source="bot",
                    action=action_name,
                    why=f"bot_eval_{action_name}",
                    score=float(dbg.get("score", 0.0)),
                    target_color=shot_color,
                    span=span,
                    decide_t=float(now),
                    aim_kind="pos",
                    ref_idx=None,
                    pos=int(pos)
                ),
                dbg
            ))

        add_bot_plan(cur_color, "shoot")
        add_bot_plan(next_color, "swap_shoot")

        if not plans:
            self.plan = None
            self.plan_dbg = None
            return None

        plans.sort(key=lambda it: (int(it[0]), -float(it[1].score)))
        self.plan = plans[0][1]
        self.plan_dbg = plans[0][2]
        self.last_eval_t = float(now)
        self.force_recompute = False
        self.last_targets.clear()
        self.tracker = TargetTracker()
        return self.plan

    def update(self, frog_center, chain, runs, risks, cumdist, cur_color, next_color, hole_center, hole_r, now, chain_updated=False):
        if self.force_recompute or (chain_updated and (now - self.last_eval_t) >= 0.12):
            self.recompute_plan(frog_center, chain, runs, risks, cumdist,
                                cur_color, next_color, hole_center, hole_r, now)

        pt = self._update_tracking(chain, now)
        if self.plan and pt is None:
            self.force_recompute = True
        return self.plan, pt

    def pre_shot(self, frog_center, chain, lead_sec=0.06):
        if not self.plan or frog_center is None or not chain:
            return False, None, None, None

        pred = self.tracker.predict(lead_sec)
        if pred is None:
            return False, None, None, None

        ok, hit_i, hit_pt = candidate_visible_by_raycast(
            frog_center, chain, pred, self.plan.span)
        if not ok:
            self.force_recompute = True
            return False, pred, hit_i, hit_pt
        return True, pred, hit_i, hit_pt

    def on_shot_success(self, now):
        extra = 0.0
        if self.plan_dbg and isinstance(self.plan_dbg, dict):
            casc = int(self.plan_dbg.get("casc", 0))
            runlen = int(self.plan_dbg.get("run", 0))
            if casc > 0:
                extra += 0.5 * float(casc)
            if runlen > 4:
                extra += 0.25 * float(runlen - 4)
        else:
            if self.plan and self.plan.span:
                runlen = int(self.plan.span[1] - self.plan.span[0] + 1)
                if runlen > 4:
                    extra += 0.25 * float(runlen - 4)

        self.next_shot_allowed_t = float(
            now) + float(SHOOT_COOLDOWN_SEC) + float(extra)
        return extra


def capture_trail_with_baseline_and_levelmask(ctx, end_center, end_r):
    os.makedirs(OUT_DIR, exist_ok=True)

    while True:
        fr = capture_zuma_frame(ctx)
        if fr is None:
            time.sleep(0.005)
            continue
        client = crop_client_area(fr)
        prev_gray = cv2.cvtColor(client, cv2.COLOR_BGR2GRAY)
        H, W = prev_gray.shape[:2]
        break

    baseline_mask = np.zeros((H, W), dtype=np.uint8)
    level_gold_mask = np.zeros((H, W), dtype=np.uint8)
    acc = np.zeros((H, W), dtype=np.uint8)
    time_map = np.zeros((H, W), dtype=np.float32)

    end_circle = None
    if STOP_NEAR_END_ENABLED and end_center is not None and end_r is not None:
        rr = int(end_r + STOP_BEFORE_END_MARGIN_PX)
        end_circle = make_circle_mask(H, W, end_center, rr)

    t_base0 = time.time()
    while (time.time() - t_base0) < BASELINE_SEC:
        fr = capture_zuma_frame(ctx)
        if fr is None:
            time.sleep(0.005)
            continue
        client = crop_client_area(fr)
        gray = cv2.cvtColor(client, cv2.COLOR_BGR2GRAY)

        ym = yellow_mask_hsv(client)
        mm = motion_mask(prev_gray, gray)
        trail_raw = cv2.bitwise_and(ym, mm)
        trail_small = filter_small_components(
            trail_raw) if USE_SPARK_FILTER else trail_raw
        baseline_mask = cv2.bitwise_or(baseline_mask, trail_small)

        gold = build_gold_mask(client)
        level_gold_mask = cv2.bitwise_or(level_gold_mask, gold)

        prev_gray = gray

    level_gold_mask = cv2.morphologyEx(
        level_gold_mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), iterations=1)
    level_gold_mask = cv2.dilate(
        level_gold_mask, np.ones((3, 3), np.uint8), iterations=1)

    cv2.imwrite(os.path.join(
        OUT_DIR, "trail_baseline_white.png"), baseline_mask)
    cv2.imwrite(os.path.join(
        OUT_DIR, "trail_level_gold_mask.png"), level_gold_mask)

    streak = 0
    t_wait0 = time.time()
    while True:
        if (time.time() - t_wait0) > WAIT_TRAIL_TIMEOUT:
            break
        fr = capture_zuma_frame(ctx)
        if fr is None:
            time.sleep(0.005)
            continue

        client = crop_client_area(fr)
        gray = cv2.cvtColor(client, cv2.COLOR_BGR2GRAY)

        ym = yellow_mask_hsv(client)
        mm = motion_mask(prev_gray, gray)
        trail_raw = cv2.bitwise_and(ym, mm)
        trail_small = filter_small_components(
            trail_raw) if USE_SPARK_FILTER else trail_raw
        trail_clean = cv2.bitwise_and(
            trail_small, cv2.bitwise_not(baseline_mask))

        px = int(cv2.countNonZero(trail_clean))
        streak = (streak + 1) if (px >= TRAIL_START_THR_PIXELS) else 0
        prev_gray = gray

        if streak >= TRAIL_START_STREAK:
            break

    t_cap0 = time.time()
    near_streak = 0
    stop_target = None

    while True:
        dt = time.time() - t_cap0
        if dt >= CAPTURE_MAX_SEC:
            break
        if stop_target is not None and dt >= stop_target:
            break

        fr = capture_zuma_frame(ctx)
        if fr is None:
            time.sleep(0.005)
            continue

        client = crop_client_area(fr)
        gray = cv2.cvtColor(client, cv2.COLOR_BGR2GRAY)

        ym = yellow_mask_hsv(client)
        mm = motion_mask(prev_gray, gray)
        trail_raw = cv2.bitwise_and(ym, mm)
        trail_small = filter_small_components(
            trail_raw) if USE_SPARK_FILTER else trail_raw
        trail_clean = cv2.bitwise_and(
            trail_small, cv2.bitwise_not(baseline_mask))

        trail_acc = trail_clean
        if PER_FRAME_DILATE > 0:
            trail_acc = cv2.dilate(trail_acc, np.ones(
                (3, 3), np.uint8), iterations=PER_FRAME_DILATE)
        if PER_FRAME_CLOSE > 0:
            trail_acc = cv2.morphologyEx(trail_acc, cv2.MORPH_CLOSE, np.ones(
                (5, 5), np.uint8), iterations=PER_FRAME_CLOSE)

        acc = cv2.bitwise_or(acc, trail_acc)

        new = (trail_clean > 0) & (time_map == 0.0)
        time_map[new] = float(dt)

        if end_circle is not None:
            near = cv2.bitwise_and(trail_clean, end_circle)
            near_px = int(cv2.countNonZero(near))
            if near_px >= END_NEAR_MIN_NEWPIX:
                near_streak += 1
            else:
                near_streak = 0

            if near_streak >= END_NEAR_STREAK and stop_target is None:
                stop_target = dt + EXTRA_AFTER_NEAR_END_SEC

        prev_gray = gray

    capture_duration = float(min(time.time() - t_cap0, CAPTURE_MAX_SEC))

    acc[level_gold_mask > 0] = 0
    time_map[level_gold_mask > 0] = 0.0

    cv2.imwrite(os.path.join(OUT_DIR, "trail_acc_raw.png"), acc)
    np.save(os.path.join(OUT_DIR, "trail_time_map.npy"), time_map)

    with open(os.path.join(OUT_DIR, "capture_duration.txt"), "w", encoding="utf-8") as f:
        f.write(f"{capture_duration:.4f}\n")

    return acc, time_map, capture_duration


def main():
    global AUTO_ENABLED
    os.makedirs(OUT_DIR, exist_ok=True)
    cv2.setUseOptimized(True)

    time.sleep(3)

    ctx = init_zuma_window(template_path=WINDOW_TEMPLATE_PATH, fixed_width=WINDOW_FIXED_W,
                           fixed_height=WINDOW_FIXED_H, match_threshold=WINDOW_MATCH_THRESHOLD)
    if ctx is None:
        print("لم يتم العثور على نافذة Zuma")
        return

    if not HAVE_INPUT:
        print("pyautogui غير متوفر -> Debug فقط بدون رمي/تبديل.")
    else:
        focus_window_once(ctx)

    tpl_bgr = cv2.imread(LEVEL_TEMPLATE_PATH)
    if tpl_bgr is None:
        print("فشل تحميل template:", LEVEL_TEMPLATE_PATH)
        return

    tpl_mask = build_gold_mask(tpl_bgr)
    tpl_mask_ds = cv2.resize(tpl_mask, None, fx=LEVEL_DOWNSCALE,
                             fy=LEVEL_DOWNSCALE, interpolation=cv2.INTER_NEAREST)

    print("WAIT_LEVEL ... (F10 للخروج)")
    level_frame = None
    t_level0 = time.time()

    while True:
        chain_updated = False
        if keyboard.is_pressed("F10"):
            print("إيقاف يدوي")
            return

        frame = capture_zuma_frame(ctx)
        if frame is None:
            time.sleep(0.005)
            continue

        score, box_full = extract_level_score_and_box(frame, tpl_mask_ds)
        if score >= LEVEL_MATCH_THRESHOLD and box_full is not None:
            level_frame = frame.copy()
            break

    print(f"🎉 LEVEL detected in {time.time() - t_level0:.3f}s")

    if POST_LEVEL_DELAY_SEC > 0:
        time.sleep(float(POST_LEVEL_DELAY_SEC))

    end_center, end_r = None, None
    t_end0 = time.time()
    if level_frame is not None:
        client0 = crop_client_area(level_frame)
        end_center, end_r = detect_end_fast(client0)
        if end_center is None:
            for _ in range(3):
                fr = capture_zuma_frame(ctx)
                if fr is None:
                    continue
                cc = crop_client_area(fr)
                c2, r2 = detect_end_fast(cc)
                if c2 is not None:
                    end_center, end_r = c2, r2
                    break
    print(
        f"END detect in {time.time() - t_end0:.3f}s | center={end_center} r={end_r}")

    print("Capturing trail ...")
    t_tr0 = time.time()
    acc_raw, time_map, cap_dur = capture_trail_with_baseline_and_levelmask(
        ctx, end_center, end_r)
    print(
        f"Trail captured in {time.time() - t_tr0:.3f}s | cap_dur={cap_dur:.3f}s")

    risk_map_u8, risk_field_u8 = build_risk_field(
        time_map, cap_dur, radius=RISK_FIELD_R)
    cv2.imwrite(os.path.join(OUT_DIR, "risk_map_u8.png"), risk_map_u8)
    cv2.imwrite(os.path.join(OUT_DIR, "risk_field_u8.png"), risk_field_u8)

    path_dt = None
    if PATH_FILTER_ENABLE:
        path_mask_u8 = (risk_map_u8 > 0).astype(np.uint8) * 255
        if int(path_mask_u8.max()) == 0:
            print("[PATH] risk_map_u8 is empty; path filtering disabled.")
            path_dt = None
        else:
            inv = cv2.bitwise_not(path_mask_u8)
            path_dt = cv2.distanceTransform(inv, cv2.DIST_L2, 3)
            cv2.imwrite(os.path.join(
                OUT_DIR, "path_mask_u8.png"), path_mask_u8)

    frog_box = None
    frog_score = -1.0
    frog_center = None
    frog_r = None

    if not HAVE_FROG:
        print("frog_detector not available.")
    else:
        tpl_gray = load_template_gray(FROG_TEMPLATE_GRAY)
        tpl_edges = load_template_edges(FROG_TEMPLATE_GRAY)

        cfg = FrogFastConfig(center_roi_ratio=0.62,
                             track_pad=90, return_debug=True)
        detector = FrogFastDetector(tpl_gray, tpl_edges, cfg)

        t_fg0 = time.time()
        last_meta = None
        last_client = None

        for _ in range(6):
            fr = capture_zuma_frame(ctx)
            if fr is None:
                continue
            client = crop_client_area(fr)

            box, sc, meta = detector.detect(client, prev_box=frog_box)

            last_meta = meta
            last_client = client

            if box is not None and float(sc) > float(frog_score):
                frog_box = box
                frog_score = float(sc)

            chosen = meta.get("chosen", {})
            if frog_box is not None and chosen.get("method", None) == "EDGES" and float(sc) >= float(cfg.edges_early_accept):
                break

        t_fg = time.time() - t_fg0

        if last_client is not None and last_meta is not None:
            vis = render_frog_roi_debug_image(last_client, last_meta)
            cv2.imwrite(os.path.join(OUT_DIR, "frog_search_roi.png"), vis)

        if frog_box is None:
            print(f"Frog not found in {t_fg:.3f}s (best={frog_score:.3f}).")
        else:
            x, y, w, h = [int(v) for v in frog_box]
            frog_center = (int(x + w / 2), int(y + h / 2))
            frog_r = int(0.5 * min(w, h))
            chosen = (last_meta or {}).get("chosen", {})
            print(
                f"Frog locked in {t_fg:.3f}s score={frog_score:.3f} method={chosen.get('method')} box={frog_box} center={frog_center}")

    balls_cfg = default_balls_cfg()

    if DEBUG_ENABLE:
        cv2.namedWindow(DEBUG_WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(DEBUG_WINDOW, DEBUG_W, DEBUG_H)

    print("RUNNING: F10 خروج | F8 Toggle AUTO | q خروج debug")
    if HAVE_INPUT:
        focus_window_once(ctx)

    last_dbg_t = 0.0
    last_shot_t = 0.0
    last_swap_t = 0.0
    swap_pending = False
    swap_pending_until = 0.0

    frame_id = 0
    chain_cached = []
    runs_cached = []
    risks_cached = []
    cumdist_cached = None

    engine = StrategyEngine()

    glow_detector = GlowDetector(interval_sec=GLOW_INTERVAL_SEC, ttl_sec=GLOW_TTL_SEC,
                                 match_dist_px=GLOW_MATCH_DIST_PX) if GLOW_ENABLE else None
    effect_monitor = PostShotEffectMonitor() if POST_SHOT_EFFECTS_ENABLE else None

    while True:
        if keyboard.is_pressed("F10"):
            print("إيقاف يدوي (F10)")
            break

        if keyboard.is_pressed("F8"):
            AUTO_ENABLED = not AUTO_ENABLED
            print(f"AUTO_ENABLED = {AUTO_ENABLED}")
            time.sleep(0.25)

        fr = capture_zuma_frame(ctx)
        if fr is None:
            time.sleep(0.003)
            continue

        client = crop_client_area(fr)
        dbg = client.copy()

        if end_center is not None and end_r is not None:
            cv2.circle(dbg, end_center, int(end_r), (0, 255, 255), 2)
            cv2.putText(dbg, "END", (end_center[0] + 8, end_center[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

        if frog_box is not None:
            draw_box(dbg, frog_box, (0, 255, 0), 2, f"FROG {frog_score:.2f}")
        if frog_center is not None:
            cv2.circle(dbg, frog_center, 4, (255, 255, 255), -1)

        cur_ball, nxt_ball = None, None
        if frog_box is not None:
            balls_near = detect_two_nearest_balls(
                client, frog_box, expand_ratio=0.45)
            if balls_near:
                cur_ball = balls_near[0] if len(balls_near) >= 1 else None
                nxt_ball = balls_near[1] if len(balls_near) >= 2 else None

        draw_ball_info(dbg, cur_ball, "CUR", color_bgr=(255, 255, 255))
        draw_ball_info(dbg, nxt_ball, "NEXT", color_bgr=(200, 200, 200))

        cur_color = cur_ball.get("color", None) if cur_ball else None
        next_color = nxt_ball.get("color", None) if nxt_ball else None

        chain_updated = False
        if frog_center is not None and frog_r is not None:
            if (frame_id % DETECT_CHAIN_EVERY) == 0:
                balls = detect_chain_balls(
                    client, frog_center=frog_center, frog_r=frog_r, hole_center=end_center, hole_r=end_r, cfg=balls_cfg)

                if PATH_FILTER_ENABLE and (path_dt is not None):
                    before_n = len(balls)
                    balls = filter_balls_on_path(balls, path_dt=path_dt, risk_field_u8=risk_field_u8, tol_base_px=PATH_DIST_TOL_PX,
                                                 fallback_use_field=PATH_FALLBACK_USE_FIELD, fallback_mult=PATH_FALLBACK_MULT)
                    after_n = len(balls)
                    if DEBUG_ENABLE and before_n > 0 and (before_n - after_n) >= 2:
                        print(
                            f"🧹 [PATH] Filtered {before_n - after_n}/{before_n} off-path circles")

                chain, cumdist = order_chain(balls, end_center, balls_cfg)

                risks = []
                for b in chain:
                    r = ball_risk_from_field(b["center"], risk_field_u8)
                    if r is None:
                        r = fallback_risk_by_end(b["center"], end_center)
                    risks.append(float(r))

                runs = build_runs(chain, risks)

                chain_cached = chain
                risks_cached = risks
                runs_cached = runs
                cumdist_cached = cumdist
                chain_updated = True

        chain = chain_cached
        risks = risks_cached
        runs = runs_cached
        cumdist = cumdist_cached

        now = time.time()

        if glow_detector is not None and chain is not None and len(chain) > 0:
            glow_detector.update(client, chain, now, dbg=dbg)
            special_candidates = glow_detector.get_active(now)
        else:
            special_candidates = []

        if effect_monitor is not None:
            evs = effect_monitor.update(client, now, dbg=dbg)
            if evs:
                for ev in evs:
                    extra = float(ev.get("extra", 0.0))
                    name = str(ev.get("name", ""))
                    if extra > 1e-6:
                        engine.next_shot_allowed_t = float(
                            engine.next_shot_allowed_t) + extra
                        engine.force_recompute = True
                        if DEBUG_ENABLE:
                            print(
                                f"⏸️ [EFFECT] {name} detected -> add wait +{extra:.2f}s")

        plan, target_pt = engine.update(frog_center=frog_center, chain=chain, runs=runs, risks=risks, cumdist=cumdist,
                                        cur_color=cur_color, next_color=next_color, hole_center=end_center, hole_r=end_r, now=now, chain_updated=chain_updated)

        if plan is None:
            action, why, target_color, span = None, "no_plan", None, (0, -1)
            occluded, first_hit_idx, first_hit_pt = False, None, None
        else:
            action = plan.action
            why = f"{plan.source}:{plan.why}"
            target_color = plan.target_color
            span = plan.span
            ok_vis, hit_i, hit_pt = candidate_visible_by_raycast(frog_center, chain, target_pt, span) if (
                frog_center is not None and target_pt is not None and chain) else (True, None, None)
            occluded, first_hit_idx, first_hit_pt = (not ok_vis), hit_i, hit_pt

        max_risk = float(np.max(risks)) if risks else 0.0
        mode_txt = "DEFEND" if max_risk >= DANGER_HIGH_THR else "SCORE"

        if target_pt is not None:
            cv2.circle(dbg, (int(target_pt[0]), int(
                target_pt[1])), 16, (0, 0, 255), 2)
            cv2.putText(dbg, f"TARGET {target_color}", (int(target_pt[0]) + 12, int(
                target_pt[1]) + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

        if RAYCAST_DRAW and frog_center is not None and target_pt is not None and chain:
            cv2.line(dbg, (int(frog_center[0]), int(frog_center[1])), (int(
                target_pt[0]), int(target_pt[1])), (255, 255, 0), 1)
            if first_hit_idx is not None:
                hb = chain[first_hit_idx]
                hc = hb.get("center", None)
                hr = int(hb.get("r", 18))
                if hc is not None:
                    cv2.circle(dbg, (int(hc[0]), int(
                        hc[1])), hr + 2, (0, 0, 255) if occluded else (0, 255, 255), 2)
                    cv2.putText(dbg, f"FIRST_HIT i={first_hit_idx}", (int(hc[0]) + 10, int(
                        hc[1]) + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255) if occluded else (0, 255, 255), 2, cv2.LINE_AA)

        if chain:
            for i, b in enumerate(chain[:DEBUG_DRAW_TOPK]):
                x, y = b["center"]
                rr = int(b.get("r", 18))
                cv2.circle(dbg, (int(x), int(y)), rr, (0, 255, 0), 2)
                cv2.putText(dbg, f"{i}:{b.get('color', '?')} {risks[i]:.0f}%", (int(
                    x) - 18, int(y) - rr - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.putText(dbg, f"AUTO={AUTO_ENABLED}  MODE={mode_txt}  maxRisk={max_risk:.0f}%  CUR={cur_color} NEXT={next_color}  chain={len(chain)}  why={why}", (
            10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)

        if swap_pending and now >= swap_pending_until:
            swap_pending = False

        if AUTO_ENABLED and HAVE_INPUT and frog_center is not None and target_pt is not None:
            if action == "swap_shoot":
                if (now - last_swap_t) >= SWAP_COOLDOWN_SEC and not swap_pending:
                    ok = right_click_swap(ctx, frog_center)
                    if ok:
                        last_swap_t = now
                        swap_pending = True
                        swap_pending_until = now + AFTER_SWAP_VERIFY_SEC
                        engine.force_recompute = True

            if not swap_pending and now >= engine.next_shot_allowed_t:
                if engine._is_stable(max_dist_px=18.0, need=2):
                    ok_pre, pred_pt, hit_i2, hit_pt2 = engine.pre_shot(
                        frog_center, chain, lead_sec=0.06)
                    if ok_pre and pred_pt is not None:
                        tx, ty = float(pred_pt[0]), float(pred_pt[1])

                        if AIM_OFFSET_PX != 0 and frog_center is not None:
                            fx, fy = float(frog_center[0]), float(
                                frog_center[1])
                            vx, vy = (tx - fx), (ty - fy)
                            norm = (vx * vx + vy * vy) ** 0.5
                            if norm > 1e-6:
                                tx += (vx / norm) * float(AIM_OFFSET_PX)
                                ty += (vy / norm) * float(AIM_OFFSET_PX)

                        ok = aim_and_shoot(ctx, frog_center, (tx, ty))
                        if ok:
                            last_shot_t = now
                            engine.on_shot_success(now)
                            if effect_monitor is not None:
                                try:
                                    effect_monitor.notify_shot(
                                        now, (tx, ty), special_candidates)
                                except Exception:
                                    pass

        if DEBUG_ENABLE:
            t = time.time()
            if (t - last_dbg_t) >= (1.0 / float(DEBUG_MAX_FPS)):
                last_dbg_t = t
                if DEBUG_SCALE != 1.0:
                    dbg2 = cv2.resize(dbg, None, fx=DEBUG_SCALE,
                                      fy=DEBUG_SCALE, interpolation=cv2.INTER_AREA)
                else:
                    dbg2 = dbg
                cv2.imshow(DEBUG_WINDOW, dbg2)
                k = cv2.waitKey(1) & 0xFF
                if k == ord("q"):
                    print("خروج (q)")
                    break

        frame_id += 1

    if DEBUG_ENABLE:
        cv2.destroyAllWindows()

    print("Done")


if __name__ == "__main__":
    main()
