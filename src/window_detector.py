from __future__ import annotations
import time
from typing import Dict

import cv2
import numpy as np
import mss


def _grab_desktop_bgr(sct: mss.mss):
    mon = sct.monitors[0]
    left, top, width, height = int(mon["left"]), int(mon["top"]), int(mon["width"]), int(mon["height"])
    raw = np.array(sct.grab(mon), dtype=np.uint8)
    bgr = cv2.cvtColor(raw, cv2.COLOR_BGRA2BGR)
    return bgr, (left, top, width, height)


def _grab_region_bgr(sct: mss.mss, left: int, top: int, width: int, height: int):
    if width <= 2 or height <= 2:
        return None
    mon = {"left": int(left), "top": int(top), "width": int(width), "height": int(height)}
    raw = np.array(sct.grab(mon), dtype=np.uint8)
    return cv2.cvtColor(raw, cv2.COLOR_BGRA2BGR)


def _match_multiscale(screen_gray: np.ndarray, tpl_gray: np.ndarray, scales, thr: float, ds: float):
    ds = 1.0 if (ds <= 0.0 or ds > 1.0) else float(ds)
    scr_ds = cv2.resize(screen_gray, None, fx=ds, fy=ds, interpolation=cv2.INTER_AREA) if ds != 1.0 else screen_gray
    Hs, Ws = scr_ds.shape[:2]

    best_score, best_loc = -1.0, None
    for s in scales:
        fx = ds * float(s)
        if fx <= 0.05:
            continue
        tpl = cv2.resize(tpl_gray, None, fx=fx, fy=fx, interpolation=cv2.INTER_AREA)
        th, tw = tpl.shape[:2]
        if th < 10 or tw < 10 or th >= Hs or tw >= Ws:
            continue

        res = cv2.matchTemplate(scr_ds, tpl, cv2.TM_CCOEFF_NORMED)
        _, maxv, _, maxl = cv2.minMaxLoc(res)
        if float(maxv) > best_score:
            best_score = float(maxv)
            best_loc = (int(round(maxl[0] / ds)), int(round(maxl[1] / ds)))

    if best_score < float(thr):
        return best_score, None
    return best_score, best_loc


def init_zuma_window(
    template_path: str = "../templates/zuma_window_head.png",
    fixed_width: int = 808,
    fixed_height: int = 636,
    match_threshold: float = 0.62,
    scales=(1.00, 0.96, 0.92, 0.88, 0.84, 0.80, 1.06),
    screen_downscale: float = 0.65,
):
    tpl = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
    if tpl is None:
        print(f"window_detector_cv_mss: template not found: {template_path}")
        return None

    sct = mss.mss()
    desktop_bgr, vrect = _grab_desktop_bgr(sct)
    if desktop_bgr is None:
        print("window_detector_cv_mss: failed to capture desktop")
        return None

    vx, vy, _, _ = vrect
    scr_gray = cv2.cvtColor(desktop_bgr, cv2.COLOR_BGR2GRAY)

    score, loc = _match_multiscale(scr_gray, tpl, scales=scales, thr=match_threshold, ds=screen_downscale)
    if loc is None:
        print(f"window_detector_cv_mss: template match failed (best={score:.3f}, thr={match_threshold})")
        return None

    mx, my = loc
    gx, gy = int(mx + vx), int(my + vy)

    bbox = (gx, gy, int(fixed_width), int(fixed_height))
    print(f"window_detector_cv_mss: found | score={score:.3f} bbox={bbox}")

    return {"sct": sct, "bbox": bbox, "vrect": vrect, "last_match_score": float(score), "created_t": time.time()}


def capture_zuma_frame(ctx: Dict):
    if ctx is None:
        return None
    sct = ctx.get("sct")
    if sct is None:
        ctx["sct"] = mss.mss()
        sct = ctx["sct"]

    left, top, w, h = ctx.get("bbox", (0, 0, 0, 0))
    return _grab_region_bgr(sct, left, top, w, h)
