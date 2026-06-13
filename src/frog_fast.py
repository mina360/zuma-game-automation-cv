from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Any

import cv2
import numpy as np

from frog_detector import (
    preprocess_gray,
    multi_angle_template_match_gray,
    multi_angle_template_match_edges,
    box_from_loc_tpl,
)

Box = Tuple[int, int, int, int]


@dataclass
class FrogFastConfig:
    center_roi_ratio: float = 0.62 
    track_pad: int = 90

    edges_coarse_downscale: float = 0.60
    edges_coarse_angles: List[int] = field(default_factory=lambda: list(range(0, 360, 45)))
    edges_coarse_scales: List[float] = field(default_factory=lambda: [0.95, 1.0, 1.05])

    edges_fine_angles: List[int] = field(default_factory=lambda: list(range(0, 360, 20)))
    edges_fine_scales: List[float] = field(default_factory=lambda: [0.90, 1.0, 1.10])

    edges_early_accept: float = 0.82
    edges_min_score: float = 0.55

    gray_coarse_downscale: float = 0.60
    gray_coarse_angles: List[int] = field(default_factory=lambda: list(range(0, 360, 40)))
    gray_coarse_scales: List[float] = field(default_factory=lambda: [0.9, 1.0, 1.1])

    gray_fine_angles: List[int] = field(default_factory=lambda: list(range(0, 360, 20)))
    gray_fine_scales: List[float] = field(default_factory=lambda: [0.85, 0.95, 1.0, 1.05, 1.15])

    gray_min_score: float = 0.60

    refine_pad: int = 140
    agree_tol_px: float = 35.0

    canny_lo: int = 80
    canny_hi: int = 160

    return_debug: bool = True


def _clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


def _clip_box(box: Box, W: int, H: int) -> Box:
    x, y, w, h = box
    x = _clamp(int(x), 0, W - 1)
    y = _clamp(int(y), 0, H - 1)
    w = _clamp(int(w), 1, W - x)
    h = _clamp(int(h), 1, H - y)
    return (x, y, w, h)


def _expand_box(box: Box, pad: int, W: int, H: int) -> Box:
    x, y, w, h = box
    x2 = x - pad
    y2 = y - pad
    w2 = w + 2 * pad
    h2 = h + 2 * pad
    return _clip_box((x2, y2, w2, h2), W, H)


def _center_roi(W: int, H: int, ratio: float) -> Box:
    rw = int(W * float(ratio))
    rh = int(H * float(ratio))
    cx = W // 2
    cy = H // 2
    x = int(cx - rw // 2)
    y = int(cy - rh // 2)
    return _clip_box((x, y, rw, rh), W, H)


def _extract_roi(img: np.ndarray, box: Box) -> Tuple[np.ndarray, Tuple[int, int]]:
    x, y, w, h = box
    return img[y:y + h, x:x + w], (x, y)


def _offset_box(box: Box, ox: int, oy: int) -> Box:
    x, y, w, h = box
    return (int(x + ox), int(y + oy), int(w), int(h))


def _center_of_box(box: Box) -> Tuple[float, float]:
    x, y, w, h = box
    return (float(x) + float(w) / 2.0, float(y) + float(h) / 2.0)


def _dist(a, b) -> float:
    dx = float(a[0] - b[0])
    dy = float(a[1] - b[1])
    return float((dx * dx + dy * dy) ** 0.5)


class FrogFastDetector:
    def __init__(self, tpl_gray: np.ndarray, tpl_edges: np.ndarray, cfg: FrogFastConfig):
        self.cfg = cfg
        self.tpl_gray = tpl_gray
        self.tpl_edges = tpl_edges

        self.tpl_gray_coarse = self._resize_tpl(self.tpl_gray, cfg.gray_coarse_downscale, interp=cv2.INTER_AREA)
        self.tpl_edges_coarse = self._resize_tpl(self.tpl_edges, cfg.edges_coarse_downscale, interp=cv2.INTER_NEAREST)

    @staticmethod
    def _resize_tpl(tpl: np.ndarray, ds: float, interp) -> np.ndarray:
        ds = float(ds)
        if ds < 0.95:
            return cv2.resize(tpl, None, fx=ds, fy=ds, interpolation=interp)
        return tpl

    def compute_roi(self, client_bgr: np.ndarray, prev_box: Optional[Box]) -> Dict[str, Any]:
        H, W = client_bgr.shape[:2]
        out: Dict[str, Any] = {}

        center_box = _center_roi(W, H, self.cfg.center_roi_ratio)
        out["center_roi_box"] = center_box

        if prev_box is not None:
            track_box = _expand_box(prev_box, self.cfg.track_pad, W, H)
            out["track_roi_box"] = track_box
            out["roi_box"] = track_box
            out["roi_reason"] = "TRACK"
            return out

        out["track_roi_box"] = None
        out["roi_box"] = center_box
        out["roi_reason"] = "CENTER"
        return out

    def _edges_from_bgr(self, bgr: np.ndarray) -> np.ndarray:
        g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        g = cv2.GaussianBlur(g, (5, 5), 1)
        e = cv2.Canny(g, self.cfg.canny_lo, self.cfg.canny_hi)
        return e

    def _match_edges_scaled_rot(self, edges_frame: np.ndarray, template_edges: np.ndarray, angles: List[int], scales: List[float]):
        best = (-1.0, None, None, None, None)
        H, W = edges_frame.shape[:2]

        for sc in scales:
            tw = int(template_edges.shape[1] * float(sc))
            th = int(template_edges.shape[0] * float(sc))
            if tw < 8 or th < 8 or tw >= W or th >= H:
                continue
            tpl_s = cv2.resize(template_edges, (tw, th), interpolation=cv2.INTER_NEAREST)

            score, loc, angle, tpl_rot = multi_angle_template_match_edges(edges_frame, tpl_s, angles)
            score = float(score)
            if tpl_rot is None or loc is None:
                continue
            if score > best[0]:
                best = (score, loc, angle, float(sc), tpl_rot)

        return best

    def _match_gray(self, gray_frame: np.ndarray, template_gray: np.ndarray, angles: List[int], scales: List[float]):
        score, loc, angle, scale, tpl_used = multi_angle_template_match_gray(gray_frame, template_gray, angles, scales)
        return float(score), loc, angle, float(scale) if scale is not None else None, tpl_used

    def _detect_edges_coarse_fine(self, roi_bgr: np.ndarray) -> Tuple[Optional[Box], float, Dict[str, Any]]:
        dbg: Dict[str, Any] = {}
        if roi_bgr is None or roi_bgr.size == 0:
            return None, 0.0, dbg

        ds = float(self.cfg.edges_coarse_downscale)
        roi_small = cv2.resize(roi_bgr, None, fx=ds, fy=ds, interpolation=cv2.INTER_AREA) if ds < 0.95 else roi_bgr
        edges_small = self._edges_from_bgr(roi_small)

        score_c, loc_c, ang_c, sc_c, tpl_c = self._match_edges_scaled_rot(
            edges_small,
            self.tpl_edges_coarse,
            self.cfg.edges_coarse_angles,
            self.cfg.edges_coarse_scales
        )

        dbg["coarse_score"] = float(score_c)
        dbg["coarse_loc"] = loc_c
        dbg["coarse_angle"] = ang_c
        dbg["coarse_scale"] = sc_c

        if tpl_c is None or loc_c is None:
            return None, float(score_c), dbg

        box_small = box_from_loc_tpl(loc_c, tpl_c)
        if box_small is None:
            return None, float(score_c), dbg

        if ds < 0.95:
            inv = 1.0 / ds
            x, y, w, h = box_small
            box_roi = (int(round(x * inv)), int(round(y * inv)),
                       int(round(w * inv)), int(round(h * inv)))
        else:
            box_roi = box_small

        dbg["coarse_box_roi"] = box_roi

        if float(score_c) >= float(self.cfg.edges_early_accept):
            dbg["stage"] = "edges_coarse_early"
            return _clip_box(box_roi, roi_bgr.shape[1], roi_bgr.shape[0]), float(score_c), dbg

        x, y, w, h = box_roi
        cx = int(x + w // 2)
        cy = int(y + h // 2)
        pad = int(self.cfg.refine_pad)

        rx0 = _clamp(cx - pad, 0, roi_bgr.shape[1] - 1)
        ry0 = _clamp(cy - pad, 0, roi_bgr.shape[0] - 1)
        rx1 = _clamp(cx + pad, 1, roi_bgr.shape[1])
        ry1 = _clamp(cy + pad, 1, roi_bgr.shape[0])

        refine_box = (rx0, ry0, rx1 - rx0, ry1 - ry0)
        dbg["refine_box_roi"] = refine_box

        ref_bgr, (rox, roy) = _extract_roi(roi_bgr, refine_box)
        if ref_bgr.size == 0:
            dbg["stage"] = "edges_ref_empty_ref_use_coarse"
            return _clip_box(box_roi, roi_bgr.shape[1], roi_bgr.shape[0]), float(score_c), dbg

        edges_ref = self._edges_from_bgr(ref_bgr)
        score_f, loc_f, ang_f, sc_f, tpl_f = self._match_edges_scaled_rot(
            edges_ref,
            self.tpl_edges,
            self.cfg.edges_fine_angles,
            self.cfg.edges_fine_scales
        )

        dbg["fine_score"] = float(score_f)
        dbg["fine_loc"] = loc_f
        dbg["fine_angle"] = ang_f
        dbg["fine_scale"] = sc_f

        if tpl_f is None or loc_f is None:
            dbg["stage"] = "edges_ref_failed_use_coarse"
            return _clip_box(box_roi, roi_bgr.shape[1], roi_bgr.shape[0]), float(score_c), dbg

        box_ref = box_from_loc_tpl(loc_f, tpl_f)
        if box_ref is None:
            dbg["stage"] = "edges_ref_box_none_use_coarse"
            return _clip_box(box_roi, roi_bgr.shape[1], roi_bgr.shape[0]), float(score_c), dbg

        box_roi2 = _offset_box(box_ref, rox, roy)
        dbg["fine_box_roi"] = box_roi2
        dbg["stage"] = "edges_refine_ok"
        return _clip_box(box_roi2, roi_bgr.shape[1], roi_bgr.shape[0]), float(score_f), dbg

    def _detect_gray_coarse_fine(self, roi_bgr: np.ndarray) -> Tuple[Optional[Box], float, Dict[str, Any]]:
        dbg: Dict[str, Any] = {}
        if roi_bgr is None or roi_bgr.size == 0:
            return None, 0.0, dbg

        ds = float(self.cfg.gray_coarse_downscale)
        roi_small = cv2.resize(roi_bgr, None, fx=ds, fy=ds, interpolation=cv2.INTER_AREA) if ds < 0.95 else roi_bgr

        gray_small = cv2.cvtColor(roi_small, cv2.COLOR_BGR2GRAY)
        gray_small = preprocess_gray(gray_small)

        score_c, loc_c, ang_c, sc_c, tpl_c = self._match_gray(
            gray_small,
            self.tpl_gray_coarse,
            self.cfg.gray_coarse_angles,
            self.cfg.gray_coarse_scales
        )

        dbg["coarse_score"] = float(score_c)
        dbg["coarse_loc"] = loc_c
        dbg["coarse_angle"] = ang_c
        dbg["coarse_scale"] = sc_c

        if tpl_c is None or loc_c is None:
            return None, float(score_c), dbg

        box_small = box_from_loc_tpl(loc_c, tpl_c)
        if box_small is None:
            return None, float(score_c), dbg

        if ds < 0.95:
            inv = 1.0 / ds
            x, y, w, h = box_small
            box_roi = (int(round(x * inv)), int(round(y * inv)),
                       int(round(w * inv)), int(round(h * inv)))
        else:
            box_roi = box_small

        dbg["coarse_box_roi"] = box_roi

        x, y, w, h = box_roi
        cx = int(x + w // 2)
        cy = int(y + h // 2)
        pad = int(self.cfg.refine_pad)

        rx0 = _clamp(cx - pad, 0, roi_bgr.shape[1] - 1)
        ry0 = _clamp(cy - pad, 0, roi_bgr.shape[0] - 1)
        rx1 = _clamp(cx + pad, 1, roi_bgr.shape[1])
        ry1 = _clamp(cy + pad, 1, roi_bgr.shape[0])

        refine_box = (rx0, ry0, rx1 - rx0, ry1 - ry0)
        dbg["refine_box_roi"] = refine_box

        ref_bgr, (rox, roy) = _extract_roi(roi_bgr, refine_box)
        if ref_bgr.size == 0:
            dbg["stage"] = "gray_ref_empty_ref_use_coarse"
            return _clip_box(box_roi, roi_bgr.shape[1], roi_bgr.shape[0]), float(score_c), dbg

        gray_ref = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2GRAY)
        gray_ref = preprocess_gray(gray_ref)

        score_f, loc_f, ang_f, sc_f, tpl_f = self._match_gray(
            gray_ref,
            self.tpl_gray,
            self.cfg.gray_fine_angles,
            self.cfg.gray_fine_scales
        )

        dbg["fine_score"] = float(score_f)
        dbg["fine_loc"] = loc_f
        dbg["fine_angle"] = ang_f
        dbg["fine_scale"] = sc_f

        if tpl_f is None or loc_f is None:
            dbg["stage"] = "gray_ref_failed_use_coarse"
            return _clip_box(box_roi, roi_bgr.shape[1], roi_bgr.shape[0]), float(score_c), dbg

        box_ref = box_from_loc_tpl(loc_f, tpl_f)
        if box_ref is None:
            dbg["stage"] = "gray_ref_box_none_use_coarse"
            return _clip_box(box_roi, roi_bgr.shape[1], roi_bgr.shape[0]), float(score_c), dbg

        box_roi2 = _offset_box(box_ref, rox, roy)
        dbg["fine_box_roi"] = box_roi2
        dbg["stage"] = "gray_refine_ok"
        return _clip_box(box_roi2, roi_bgr.shape[1], roi_bgr.shape[0]), float(score_f), dbg

    def detect(self, client_bgr: np.ndarray, prev_box: Optional[Box] = None):
        dbg: Dict[str, Any] = {}

        roi_info = self.compute_roi(client_bgr, prev_box)
        roi_box = roi_info["roi_box"]
        dbg.update(roi_info)

        roi_bgr, (ox, oy) = _extract_roi(client_bgr, roi_box)
        if roi_bgr.size == 0:
            return None, 0.0, dbg

        edges_box_roi, edges_score, edges_dbg = self._detect_edges_coarse_fine(roi_bgr)
        dbg["edges"] = {"box_roi": edges_box_roi, "score": float(edges_score), "dbg": edges_dbg}

        if edges_box_roi is not None and float(edges_score) >= float(self.cfg.edges_early_accept):
            best_client = _offset_box(edges_box_roi, ox, oy)
            dbg["chosen"] = {"method": "EDGES", "reason": "EARLY_ACCEPT", "score": float(edges_score)}
            return best_client, float(edges_score), dbg

        gray_box_roi, gray_score, gray_dbg = self._detect_gray_coarse_fine(roi_bgr)
        dbg["gray"] = {"box_roi": gray_box_roi, "score": float(gray_score), "dbg": gray_dbg}

        edges_ok = (edges_box_roi is not None and float(edges_score) >= float(self.cfg.edges_min_score))
        gray_ok = (gray_box_roi is not None and float(gray_score) >= float(self.cfg.gray_min_score))

        best_method = None
        best_box_roi = None
        best_score = 0.0
        choose_reason = "NONE"

        if edges_ok and not gray_ok:
            best_method, best_box_roi, best_score = "EDGES", edges_box_roi, float(edges_score)
            choose_reason = "ONLY_EDGES_OK"
        elif gray_ok and not edges_ok:
            best_method, best_box_roi, best_score = "GRAY", gray_box_roi, float(gray_score)
            choose_reason = "ONLY_GRAY_OK"
        elif edges_ok and gray_ok:
            ce = _center_of_box(edges_box_roi)
            cg = _center_of_box(gray_box_roi)
            d = _dist(ce, cg)
            dbg["agree_dist_px"] = float(d)

            if d <= float(self.cfg.agree_tol_px):
                if float(edges_score) >= float(gray_score):
                    best_method, best_box_roi, best_score = "EDGES", edges_box_roi, float(edges_score)
                    choose_reason = f"AGREE_PICK_EDGES d={d:.1f}"
                else:
                    best_method, best_box_roi, best_score = "GRAY", gray_box_roi, float(gray_score)
                    choose_reason = f"AGREE_PICK_GRAY d={d:.1f}"
            else:
                best_method, best_box_roi, best_score = "EDGES", edges_box_roi, float(edges_score)
                choose_reason = f"DISAGREE_PREFER_EDGES d={d:.1f}"
        else:
            if edges_box_roi is not None and float(edges_score) >= float(gray_score):
                best_method, best_box_roi, best_score = "EDGES", edges_box_roi, float(edges_score)
                choose_reason = "WEAK_BUT_EDGES_BEST"
            elif gray_box_roi is not None:
                best_method, best_box_roi, best_score = "GRAY", gray_box_roi, float(gray_score)
                choose_reason = "WEAK_BUT_GRAY_BEST"
            else:
                dbg["chosen"] = {"method": None, "reason": "NO_BOXES"}
                return None, float(max(edges_score, gray_score)), dbg

        best_client = _offset_box(best_box_roi, ox, oy)
        dbg["chosen"] = {"method": best_method, "reason": choose_reason, "score": float(best_score)}
        return best_client, float(best_score), dbg


def render_frog_roi_debug_image(client_bgr: np.ndarray, meta: Dict[str, Any]) -> np.ndarray:
    vis = client_bgr.copy()

    cbox = meta.get("center_roi_box", None)
    if cbox is not None:
        x, y, w, h = cbox
        cv2.rectangle(vis, (x, y), (x + w, y + h), (255, 180, 0), 2)
        cv2.putText(vis, "CENTER_ROI", (x, max(0, y - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 180, 0), 2, cv2.LINE_AA)

    trk = meta.get("track_roi_box", None)
    if trk is not None:
        x, y, w, h = trk
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 255), 2)
        cv2.putText(vis, "TRACK_ROI", (x, max(0, y - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)

    chosen = meta.get("chosen", {})
    cm = chosen.get("method", None)
    cr = chosen.get("reason", "")
    cv2.putText(vis, f"CHOSEN={cm}  {cr}", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    e = meta.get("edges", {})
    g = meta.get("gray", {})
    es = float(e.get("score", 0.0) or 0.0)
    gs = float(g.get("score", 0.0) or 0.0)
    cv2.putText(vis, f"edges={es:.3f}   gray={gs:.3f}", (10, 58),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2, cv2.LINE_AA)

    rr = meta.get("roi_reason", "")
    cv2.putText(vis, f"ROI_REASON={rr}", (10, 88),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2, cv2.LINE_AA)

    return vis
