from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

@dataclass
class BallsDetectorConfig:
    ignore_top_ui_px: int = 55
    ignore_frog_radius: int = 95

    ignore_menu_box: bool = True
    menu_box_w: int = 210
    menu_box_h: int = 90

    ignore_right_queue: bool = True
    ignore_right_queue_w: int = 130
    ignore_right_queue_top: int = 70
    ignore_right_queue_bottom: int = 70

    hough_dp: float = 1.20
    hough_min_dist: int = 28
    hough_param1: int = 110
    hough_param2: int = 22
    hough_min_r: int = 14
    hough_max_r: int = 28

    color_hue_centers: Dict[str, int] = None

    min_s: int = 70
    min_v: int = 70

    pure_dist: int = 14
    pure_min_ratio: float = 0.38
    max_color_center_dist: int = 22

    min_valid_ring_px: int = 45

    keep_sep_ratio: float = 0.75

    use_mst_fallback: bool = True
    mst_fallback_min_ratio: float = 0.70
    order_jump_mult: float = 3.0


def default_cfg():
    cfg = BallsDetectorConfig()
    cfg.color_hue_centers = {
        "red": 0,
        "yellow": 25,
        "green": 60,
        "blue": 110,
        "purple": 145,
    }
    return cfg

def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def dist(a, b) -> float:
    return float(math.hypot(a[0] - b[0], a[1] - b[1]))


def hue_circ_dist(h: np.ndarray, c: int) -> np.ndarray:
    d = np.abs(h - c)
    return np.minimum(d, 180 - d)

def detect_circles_hough(client_bgr: np.ndarray, cfg: BallsDetectorConfig):
    gray = cv2.cvtColor(client_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (9, 9), 1.5)

    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT,
        dp=cfg.hough_dp,
        minDist=cfg.hough_min_dist,
        param1=cfg.hough_param1,
        param2=cfg.hough_param2,
        minRadius=cfg.hough_min_r,
        maxRadius=cfg.hough_max_r
    )
    if circles is None:
        return []

    circles = np.round(circles[0, :]).astype(int).tolist()
    circles.sort(key=lambda c: c[2], reverse=True)

    kept = []
    for x, y, r in circles:
        ok = True
        for kx, ky, kr in kept:
            if dist((x, y), (kx, ky)) <= cfg.keep_sep_ratio * max(r, kr):
                ok = False
                break
        if ok:
            kept.append((x, y, r))
    return kept


def classify_circle_color_with_score(
    client_bgr: np.ndarray,
    circle: Tuple[int, int, int],
    cfg: BallsDetectorConfig
):
    if cfg.color_hue_centers is None:
        raise ValueError("cfg.color_hue_centers is None (set it first)")

    x, y, r = circle
    h, w = client_bgr.shape[:2]

    x0 = clamp(x - r, 0, w - 1)
    x1 = clamp(x + r, 0, w - 1)
    y0 = clamp(y - r, 0, h - 1)
    y1 = clamp(y + r, 0, h - 1)

    roi = client_bgr[y0:y1 + 1, x0:x1 + 1]
    if roi.size == 0:
        return None, 0.0

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    H = hsv[..., 0].astype(np.int16)
    S = hsv[..., 1]
    V = hsv[..., 2]

    inner = int(r * 0.45)
    outer = int(r * 0.85)

    yy, xx = np.ogrid[:roi.shape[0], :roi.shape[1]]
    cx, cy = (x - x0), (y - y0)
    d2 = (xx - cx) ** 2 + (yy - cy) ** 2
    ring = (d2 <= outer ** 2) & (d2 > inner ** 2)

    valid = ring & (S > cfg.min_s) & (V > cfg.min_v)
    cnt = int(np.count_nonzero(valid))
    if cnt < cfg.min_valid_ring_px:
        return None, 0.0

    hue_vals = H[valid]
    hue_med = int(np.median(hue_vals))

    best = None
    best_d = 1e9
    for cname, hc in cfg.color_hue_centers.items():
        d = int(np.min(hue_circ_dist(hue_med, hc)))
        if d < best_d:
            best_d = d
            best = cname

    if best is None or best_d > cfg.max_color_center_dist:
        return None, 0.0

    hc = cfg.color_hue_centers[best]
    dists = hue_circ_dist(hue_vals, hc)
    purity = float(np.count_nonzero(dists <= cfg.pure_dist)) / (cnt + 1e-6)
    if purity < cfg.pure_min_ratio:
        return None, 0.0

    center_closeness = 1.0 - (best_d / (cfg.max_color_center_dist + 1e-6))
    cnt_norm = min(1.0, cnt / 220.0)
    score = 0.55 * purity + 0.30 * center_closeness + 0.15 * cnt_norm
    return best, float(score)


def classify_circle_color(client_bgr: np.ndarray, circle: Tuple[int, int, int], cfg: BallsDetectorConfig) -> Optional[str]:
    c, _ = classify_circle_color_with_score(client_bgr, circle, cfg)
    return c


def detect_chain_balls(
    client_bgr: np.ndarray,
    *,
    frog_center: Optional[Tuple[int, int]] = None,
    frog_r: Optional[int] = None,
    hole_center: Optional[Tuple[int, int]] = None,
    hole_r: Optional[int] = None,
    cfg: Optional[BallsDetectorConfig] = None,
):

    if cfg is None:
        cfg = default_cfg()

    h, w = client_bgr.shape[:2]
    circles = detect_circles_hough(client_bgr, cfg)
    balls: List[Dict] = []

    menu_rect = (w - cfg.menu_box_w, 0, w, cfg.menu_box_h)
    rq_x1 = w - cfg.ignore_right_queue_w
    rq_y1 = cfg.ignore_right_queue_top
    rq_y2 = h - cfg.ignore_right_queue_bottom

    ignore_frog_r = cfg.ignore_frog_radius
    if frog_r is not None:
        ignore_frog_r = max(cfg.ignore_frog_radius, int(1.05 * frog_r))

    for (x, y, r) in circles:
        if y < cfg.ignore_top_ui_px:
            continue

        if frog_center is not None and dist((x, y), frog_center) < ignore_frog_r:
            continue

        if cfg.ignore_menu_box:
            x1, y1, x2, y2 = menu_rect
            if x1 <= x <= x2 and y1 <= y <= y2:
                continue

        if hole_center is not None and hole_r is not None:
            if dist((x, y), hole_center) <= hole_r:
                continue

        if cfg.ignore_right_queue and (x >= rq_x1) and (rq_y1 <= y <= rq_y2):
            continue

        color, score = classify_circle_color_with_score(client_bgr, (x, y, r), cfg)
        if color is None:
            continue

        balls.append({
            "center": (int(x), int(y)),
            "r": int(r),
            "color": color,
            "score": float(score)
        })

    return balls


def build_neighbors(points: List[Tuple[int, int]]):
    pts = np.array(points, dtype=np.float32)
    n = len(pts)
    if n < 2:
        return [[] for _ in range(n)], pts

    nn = []
    for i in range(n):
        mask = np.arange(n) != i
        dists = np.linalg.norm(pts[i] - pts[mask], axis=1)
        nn.append(float(np.min(dists)))
    neighbor_dist = float(np.median(nn))

    max_edge = neighbor_dist * 2.35
    neighbors = [[] for _ in range(n)]
    for i in range(n):
        d = np.linalg.norm(pts - pts[i], axis=1)
        idx = np.argsort(d)
        for j in idx[1:9]:
            if d[j] <= max_edge:
                neighbors[i].append(int(j))
    return neighbors, pts


def connected_components(neighbors):
    n = len(neighbors)
    seen = [False] * n
    comps = []
    for i in range(n):
        if seen[i]:
            continue
        stack = [i]
        seen[i] = True
        comp = []
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in neighbors[u]:
                if not seen[v]:
                    seen[v] = True
                    stack.append(v)
        comps.append(comp)
    return comps


def _knn_graph(pts, k=6):
    n = len(pts)
    adj = [[] for _ in range(n)]
    P = np.array(pts, dtype=np.float32)
    for i in range(n):
        d = np.linalg.norm(P - P[i], axis=1)
        idx = np.argsort(d)
        for j in idx[1:k + 1]:
            w = float(d[j])
            adj[i].append((j, w))
            adj[j].append((i, w))
    return adj


def _prim_mst(adj):
    import heapq
    n = len(adj)
    in_mst = [False] * n
    mst = [[] for _ in range(n)]
    heap = [(0.0, 0, -1)]
    while heap:
        w, u, p = heapq.heappop(heap)
        if in_mst[u]:
            continue
        in_mst[u] = True
        if p != -1:
            mst[u].append((p, w))
            mst[p].append((u, w))
        for v, ww in adj[u]:
            if not in_mst[v]:
                heapq.heappush(heap, (ww, v, u))
    return mst


def _dijkstra_farthest(tree_adj, start):
    import heapq
    n = len(tree_adj)
    distv = [1e18] * n
    parent = [-1] * n
    distv[start] = 0.0
    heap = [(0.0, start)]
    while heap:
        d, u = heapq.heappop(heap)
        if d != distv[u]:
            continue
        for v, w in tree_adj[u]:
            nd = d + w
            if nd < distv[v]:
                distv[v] = nd
                parent[v] = u
                heapq.heappush(heap, (nd, v))
    far = max(range(n), key=lambda i: distv[i])
    return far, distv, parent


def _tree_path(parent, end):
    path = []
    cur = end
    while cur != -1:
        path.append(cur)
        cur = parent[cur]
    path.reverse()
    return path


def mst_diameter_path(points, hole_center=None):
    if len(points) < 3:
        return list(range(len(points)))

    adj = _knn_graph(points, k=6)
    mst = _prim_mst(adj)

    a, _, _ = _dijkstra_farthest(mst, 0)
    b, _, parent_b = _dijkstra_farthest(mst, a)
    path = _tree_path(parent_b, b)

    if hole_center is not None:
        hc = np.array(hole_center, dtype=np.float32)
        Pa = np.array(points[path[0]], dtype=np.float32)
        Pb = np.array(points[path[-1]], dtype=np.float32)
        if np.linalg.norm(Pb - hc) < np.linalg.norm(Pa - hc):
            path = list(reversed(path))

    return path


def order_chain(
    balls: List[Dict],
    hole_center: Optional[Tuple[int, int]],
    cfg: Optional[BallsDetectorConfig] = None
):

    if cfg is None:
        cfg = default_cfg()

    if len(balls) < 6:
        return balls, None

    points = [b["center"] for b in balls]
    neighbors, pts = build_neighbors(points)
    comps = connected_components(neighbors)

    main = max(comps, key=len)
    main_set = set(main)

    if hole_center is not None:
        hc = np.array(hole_center, dtype=np.float32)
        start = min(main, key=lambda i: float(np.linalg.norm(pts[i] - hc)))
    else:
        start = main[0]

    order = [int(start)]
    used = {int(start)}
    prev = None
    cur = int(start)

    while len(order) < len(main):
        cands = [j for j in neighbors[cur] if (j in main_set and j not in used)]
        if not cands:
            remaining = [i for i in main if i not in used]
            if not remaining:
                break
            nxt = min(remaining, key=lambda i: float(np.linalg.norm(pts[i] - pts[cur])))
        else:
            if prev is None:
                nxt = min(cands, key=lambda i: float(np.linalg.norm(pts[i] - pts[cur])))
            else:
                v1 = pts[cur] - pts[prev]
                v1n = v1 / (np.linalg.norm(v1) + 1e-6)
                best = None
                best_cost = 1e9
                for j in cands:
                    v2 = pts[j] - pts[cur]
                    v2n = v2 / (np.linalg.norm(v2) + 1e-6)
                    cost = 1.0 - float(np.dot(v1n, v2n))
                    if cost < best_cost:
                        best_cost = cost
                        best = j
                nxt = int(best)

        order.append(int(nxt))
        used.add(int(nxt))
        prev, cur = cur, int(nxt)

    if cfg.use_mst_fallback and len(order) >= 10:
        chain_pts = [balls[i]["center"] for i in order]
        steps = [dist(chain_pts[i - 1], chain_pts[i]) for i in range(1, len(chain_pts))]
        if steps:
            med = float(np.median(steps))
            mx = float(np.max(steps))
            if med > 1e-6 and mx > cfg.order_jump_mult * med:
                main_pts = [balls[i]["center"] for i in main]
                path_local = mst_diameter_path(main_pts, hole_center=hole_center)
                if len(path_local) >= int(cfg.mst_fallback_min_ratio * len(main)):
                    order = [int(main[p]) for p in path_local]

    chain = [balls[i] for i in order]

    cum = [0.0]
    for i in range(1, len(chain)):
        cum.append(cum[-1] + dist(chain[i - 1]["center"], chain[i]["center"]))

    return chain, cum
