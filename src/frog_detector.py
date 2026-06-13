import cv2
import math

def rotate_image(image, angle, border_value=0):
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(
        image, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value
    )

def preprocess_gray(gray):
    g = cv2.GaussianBlur(gray, (5, 5), 1)
    g = cv2.equalizeHist(g)
    return g

def load_template_gray(path):
    tpl = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if tpl is None:
        raise FileNotFoundError(f"لم يتم تحميل {path}")
    return preprocess_gray(tpl)

def load_template_edges(path):
    tpl = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if tpl is None:
        raise FileNotFoundError(f"لم يتم تحميل {path}")
    tpl = cv2.GaussianBlur(tpl, (5, 5), 1)
    return cv2.Canny(tpl, 80, 160)

def multi_angle_template_match_gray(gray_frame, template_gray, angles, scales):
    best = (-1, None, None, None, None)
    H, W = gray_frame.shape[:2]

    for scale in scales:
        tw = int(template_gray.shape[1] * scale)
        th = int(template_gray.shape[0] * scale)
        if tw < 8 or th < 8 or tw >= W or th >= H:
            continue

        tpl_scaled = cv2.resize(template_gray, (tw, th))

        for angle in angles:
            tpl_rot = rotate_image(tpl_scaled, angle)
            res = cv2.matchTemplate(gray_frame, tpl_rot, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)

            if max_val > best[0]:
                best = (max_val, max_loc, angle, scale, tpl_rot)

    return best

def multi_angle_template_match_edges(edges_frame, template_edges, angles):
    best = (-1, None, None, None)
    H, W = edges_frame.shape[:2]

    for angle in angles:
        tpl_rot = rotate_image(template_edges, angle)
        th, tw = tpl_rot.shape[:2]
        if tw >= W or th >= H:
            continue

        res = cv2.matchTemplate(edges_frame, tpl_rot, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)

        if max_val > best[0]:
            best = (max_val, max_loc, angle, tpl_rot)

    return best

def box_from_loc_tpl(loc, tpl):
    if loc is None or tpl is None:
        return None
    x, y = loc
    h, w = tpl.shape[:2]
    return (x, y, w, h)

def center_of_box(box):
    x, y, w, h = box
    return (x + w / 2, y + h / 2)

def dist2(a, b):
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2

def choose_frog_candidate(gray_cand, edges_cand, frame_shape, agree_tol_px=35):
    H, W = frame_shape[:2]
    center = (W / 2, H / 2)

    if gray_cand is None and edges_cand is None:
        return None, "NONE"
    if gray_cand and not edges_cand:
        return gray_cand, "ONLY_GRAY"
    if edges_cand and not gray_cand:
        return edges_cand, "ONLY_EDGES"

    cg = center_of_box(gray_cand["box"])
    ce = center_of_box(edges_cand["box"])
    d = math.sqrt(dist2(cg, ce))

    if d <= agree_tol_px:
        return (gray_cand if gray_cand["score"] >= edges_cand["score"] else edges_cand,
                f"AGREE d={d:.1f}")

    dg = math.sqrt(dist2(cg, center))
    de = math.sqrt(dist2(ce, center))
    return (gray_cand if dg <= de else edges_cand,
            f"DISAGREE d={d:.1f}")
