import cv2
import numpy as np


def non_max_suppression(boxes, scores, overlap=0.3):
    if not boxes:
        return [], []

    boxes = np.array(boxes)
    scores = np.array(scores)
    idxs = np.argsort(scores)

    pick = []
    while len(idxs) > 0:
        i = idxs[0]
        pick.append(i)

        xx1 = np.maximum(boxes[i, 0], boxes[idxs[1:], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[idxs[1:], 1])
        xx2 = np.minimum(boxes[i, 0] + boxes[i, 2],
                          boxes[idxs[1:], 0] + boxes[idxs[1:], 2])
        yy2 = np.minimum(boxes[i, 1] + boxes[i, 3],
                          boxes[idxs[1:], 1] + boxes[idxs[1:], 3])

        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)

        overlap_area = (w * h) / (boxes[idxs[1:], 2] * boxes[idxs[1:], 3])
        idxs = np.delete(idxs, np.where(overlap_area > overlap)[0] + 1)
        idxs = np.delete(idxs, 0)

    return boxes[pick].tolist(), scores[pick].tolist()


def detect_ends(frame, skull_mask_path):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(
        hsv,
        np.array([20, 80, 30]),
        np.array([60, 255, 255])
    )

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, 2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, 2)

    skull = cv2.imread(skull_mask_path, cv2.IMREAD_GRAYSCALE)
    _, skull = cv2.threshold(skull, 127, 255, cv2.THRESH_BINARY)

    res = cv2.matchTemplate(mask, skull, cv2.TM_SQDIFF_NORMED)
    h, w = skull.shape

    ys, xs = np.where(res <= 0.35)

    boxes, scores = [], []
    for y, x in zip(ys, xs):
        boxes.append((x, y, w, h))
        scores.append(res[y, x])

    boxes, scores = non_max_suppression(boxes, scores)

    if len(boxes) == 0:
        min_val, _, min_loc, _ = cv2.minMaxLoc(res)
        return [(min_loc[0], min_loc[1], w, h)], [min_val], True

    return boxes, scores, False
