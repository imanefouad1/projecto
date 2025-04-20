import cv2
import numpy as np

def draw_flow_arrows(frame, points_old, points_new):
    for (x1, y1), (x2, y2) in zip(points_old, points_new):
        cv2.arrowedLine(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

def draw_dense_flow(flow):
    h, w = flow.shape[:2]
    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 1] = 255
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
