import cv2
import numpy as np

def compute_lucas_kanade(prev_gray, gray, prev_points):
    next_points, status, error = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_points, None)
    good_new = next_points[status == 1]
    good_old = prev_points[status == 1]
    return good_old, good_new

def compute_farneback(prev_gray, gray):
    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 
                                        0.5, 3, 15, 3, 5, 1.2, 0)
    return flow
