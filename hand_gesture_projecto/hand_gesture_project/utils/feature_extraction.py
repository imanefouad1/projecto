import numpy as np

def extract_histograms(flow_vectors):
    magnitudes = np.linalg.norm(flow_vectors, axis=1)
    angles = np.arctan2(flow_vectors[:, 1], flow_vectors[:, 0]) * 180 / np.pi
    angles = (angles + 360) % 360

    mag_hist, _ = np.histogram(magnitudes, bins=[0, 1, 3, 5, 10, 20])
    ang_hist, _ = np.histogram(angles, bins=[0, 45, 90, 135, 180, 225, 270, 315, 360])
    return np.concatenate([mag_hist, ang_hist])
