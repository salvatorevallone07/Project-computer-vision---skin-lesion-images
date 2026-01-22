import cv2 as cv
import numpy as np

def extract_color(image_rgb, mask):
    features = {}
    lab = cv.cvtColor(image_rgb, cv.COLOR_BGR2LAB)
    lesion_pixel = mask > 0
    if np.sum (lesion_pixel) == 0:
        for ch in ["1", "a", "b"]:
            features[f"{ch}_mean"] = 0
            features[f"{ch}_std"] = 0
        return features
    l_channel, a_channel, b_channel = lab[:, :, 0][lesion_pixel], lab[:, :, 1][lesion_pixel], lab[:, :, 2][lesion_pixel]
    features["l_mean"] = np.mean(l_channel)
    features["l_std"] = np.std(l_channel)
    features["a_mean"] = np.mean(a_channel)
    features["a_std"] = np.std(a_channel)
    features["b_mean"] = np.mean(b_channel)
    features["b_std"] = np.std(b_channel)
    return features
