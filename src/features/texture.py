import cv2 as cv
import numpy as np

def extract_texture_features(image_rgb, mask):
    features = {}

    gray = cv.cvtColor(image_rgb, cv.COLOR_RGB2GRAY)
    gray_lesion = gray.copy()
    gray_lesion[mask == 0] = 0

    if np.sum(mask) == 0:
        for i in range(8):
            features[f"lbp_{i}"] = 0
        features["contrast"] = 0
        return features

    # Check the 8 near pixel
    lbp = np.zeros_like(gray_lesion, dtype=np.uint8)
    for dy in [-1,0,1]:
        for dx in [-1,0,1]:
            if dy == 0 and dx == 0:
                continue
            shifted = np.roll(np.roll(gray_lesion, dy, axis=0), dx, axis=1)
            lbp += ((shifted >= gray_lesion).astype(np.uint8))

    # histogram
    hist, _ = np.histogram(lbp[mask>0].ravel(), bins=np.arange(0,10), density=True)
    for i in range(8):
        features[f"lbp_{i}"] = hist[i]

    lesion_pixels = gray_lesion[mask>0]
    features["contrast"] = lesion_pixels.max() - lesion_pixels.min()

    return features
