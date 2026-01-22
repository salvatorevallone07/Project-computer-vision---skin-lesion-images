import cv2 as cv
import numpy as np

# Utilizzo solo la maschera => indipendente da colore o illuminazione


def extract_shape_feature (mask):
    features = {}
    binary_mask = (mask > 0).astype(np.uint8)
    contours, _ = cv.findContours(
        binary_mask,
        cv.RETR_TREE,
        cv.CHAIN_APPROX_SIMPLE
    )
    if len(contours) == 0:
        features["area"] = 0
        features["perimeter"] = 0
        features["circularity"] = 0
        features["eccentricity"] = 0
        features["solidity"] = 0
        return features

    contour = max (contours, key = cv.contourArea)
    area = cv.contourArea(contour)
    perimeter = cv.arcLength (contour, True)

    if (perimeter > 0):
        circularity = 4 * np.pi * area / (perimeter ** 2)
    else:
        circularity = 0

    eccentricity = 0
    if (len(contour) >= 5):
        ellipse = cv.fitEllipse (contour)
        (_, _), (Ma, ma), _ = ellipse
        if (ma > 0):
            eccentricity = np.sqrt (1 - (Ma/ma) ** 2)

    hull = cv.convexHull (contour)
    hull_area = cv.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 0

    features["area"] = area
    features["perimeter"] = perimeter
    features["circularity"] = circularity
    features["eccentricity"] = eccentricity
    features["solidity"] = solidity

    return features
