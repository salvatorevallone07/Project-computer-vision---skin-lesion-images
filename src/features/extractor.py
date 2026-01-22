import cv2 as cv
import numpy as np
from src.features.shape import extract_shape_feature
from src.features.color import extract_color
from src.features.texture import extract_texture_features

def extract_features(image_rgb, mask):
    shape_features = extract_shape_feature(mask)
    color_features = extract_color(image_rgb, mask)
    texture_feats = extract_texture_features(image_rgb, mask)
    features = {**shape_features, **color_features, **texture_feats}
    feature_names = list(features.keys())
    feature_values = np.array(list(features.values()), dtype=np.float32)
    return feature_names, feature_values
