import numpy as np
from src.preprocessing import preprocess_denoise, preprocess_postsegment
from src.segmentation import segment_lesion
from src.features.extractor import extract_features

def build_features_dataset (dataset):
    X = []
    Y = []
    features_names = None

    n_samples = len(dataset)

    for i in range (n_samples):
        image,label = dataset[i]
        image = preprocess_denoise(image)
        mask = segment_lesion(image)
        image = preprocess_postsegment(image, mask)

        names, features = extract_features(image, mask)

        if features_names is None:
            features_names = names
        X.append(features)
        Y.append(label)
    X  = np.array(X)
    Y = np.array(Y)
    return X, Y, features_names

def stratify_dataset_split (X, Y, test_ratio = 0.3, seed = 42):
    np.random.seed(seed)
    idx_0 = np.where(Y == 0)[0]
    idx_1 = np.where(Y == 1)[0]

    np.random.shuffle(idx_0)
    np.random.shuffle(idx_1)

    n_test_0 = int(len(idx_0)*test_ratio)
    n_test_1 = int(len(idx_1)*test_ratio)

    test_idx = np.concatenate((idx_0[:n_test_0],idx_1[:n_test_1]))
    train_idx = np.concatenate((idx_0[n_test_0:],idx_1[n_test_1:]))

    np.random.shuffle(train_idx)
    np.random.shuffle(test_idx)

    X_train = X[train_idx]
    Y_train = Y[train_idx]
    X_test = X[test_idx]
    Y_test = Y[test_idx]

    return X_train, Y_train, X_test, Y_test

def normalize_dataset (X_train, X_test, eps=1e-8):
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    X_train_norm = (X_train - mean) / (std + eps)
    X_test_norm = (X_test - mean) / (std + eps)

    return X_train_norm, X_test_norm, mean, std


