import os
import cv2 as cv
import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

from src.dataset import ISIC2019Dataset
from src.preprocessing import preprocess_denoise, preprocess_postsegment
from src.segmentation import segment_lesion
from src.features.shape import extract_shape_feature
from src.features.color import extract_color
from src.feature_dataset import build_features_dataset, stratify_dataset_split, normalize_dataset
from src.classifier import Classifier
from src.features.texture import extract_texture_features


def download_dataset():
    import kagglehub
    print("Downloading ISIC 2019 dataset...")
    path = kagglehub.dataset_download("andrewmvd/isic-2019")
    print("Dataset downloaded to:", path)
    return path


def main():

    # Download dataset
    dataset_path = download_dataset()

    images_dir = os.path.join(
        dataset_path,
        "ISIC_2019_Training_Input",
        "ISIC_2019_Training_Input"
    )

    csv_file = os.path.join(
        dataset_path,
        "ISIC_2019_Training_GroundTruth.csv"
    )

    dataset = ISIC2019Dataset(
        root_dir=images_dir,
        csv_file=csv_file,
        transform=None
    )

    # Example access - load check
    image, label = dataset[0]
    print("Sample label:", label)
    print("Image shape:", image.shape)

    # Preprocessing and segmentation check
    image = preprocess_denoise(image, method="median")
    mask = segment_lesion(image)
    image_processed = preprocess_postsegment(image, mask)

    # Extract features check
    shape_features = extract_shape_feature(mask)
    print("Shape features:", shape_features)

    color_features = extract_color(image_processed, mask)
    print("Color features:", color_features)

    texture_features = extract_texture_features(image_processed, mask)
    print("Texture features:", texture_features)


    # cv.imshow("Original Image", cv.cvtColor(image, cv.COLOR_RGB2BGR))
    # cv.imshow("Lesion Mask", mask)
    # cv.imshow(
    #     "Processed Lesion",
    #     cv.cvtColor(image_processed, cv.COLOR_RGB2BGR)
    # )
    #
    # cv.waitKey(0)
    # cv.destroyAllWindows()

    # Dataset constract for the ML model
    X, Y, features_names = build_features_dataset(dataset)
    print ("Features matrix:", X.shape)
    print ("Labels:", Y.shape)

    X_train, Y_train, X_test, Y_test = stratify_dataset_split(X, Y)

    print ("Train set:", X_train.shape)
    print ("Test set:", X_test.shape)

    # Normalize train and test set
    X_train, X_test, mean, std = normalize_dataset(X_train, X_test)

    # Define and training model
    Xtr = torch.tensor(X_train, dtype=torch.float32)
    Ytr = torch.tensor(Y_train, dtype=torch.float32)
    Xte = torch.tensor(X_test, dtype=torch.float32)
    Yte = torch.tensor(Y_test, dtype=torch.float32)

    # Mini-batch training
    batch_size = 64
    train_dataset = TensorDataset(Xtr, Ytr)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model = Classifier(input_dim=Xtr.shape[1])
    num_pos = np.sum(Y_train)
    num_neg = len(Y_train) - num_pos
    pos_weight = torch.tensor([num_neg / num_pos])
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    epochs = 100
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch_x.size(0)

        epoch_loss /= len(train_loader.dataset)
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{epochs} - Loss: {epoch_loss:.4f}")

    model.eval()
    with torch.no_grad():
        logits = model(Xte)
        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).long()
    accuracy = (preds == Yte.long()).float().mean().item()
    print ("Accuracy:", accuracy)


if __name__ == "__main__":
    main()