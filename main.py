import os
import cv2 as cv

from src.dataset import ISIC2019Dataset
from src.preprocessing import preprocess_denoise, preprocess_postsegment
from src.segmentation import segment_lesion


def download_dataset():
    import kagglehub
    print("Downloading ISIC 2019 dataset...")
    path = kagglehub.dataset_download("andrewmvd/isic-2019")
    print("Dataset downloaded to:", path)
    return path


def main():

    # 1. Download dataset (setup step)
    dataset_path = download_dataset()

    # 2. Define paths to images and ground truth
    images_dir = os.path.join(
        dataset_path,
        "ISIC_2019_Training_Input",
        "ISIC_2019_Training_Input"
    )

    csv_file = os.path.join(
        dataset_path,
        "ISIC_2019_Training_GroundTruth.csv"
    )

    # 3. Initialize PyTorch Dataset
    dataset = ISIC2019Dataset(
        root_dir=images_dir,
        csv_file=csv_file,
        transform=None  # transforms can be added later
    )

    print(f"Loaded {len(dataset)} samples")

    # 4. Example access (sanity check)
    image, label = dataset[0]
    print("Sample label:", label)
    print("Image shape:", image.shape)

    image = preprocess_denoise(image, method="median")
    mask = segment_lesion(image)
    image_processed = preprocess_postsegment(image, mask)

    cv.imshow("Original Image", cv.cvtColor(image, cv.COLOR_RGB2BGR))
    cv.imshow("Lesion Mask", mask)
    cv.imshow(
        "Processed Lesion",
        cv.cvtColor(image_processed, cv.COLOR_RGB2BGR)
    )

    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()