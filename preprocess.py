import os
import argparse
import torch
import tifffile as tiff
from tqdm import tqdm


def normalize_bandwise(image):
    """
    Normalize each spectral band independently to [-1, 1].

    Args:
        image (numpy.ndarray): Input image of shape [C, H, W]

    Returns:
        torch.Tensor: Normalized tensor
    """
    image = torch.from_numpy(image).float()

    for band_idx in range(image.shape[0]):

        band = image[band_idx]

        band_min = band.min()
        band_max = band.max()

        if band_max > band_min:
            image[band_idx] = (
                2.0 * (band - band_min)
                / (band_max - band_min)
                - 1.0
            )
        else:
            image[band_idx] = torch.zeros_like(band)

    return image


def build_file_dictionary(directory, prefix):
    """
    Creates a lookup dictionary for image files.

    Example:
        s1_12345.tif -> 12345
        s2_12345.tif -> 12345
    """

    file_dict = {}

    for filename in os.listdir(directory):

        if filename.endswith(".tif") and prefix in filename:

            key = filename.split(f"{prefix}_")[1]
            key = os.path.splitext(key)[0]

            file_dict[key] = filename

    return file_dict


def main():

    parser = argparse.ArgumentParser(
        description="Preprocess SAR and EO imagery for CycleGAN training"
    )

    parser.add_argument(
        "--sar_dir",
        required=True,
        help="Path to Sentinel-1 SAR TIFF files"
    )

    parser.add_argument(
        "--eo_dir",
        required=True,
        help="Path to Sentinel-2 EO TIFF files"
    )

    parser.add_argument(
        "--output_dir",
        default="data/processed",
        help="Directory to save processed tensors"
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Scanning dataset...")

    sar_files = build_file_dictionary(args.sar_dir, "s1")
    eo_files = build_file_dictionary(args.eo_dir, "s2")

    paired_keys = sorted(
        set(sar_files.keys()) &
        set(eo_files.keys())
    )

    print(f"Found {len(paired_keys)} valid SAR-EO image pairs.")

    for idx, key in enumerate(
        tqdm(paired_keys, desc="Processing")
    ):

        sar_path = os.path.join(
            args.sar_dir,
            sar_files[key]
        )

        eo_path = os.path.join(
            args.eo_dir,
            eo_files[key]
        )

        # Load TIFF files
        sar_image = tiff.imread(sar_path)
        eo_image = tiff.imread(eo_path)

        # Normalize
        sar_tensor = normalize_bandwise(sar_image)
        eo_tensor = normalize_bandwise(eo_image)

        # Save processed tensors
        torch.save(
            sar_tensor,
            os.path.join(
                args.output_dir,
                f"{idx:04d}_sar.pt"
            )
        )

        torch.save(
            eo_tensor,
            os.path.join(
                args.output_dir,
                f"{idx:04d}_eo.pt"
            )
        )

    print(
        f"Successfully processed and saved "
        f"{len(paired_keys)} image pairs."
    )


if __name__ == "__main__":
    main()
