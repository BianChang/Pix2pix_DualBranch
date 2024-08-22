import os
import cv2
import numpy as np
import csv
from skimage.metrics import structural_similarity as ssim
from scipy.stats import pearsonr
from math import log10, sqrt


def calculate_metrics(real_image, fake_image):
    """Calculate SSIM, Pearson Correlation, and PSNR between two images."""

    # Initialize metrics
    ssim_values = []
    psnr_values = []
    pearson_values = []

    # Ensure the images have the same shape
    if real_image.shape != fake_image.shape:
        raise ValueError("Input images must have the same dimensions")

    # Calculate metrics for each channel
    for i in range(real_image.shape[2]):  # Iterate over each channel
        real_channel = real_image[:, :, i]
        fake_channel = fake_image[:, :, i]

        # SSIM
        ssim_value = ssim(real_channel, fake_channel, data_range=fake_channel.max() - fake_channel.min())
        ssim_values.append(ssim_value)

        # PSNR
        mse = np.mean((real_channel - fake_channel) ** 2)
        if mse == 0:
            psnr_value = float('inf')
        else:
            psnr_value = 20 * log10(255.0 / sqrt(mse))
        psnr_values.append(psnr_value)

        # Pearson Correlation
        pearson_value, _ = pearsonr(real_channel.flatten(), fake_channel.flatten())
        pearson_values.append(pearson_value)

    # Average metrics across channels
    ssim_avg = np.mean(ssim_values)
    psnr_avg = np.mean(psnr_values)
    pearson_avg = np.mean(pearson_values)

    return ssim_values, psnr_values, pearson_values, ssim_avg, psnr_avg, pearson_avg


def process_images(input_dir, output_csv):
    """Process all image pairs in the directory, calculate metrics, and save to CSV."""

    # Find all fake and real images
    real_images = [f for f in os.listdir(input_dir) if f.endswith("_real_B.tif")]
    fake_images = [f for f in os.listdir(input_dir) if f.endswith("_fake_B.tif")]

    # Open the CSV file for writing
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            "Base Name",
            "SSIM_R", "SSIM_G", "SSIM_B", "SSIM_Avg",
            "PSNR_R", "PSNR_G", "PSNR_B", "PSNR_Avg",
            "Pearson_R", "Pearson_G", "Pearson_B", "Pearson_Avg"
        ])

        for real_image_name in real_images:
            base_name = real_image_name.replace("_real_B.tif", "")
            fake_image_name = f"{base_name}_fake_B.tif"

            if fake_image_name in fake_images:
                real_image_path = os.path.join(input_dir, real_image_name)
                fake_image_path = os.path.join(input_dir, fake_image_name)

                # Read images
                real_image = cv2.imread(real_image_path, cv2.IMREAD_UNCHANGED)
                fake_image = cv2.imread(fake_image_path, cv2.IMREAD_UNCHANGED)

                # Calculate metrics
                ssim_values, psnr_values, pearson_values, ssim_avg, psnr_avg, pearson_avg = calculate_metrics(
                    real_image, fake_image)

                # Write the metrics to the CSV
                writer.writerow([
                    base_name,
                    ssim_values[0], ssim_values[1], ssim_values[2], ssim_avg,
                    psnr_values[0], psnr_values[1], psnr_values[2], psnr_avg,
                    pearson_values[0], pearson_values[1], pearson_values[2], pearson_avg
                ])

                print(
                    f"Processed {base_name}: SSIM_Avg={ssim_avg:.4f}, PSNR_Avg={psnr_avg:.2f}, Pearson_Avg={pearson_avg:.4f}")


if __name__ == "__main__":
    input_dir = r"D:\Chang_files\work_records\swinT\hl_SwinTResnet\test_55\images"
    output_csv = r"D:\Chang_files\work_records\swinT\hl_SwinTResnet\test_55\image_metrics.csv"

    process_images(input_dir, output_csv)
