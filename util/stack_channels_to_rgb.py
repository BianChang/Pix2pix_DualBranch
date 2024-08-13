import os
from PIL import Image, ImageFilter
import numpy as np
import re

# Paths to the three folders containing the channels
red_channel_dir = r'D:\Chang_files\workspace\Qupath_proj\hdk_codex\run4_mIHC\processed_images\channel_5'
green_channel_dir = r'D:\Chang_files\workspace\Qupath_proj\hdk_codex\run4_mIHC\processed_images\channel_9'
blue_channel_dir = r'D:\Chang_files\workspace\Qupath_proj\hdk_codex\run4_mIHC\processed_images\channel_1'

# Output directories for normalized images and final RGB images
output_dir = r'D:\Chang_files\workspace\Qupath_proj\hdk_codex\run4_mIHC\processed_images\RGB_3channels'
os.makedirs(output_dir, exist_ok=True)

norm_red_dir = os.path.join(output_dir, 'normalized_channel_5')
norm_green_dir = os.path.join(output_dir, 'normalized_channel_9')
norm_blue_dir = os.path.join(output_dir, 'normalized_channel_1')

os.makedirs(norm_red_dir, exist_ok=True)
os.makedirs(norm_green_dir, exist_ok=True)
os.makedirs(norm_blue_dir, exist_ok=True)

# Function to extract the common prefix from the filename
def extract_prefix(filename):
    match = re.match(r"(.+)_channel_\d+.tif", filename)
    if match:
        return match.group(1)
    return None

# Function to normalize the image intensity using contrast stretching
def normalize_image(image, lower_percentile=0.5, upper_percentile=99.5):
    img_array = np.array(image)

    # Compute the percentiles
    lower = np.percentile(img_array, lower_percentile)
    upper = np.percentile(img_array, upper_percentile)

    # Apply contrast stretching
    normalized_array = np.clip((img_array - lower) * 255 / (upper - lower), 0, 255)

    # Convert back to PIL image
    normalized_image = Image.fromarray(normalized_array.astype(np.uint8))

    # Apply slight sharpening to counteract any blurring effects
    normalized_image = normalized_image.filter(ImageFilter.MedianFilter(size=3))

    return normalized_image


# Function to standardize, save, and stack images into an RGB TIFF
def stack_channels_to_rgb(red_dir, green_dir, blue_dir, norm_red_dir, norm_green_dir, norm_blue_dir, output_dir):
    # Assuming all three folders contain corresponding images for the same prefixes
    for file_name in os.listdir(red_dir):
        if file_name.endswith('.tif') or file_name.endswith('.tiff'):
            # Extract the common prefix from the filename
            prefix = extract_prefix(file_name)
            if not prefix:
                continue

            # Construct the corresponding filenames in the other folders
            red_image_path = os.path.join(red_dir, file_name)
            green_image_path = os.path.join(green_dir, f"{prefix}_channel_9.tif")
            blue_image_path = os.path.join(blue_dir, f"{prefix}_channel_1.tif")

            # Check if all corresponding files exist
            if not (os.path.exists(green_image_path) and os.path.exists(blue_image_path)):
                print(f"Skipping {prefix}: Corresponding files not found in all channels.")
                continue

            # Load the red, green, and blue channel images
            red_image = Image.open(red_image_path)
            green_image = Image.open(green_image_path)
            blue_image = Image.open(blue_image_path)

            # Normalize the images using contrast stretching
            norm_red_image = normalize_image(red_image)
            norm_green_image = normalize_image(green_image)
            norm_blue_image = normalize_image(blue_image)

            # Save the normalized images
            norm_red_image.save(os.path.join(norm_red_dir, file_name))
            norm_green_image.save(os.path.join(norm_green_dir, f"{prefix}_channel_9.tif"))
            norm_blue_image.save(os.path.join(norm_blue_dir, f"{prefix}_channel_1.tif"))

            # Stack the normalized channels into an RGB image
            rgb_image = Image.merge("RGB", (norm_red_image, norm_green_image, norm_blue_image))

            # Save the RGB image with a new organized filename
            output_file_name = f"{prefix}_rgb.tif"
            output_file_path = os.path.join(output_dir, output_file_name)
            rgb_image.save(output_file_path)
            print(f"Saved {output_file_path}")

# Run the normalization and stacking process
stack_channels_to_rgb(red_channel_dir, green_channel_dir, blue_channel_dir, norm_red_dir, norm_green_dir, norm_blue_dir, output_dir)

print("Normalization and RGB stacking complete.")
