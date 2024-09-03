import os
from PIL import Image, ImageFilter
import numpy as np
import re
import tifffile as tiff

# Paths to the folders containing the channels
dapi_dir = r'D:\Chang_files\workspace\Qupath_proj\hdk_codex\run6_2nd_reg\registered_dapi'
cd20_dir = r'D:\Chang_files\workspace\Qupath_proj\hdk_codex\run6_2nd_reg\registered_cd20'
cd4_dir = r'D:\Chang_files\workspace\Qupath_proj\hdk_codex\run6_2nd_reg\registered_cd4'
bcl2_dir = r'D:\Chang_files\workspace\Qupath_proj\hdk_codex\run6_2nd_reg\registered_bcl2'
irf4_dir = r'D:\Chang_files\workspace\Qupath_proj\hdk_codex\run6_2nd_reg\registered_irf4'
cd15_dir = r'D:\Chang_files\workspace\Qupath_proj\hdk_codex\run6_2nd_reg\registered_cd15'
pax5_dir = r'D:\Chang_files\workspace\Qupath_proj\hdk_codex\run6_2nd_reg\registered_pax5'
pd1_dir = r'D:\Chang_files\workspace\Qupath_proj\hdk_codex\run6_2nd_reg\registered_pd1'

# Output directory for concatenated images
output_dir = r'D:\Chang_files\workspace\Qupath_proj\hdk_codex\run6_2nd_reg\multi_channel_images'
os.makedirs(output_dir, exist_ok=True)

# Function to extract the common prefix from the filename
def extract_prefix(filename):
    match = re.match(r"(.+)_channel_\d+.tif", filename)
    if match:
        return match.group(1)
    return None

# Function to normalize the image intensity using contrast stretching
def normalize_image(image, lower_percentile=0, upper_percentile=100):
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

# Function to load, normalize, and concatenate channels into a multi-channel image and save as TIFF
def concatenate_channels_to_multichannel_image(dapi_dir, cd20_dir, cd4_dir, bcl2_dir, irf4_dir, cd15_dir, pax5_dir, pd1_dir, output_dir):
    # Assuming all folders contain corresponding images for the same prefixes
    for file_name in os.listdir(dapi_dir):
        if file_name.endswith('.tif') or file_name.endswith('.tiff'):
            print(file_name)

            # Construct the corresponding filenames in each directory
            dapi_image_path = os.path.join(dapi_dir, file_name)
            cd20_image_path = os.path.join(cd20_dir, file_name)
            cd4_image_path = os.path.join(cd4_dir, file_name)
            bcl2_image_path = os.path.join(bcl2_dir, file_name)
            irf4_image_path = os.path.join(irf4_dir, file_name)
            cd15_image_path = os.path.join(cd15_dir, file_name)
            pax5_image_path = os.path.join(pax5_dir, file_name)
            pd1_image_path = os.path.join(pd1_dir, file_name)

            # Check if all corresponding files exist
            if not (os.path.exists(cd20_image_path) and os.path.exists(cd4_image_path) and
                    os.path.exists(bcl2_image_path) and os.path.exists(irf4_image_path) and
                    os.path.exists(cd15_image_path) and os.path.exists(pax5_image_path) and
                    os.path.exists(pd1_image_path)):
                print(f"Skipping {file_name}: Corresponding files not found in all channels.")
                continue

            # Load and normalize each channel image
            dapi_image = normalize_image(Image.open(dapi_image_path))
            cd20_image = normalize_image(Image.open(cd20_image_path))
            cd4_image = normalize_image(Image.open(cd4_image_path))
            bcl2_image = normalize_image(Image.open(bcl2_image_path))
            irf4_image = normalize_image(Image.open(irf4_image_path))
            cd15_image = normalize_image(Image.open(cd15_image_path))
            pax5_image = normalize_image(Image.open(pax5_image_path))
            pd1_image = normalize_image(Image.open(pd1_image_path))

            # Convert all images to numpy arrays
            dapi_array = np.array(dapi_image)
            cd20_array = np.array(cd20_image)
            cd4_array = np.array(cd4_image)
            bcl2_array = np.array(bcl2_image)
            irf4_array = np.array(irf4_image)
            cd15_array = np.array(cd15_image)
            pax5_array = np.array(pax5_image)
            pd1_array = np.array(pd1_image)

            # Stack them along the last axis to create a multi-channel image
            multi_channel_image = np.stack([cd20_array, cd4_array, dapi_array,
                                            bcl2_array, irf4_array, cd15_array,
                                            pax5_array], axis=-1)

            # Save the multi-channel image as a TIFF file
            output_file_name = file_name
            output_file_path = os.path.join(output_dir, output_file_name)
            tiff.imwrite(output_file_path, multi_channel_image, photometric='rgb')
            print(f"Saved {output_file_path}")

# Run the normalization and concatenation process
concatenate_channels_to_multichannel_image(dapi_dir, cd20_dir, cd4_dir, bcl2_dir, irf4_dir, cd15_dir, pax5_dir, pd1_dir, output_dir)

print("Normalization and multi-channel image concatenation complete.")
