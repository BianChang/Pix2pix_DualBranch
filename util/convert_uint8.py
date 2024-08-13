import SimpleITK as sitk
import os
from PIL import Image, ImageFilter
import numpy as np
import re

def convert_to_uint8(image):
    """
    Converts a SimpleITK image from single (float32) to uint8.
    The function rescales the intensity values to the range [0, 255].
    """
    # Normalize the image to the range [0, 1]
    image = sitk.RescaleIntensity(image, 0, 1)
    # Scale the image to the range [0, 255] and cast to uint8
    image = sitk.Cast(sitk.RescaleIntensity(image, 0, 255), sitk.sitkUInt8)
    return image

def normalize_image(image, lower_percentile=0.5, upper_percentile=99.5):
    """
    Normalize an image using contrast stretching and apply a median filter.
    """
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

def convert_images_in_folder(input_folder, output_folder):
    """
    Converts all single-type images in the input folder to uint8,
    normalizes them, and saves them to the output folder.
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Iterate over all files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".tif") or filename.endswith(".tiff"):
            # Construct the full file path
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            # Read the image
            image = sitk.ReadImage(input_path, sitk.sitkFloat32)

            # Convert the image to uint8
            image_uint8 = convert_to_uint8(image)

            # Convert the SimpleITK image to a NumPy array for normalization
            image_pil = Image.fromarray(sitk.GetArrayFromImage(image_uint8))

            # Normalize the image
            normalized_image_pil = normalize_image(image_pil)

            # Convert the normalized PIL image back to a SimpleITK image
            normalized_image_sitk = sitk.GetImageFromArray(np.array(normalized_image_pil))
            normalized_image_sitk.CopyInformation(image_uint8)  # Retain the original image metadata

            # Save the normalized and converted image
            sitk.WriteImage(normalized_image_sitk, output_path)

            print(f"Converted, normalized, and saved: {filename}")

if __name__ == "__main__":
    # Define the input and output directories
    input_folder = r'D:\Chang_files\workspace\Qupath_proj\hdk_codex\run6_hema'
    output_folder = r'D:\Chang_files\workspace\Qupath_proj\hdk_codex\run6_hema\uint8'

    # Convert all images in the input folder
    convert_images_in_folder(input_folder, output_folder)

    print("All images have been converted, normalized, and saved.")
