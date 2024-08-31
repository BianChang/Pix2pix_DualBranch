import os
from PIL import Image, ImageFilter
import numpy as np


def convert_rgb_to_grayscale(input_folder, output_folder):
    """
    Converts all RGB TIFF images in the input folder to grayscale and saves them to the output folder.

    Parameters:
    input_folder (str): Path to the folder containing the RGB TIFF images.
    output_folder (str): Path to the folder where the grayscale images will be saved.
    """

    # Create the output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Iterate over all files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith('.tif') or filename.endswith('.tiff'):
            # Full path to the input file
            input_path = os.path.join(input_folder, filename)

            # Open the image
            with Image.open(input_path) as img:
                # Convert to grayscale
                grayscale_img = img.convert('L')

                # Full path to the output file
                output_path = os.path.join(output_folder, filename)

                # Save the grayscale image
                grayscale_img.save(output_path)

                print(f"Converted and saved: {filename}")

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
    #normalized_image = normalized_image.filter(ImageFilter.MedianFilter(size=11))

    return normalized_image

def process_images_in_folder(input_folder, output_folder, lower_percentile=0.5, upper_percentile=99.5):
    """
    Process all .tif images in the input folder, normalize them, and save them to the output folder.
    """
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Iterate over all files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith('.tif'):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            # Open the image
            image = Image.open(input_path)

            # Normalize the image
            normalized_image = normalize_image(image, lower_percentile, upper_percentile)

            # Save the normalized image to the output folder
            normalized_image.save(output_path)

            print(f"Processed and saved: {output_path}")

if __name__ == "__main__":
    # Define the input and output directories
    input_folder = r'D:\Chang_files\workspace\Qupath_proj\hdk_codex\run6_HE\Registered_HE'
    output_folder = r'D:\Chang_files\workspace\Qupath_proj\hdk_codex\run6_HE\Registered_HE\gray'

    # Convert images
    convert_rgb_to_grayscale(input_folder, output_folder)
    #process_images_in_folder(input_folder, output_folder)

    print("All images have been converted and saved.")
