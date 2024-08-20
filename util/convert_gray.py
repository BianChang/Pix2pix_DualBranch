import os
from PIL import Image


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


if __name__ == "__main__":
    # Define the input and output directories
    input_folder = r'D:\Chang_files\workspace\Qupath_proj\hdk_codex\run4_HE'  # Replace with your input folder path
    output_folder = r'D:\Chang_files\workspace\Qupath_proj\hdk_codex\run4_HE\HE_gray'  # Replace with your output folder path

    # Convert images
    convert_rgb_to_grayscale(input_folder, output_folder)

    print("All images have been converted and saved.")
