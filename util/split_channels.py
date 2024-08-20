import os
from PIL import Image
import tifffile as tiff
import numpy as np

# Directory containing multi-channel TIFF images
input_dir = r'D:\Chang_files\workspace\Qupath_proj\hdk_codex\run6_mIHC'  # Change this to your folder path
output_dir = os.path.join(input_dir, 'processed_images')

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Function to process (rotate, flip) TIFF image and save the transformed image and its channels
def process_tiff(file_path, output_base_dir):
    with tiff.TiffFile(file_path) as tif:
        # Load the image array
        images = tif.asarray()

        # Initialize a list to store the transformed channels
        transformed_channels = []

        # If the image is grayscale with multiple channels (shape: (channels, height, width))
        if len(images.shape) == 3:
            num_channels = images.shape[0]
            for i in range(num_channels):
                # Select channel i
                channel_image = Image.fromarray(images[i])

                # Rotate 90 degrees to the right
                channel_image = channel_image.rotate(-90, expand=True)

                # Flip horizontally
                channel_image = channel_image.transpose(method=Image.FLIP_LEFT_RIGHT)

                # Flip vertically
                #channel_image = channel_image.transpose(method=Image.FLIP_TOP_BOTTOM)

                # Append the transformed channel to the list
                transformed_channels.append(np.array(channel_image))

            # Stack the channels back together into a multi-channel image
            transformed_image = np.stack(transformed_channels, axis=0)

            # Save the transformed multi-channel TIFF image
            base_name = os.path.basename(file_path)
            transformed_file_path = os.path.join(output_base_dir, f"transformed_{base_name}")
            tiff.imwrite(transformed_file_path, transformed_image)
            print(f"Saved transformed image {transformed_file_path}")

            # Save each channel in its own subfolder
            for i in range(num_channels):
                # Create a subfolder for the current channel
                channel_folder_name = f"channel_{i + 1}"
                channel_output_dir = os.path.join(output_base_dir, channel_folder_name)
                os.makedirs(channel_output_dir, exist_ok=True)

                # Define the output filename for the channel
                output_file_name = f"{os.path.splitext(base_name)[0]}_channel_{i + 1}.tif"
                output_file_path = os.path.join(channel_output_dir, output_file_name)

                # Save the transformed channel image
                channel_image = Image.fromarray(transformed_channels[i])
                channel_image.save(output_file_path)
                print(f"Saved {output_file_path}")

        else:
            print(f"Skipping {file_path}, not a multi-channel image.")

# Process each TIFF file in the directory
for file_name in os.listdir(input_dir):
    if file_name.endswith('.tif') or file_name.endswith('.tiff'):
        file_path = os.path.join(input_dir, file_name)
        process_tiff(file_path, output_dir)

print("Processing complete.")
