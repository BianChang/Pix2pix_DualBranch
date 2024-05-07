import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def save_error_map(gt_img, pred_img, output_path, channel_name):
    # Convert PIL images to numpy arrays for processing
    gt_array = np.array(gt_img)
    pred_array = np.array(pred_img)

    # Calculate the difference and determine the direction of the difference
    difference = np.abs(gt_array - pred_array)
    gt_bigger = gt_array > pred_array
    pred_bigger = gt_array < pred_array

    # Initialize an error map with the same dimensions and three channels for RGB
    error_map = np.zeros((gt_array.shape[0], gt_array.shape[1], 3), dtype=np.uint8)

    # Set colors based on which image has higher intensity
    # If GT has higher intensity, use magenta to indicate the areas (R=255, G=0, B=255)
    error_map[gt_bigger] = [255, 0, 255]  # Magenta
    # If Prediction has higher intensity, use cyan to indicate the areas (R=0, G=255, B=255)
    error_map[pred_bigger] = [0, 255, 255]  # Cyan

    # Adjust the intensity of the error map to reflect the magnitude of the difference
    # Scale the color intensity based on the maximum possible difference
    max_difference = 255  # Assuming 8-bit images
    scaled_difference = (difference / max_difference)
    error_map = (error_map * scaled_difference[:, :, None]).astype(np.uint8)

    # Convert the error map back to an image
    error_img = Image.fromarray(error_map)
    # Save the error map image
    error_img.save(output_path)
    print(f'Saved error map for {channel_name} at {output_path}')

    # Additionally save the original channel images with their respective color coding
    save_channel_image(gt_array, output_path.replace('.png', '_GT_original.png'), channel_name)
    save_channel_image(pred_array, output_path.replace('.png', '_Pred_original.png'), channel_name)

def Gen_overall_error_maps(real_img, fake_img, base_name, output_folder):
    gt_array = np.array(real_img)
    pred_array = np.array(fake_img)
    gt_array = np.sum(gt_array, axis=-1, dtype=np.uint8)
    pred_array = np.sum(pred_array, axis=-1, dtype=np.uint8)
    difference = np.abs(gt_array - pred_array)

    gt_bigger = gt_array > pred_array
    pred_bigger = gt_array < pred_array

    error_map = np.zeros((gt_array.shape[0], gt_array.shape[1], 3), dtype=np.uint8)

    # If GT has higher intensity, use magenta to indicate the areas (R=255, G=0, B=255)
    error_map[gt_bigger] = [255, 0, 255]  # Magenta
    # If Prediction has higher intensity, use cyan to indicate the areas (R=0, G=255, B=255)
    error_map[pred_bigger] = [0, 255, 255]  # Cyan

    # Adjust the intensity of the error map to reflect the magnitude of the difference
    # Scale the color intensity based on the maximum possible difference
    max_difference = 255  # Assuming 8-bit images
    scaled_difference = (difference / max_difference)
    error_map = (error_map * scaled_difference[:, :, None]).astype(np.uint8)

    # Convert the error map back to an image
    error_img = Image.fromarray(error_map)
    overall_errormap_path = os.path.join(output_folder, 'overall_errormap')
    if not os.path.exists(overall_errormap_path):
        os.makedirs(overall_errormap_path)
    error_img.save(os.path.join(overall_errormap_path, base_name + 'errormap.png'))


def save_channel_image(channel_array, output_path, channel_name):
    # Create a color image where the specified channel is in its original intensity and others are zeroed
    channel_image = np.zeros((channel_array.shape[0], channel_array.shape[1], 3), dtype=np.uint8)
    if channel_name == 'R':
        channel_image[:, :, 0] = channel_array  # Red channel
    elif channel_name == 'G':
        channel_image[:, :, 1] = channel_array  # Green channel
    elif channel_name == 'B':
        channel_image[:, :, 2] = channel_array  # Blue channel

    # Save the channel-specific image
    Image.fromarray(channel_image).save(output_path)


def process_images(folder_path, output_folder):
    # Iterate over files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith('real_B.tif'):
            base_name = filename.replace('real_B.tif', '')
            real_path = os.path.join(folder_path, filename)
            fake_path = os.path.join(folder_path, base_name + 'fake_B.tif')
            if os.path.exists(fake_path):
                real_img = Image.open(real_path)
                fake_img = Image.open(fake_path)
                Gen_overall_error_maps(real_img, fake_img, base_name, output_folder)
                channels = ['R', 'G', 'B']
                for i, channel_name in enumerate(channels):
                    # Extract channels
                    gt_channel = real_img.getchannel(i)
                    pred_channel = fake_img.getchannel(i)
                    # Define output path for error maps
                    channel_folder = os.path.join(output_folder, channel_name)
                    if not os.path.exists(channel_folder):
                        os.makedirs(channel_folder)
                    output_path = os.path.join(channel_folder, f'{base_name}{channel_name}_error_map.png')
                    # Save error map
                    save_error_map(gt_channel, pred_channel, output_path, channel_name)



# Example usage
folder_path = r'D:\Chang_files\work_records\swinT\insillico\test'
output_folder = r'D:\Chang_files\work_records\swinT\insillico\test_errormap'
process_images(folder_path, output_folder)
