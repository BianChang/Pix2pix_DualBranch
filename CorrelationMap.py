import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from tqdm import tqdm
import tifffile as tiff
import cv2

def percentile_normalization(image, lower_percentile=0, upper_percentile=99.5):
    """
    Normalize the image using the specified lower and upper percentiles.
    Works for both grayscale (1-channel) and RGB (3-channel) images.
    """
    # Convert the image to a numpy array
    image_array = np.array(image)

    # Check if the image is 1-channel (grayscale) or 3-channel (RGB)
    if len(image_array.shape) == 2:  # Grayscale image
        lower = np.percentile(image_array, lower_percentile)
        upper = np.percentile(image_array, upper_percentile)
        image_array = np.clip(image_array, lower, upper)
        image_array = (image_array - lower) / (upper - lower) * 255
    elif len(image_array.shape) == 3 and image_array.shape[2] == 3:  # RGB image
        # Apply percentile normalization for each channel
        for i in range(image_array.shape[2]):
            lower = np.percentile(image_array[:, :, i], lower_percentile)
            upper = np.percentile(image_array[:, :, i], upper_percentile)
            image_array[:, :, i] = np.clip(image_array[:, :, i], lower, upper)
            image_array[:, :, i] = (image_array[:, :, i] - lower) / (upper - lower) * 255

    return Image.fromarray(image_array.astype(np.uint8))

def histogram_equalization(image):
    """
    Apply histogram equalization to all channels of the image.
    Works for both grayscale (1-channel) and RGB (3-channel) images.
    """
    # Convert the image to a numpy array
    image_array = np.array(image)

    # Check if the image is 1-channel (grayscale) or 3-channel (RGB)
    if len(image_array.shape) == 2:  # Grayscale image
        # Apply histogram equalization
        image_array = cv2.equalizeHist(image_array)
    elif len(image_array.shape) == 3 and image_array.shape[2] == 3:  # RGB image
        # Apply histogram equalization to each channel separately
        for i in range(3):
            image_array[:, :, i] = cv2.equalizeHist(image_array[:, :, i])
    image_array = image_array.astype(np.uint8)

    return Image.fromarray(image_array)


def process_images(folder_path):
    # List all files
    files = os.listdir(folder_path)
    # Filter out the relevant files
    real_files = sorted([f for f in files if 'real_B.tif' in f])
    fake_files = sorted([f for f in files if 'fake_B.tif' in f])

    # Initialize confusion matrices for each channel + overall
    num_bins = 10  # Example: 10 bins for 0.1 increments
    bin_edges = np.linspace(0, 255, num_bins + 1)
    matrices = {channel: np.zeros((num_bins, num_bins)) for channel in ['red', 'green', 'blue', 'overall']}
    hist_data = {channel: {'real': [], 'fake': []} for channel in ['red', 'green', 'blue']}

    # Process each pair of real and fake images
    for real_file, fake_file in zip(real_files, fake_files):
        print(real_file)
        real_img = Image.open(os.path.join(folder_path, real_file))
        fake_img = Image.open(os.path.join(folder_path, fake_file))

        # Apply percentile normalization
        #real_img = percentile_normalization(real_img)
        #fake_img = percentile_normalization(fake_img)
        real_img = histogram_equalization(real_img)
        fake_img = histogram_equalization(fake_img)

        # Convert images to numpy arrays
        real_data = np.array(real_img)
        fake_data = np.array(fake_img)

        # Process each channel
        for i, color in enumerate(['red', 'green', 'blue']):
            real_channel = real_data[:, :, i].flatten()
            fake_channel = fake_data[:, :, i].flatten()
            hist_real, _ = np.histogram(real_channel, bins=num_bins, range=(0, 256))
            hist_fake, _ = np.histogram(fake_channel, bins=num_bins, range=(0, 256))
            hist_data[color]['real'].append(hist_real)
            hist_data[color]['fake'].append(hist_fake)

            # Update the channel-specific matrix
            for j in range(num_bins):
                for k in range(num_bins):
                    matrices[color][j, k] += np.sum(
                        (real_channel // (256 // num_bins) == j) & (fake_channel // (256 // num_bins) == k))

        # Update the overall matrix (average of all channels)
        matrices['overall'] += (matrices['red'] + matrices['green'] + matrices['blue']) / 3

    # Normalize each matrix to sum to 100 and calculate Pearson correlation
    correlations = {}
    for key, matrix in matrices.items():
        total = np.sum(matrix)
        matrix /= total / 100  # Normalize to sum to 100
        real_hist = np.sum(hist_data[key]['real'], axis=0) if key != 'overall' else np.mean(
            [np.sum(hist_data[channel]['real'], axis=0) for channel in ['red', 'green', 'blue']], axis=0)
        fake_hist = np.sum(hist_data[key]['fake'], axis=0) if key != 'overall' else np.mean(
            [np.sum(hist_data[channel]['fake'], axis=0) for channel in ['red', 'green', 'blue']], axis=0)
        correlations[key] = pearsonr(real_hist, fake_hist)[0]

        # Plotting
        plt.figure(figsize=(5, 5), dpi=300)
        plt.imshow(matrix, cmap='turbo', interpolation='nearest')
        # plt.colorbar()

        tick_positions = np.arange(num_bins + 1) - 0.5  # This includes one extra for the last edge
        tick_labels = [f'{int(bin_edges[i])}' for i in range(num_bins)] + [
            '255']  # Adding the last label manually

        plt.xticks(tick_positions, tick_labels, rotation=45)
        plt.yticks(tick_positions, tick_labels)
        plt.tick_params(which='minor', size=0)  # Hide minor ticks
        plt.grid(True, which='major', color='white', linewidth=2)
        plt.xlabel('Predicted Intensity')
        plt.ylabel('True Intensity')
        # plt.title(f'Confusion Matrix - {key.capitalize()} Channel')

        # Add text annotations for proportions
        for j in range(num_bins):
            for k in range(num_bins):
                plt.text(k, j, f'{int(matrix[j, k])}', ha='center', va='center', color='white', fontsize=15)

        # Overlay Pearson Correlation
        plt.text(0.05, 0.05, f'ρ= {correlations[key]:.3f}', verticalalignment='bottom', horizontalalignment='left',
                 transform=plt.gca().transAxes, color='black', fontsize=15, bbox=dict(facecolor='white', alpha=0.8))

        # Save the figure
        plt.tight_layout()
        plt.savefig(os.path.join(folder_path, f'{key}_confusion_matrix.png'))
        plt.close()  # Close the plot to avoid displaying it in the notebook


def process_images_single(folder_path):
    # List all files
    files = os.listdir(folder_path)
    # Filter out the relevant files
    real_files = sorted([f for f in files if 'real_B.tif' in f])
    fake_files = sorted([f for f in files if 'fake_B.tif' in f])

    # Initialize confusion matrix for single channel
    num_bins = 10  # Example: 10 bins for 0.1 increments
    max_value = 255  # Set a custom maximum value for better normalization
    bin_edges = np.linspace(0, max_value, num_bins + 1)
    confusion_matrix = np.zeros((num_bins, num_bins))
    hist_data = {'real': [], 'fake': []}

    # Process each pair of real and fake images
    for real_file, fake_file in zip(real_files, fake_files):
        print(f"Processing: {real_file}")

        # Read single-channel images (assumed grayscale TIFF)
        real_img = Image.open(os.path.join(folder_path, real_file)).convert('L')
        fake_img = Image.open(os.path.join(folder_path, fake_file)).convert('L')

        # Convert images to numpy arrays
        real_data = np.array(real_img).flatten()
        fake_data = np.array(fake_img).flatten()

        # Clip the values to the desired max_value
        real_data = np.clip(real_data, 0, max_value)
        fake_data = np.clip(fake_data, 0, max_value)

        # Calculate histograms for the single channel
        hist_real, _ = np.histogram(real_data, bins=num_bins, range=(0, max_value))
        hist_fake, _ = np.histogram(fake_data, bins=num_bins, range=(0, max_value))
        hist_data['real'].append(hist_real)
        hist_data['fake'].append(hist_fake)

        # Update the confusion matrix for the single channel
        for j in range(num_bins):
            for k in range(num_bins):
                confusion_matrix[j, k] += np.sum(
                    (real_data // (max_value // num_bins) == j) & (fake_data // (max_value // num_bins) == k))

    # Normalize the confusion matrix to sum to 100 and calculate Pearson correlation
    total = np.sum(confusion_matrix)
    if total > 0:  # Avoid division by zero
        confusion_matrix /= total / 100  # Normalize to sum to 100

    # Use histograms (full arrays) for Pearson correlation, not the sum
    real_hist = np.sum(hist_data['real'], axis=0)  # Sum over all images for the histogram
    fake_hist = np.sum(hist_data['fake'], axis=0)  # Sum over all images for the histogram
    if len(real_hist) > 1 and len(fake_hist) > 1:  # Ensure valid arrays for correlation
        correlation = pearsonr(real_hist, fake_hist)[0]
    else:
        correlation = 0  # Set correlation to 0 if not computable

    # Plotting
    plt.figure(figsize=(5, 5), dpi=300)
    plt.imshow(confusion_matrix, cmap='turbo', interpolation='nearest')

    tick_positions = np.arange(num_bins + 1) - 0.5  # This includes one extra for the last edge
    tick_labels = [f'{int(bin_edges[i])}' for i in range(num_bins)] + [f'{int(max_value)}']  # Adding the last label manually

    plt.xticks(tick_positions, tick_labels, rotation=45)
    plt.yticks(tick_positions, tick_labels)
    plt.tick_params(which='minor', size=0)  # Hide minor ticks
    plt.grid(True, which='major', color='white', linewidth=2)
    plt.xlabel('Predicted Intensity')
    plt.ylabel('True Intensity')

    # Add text annotations for proportions
    for j in range(num_bins):
        for k in range(num_bins):
            plt.text(k, j, f'{int(confusion_matrix[j, k])}', ha='center', va='center', color='white', fontsize=15)

    # Overlay Pearson Correlation
    plt.text(0.05, 0.05, f'ρ= {correlation:.3f}', verticalalignment='bottom', horizontalalignment='left',
             transform=plt.gca().transAxes, color='black', fontsize=15, bbox=dict(facecolor='white', alpha=0.8))

    # Save the figure
    plt.tight_layout()
    plt.savefig(os.path.join(folder_path, 'confusion_matrix_single_channel.png'))
    plt.close()  # Close the plot to avoid displaying it in the notebook



# Example usage
folder_path = r'D:\Chang_files\work_records\swinT\resnetswinT\images'
process_images(folder_path)
