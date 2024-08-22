import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from tqdm import tqdm

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

        # Convert images to numpy arrays
        real_data = np.array(real_img)
        fake_data = np.array(fake_img)

        # Process each channel
        for i, color in enumerate(['red', 'green', 'blue']):
            real_channel = real_data[:,:,i].flatten()
            fake_channel = fake_data[:,:,i].flatten()
            hist_real, _ = np.histogram(real_channel, bins=num_bins, range=(0, 256))
            hist_fake, _ = np.histogram(fake_channel, bins=num_bins, range=(0, 256))
            hist_data[color]['real'].append(hist_real)
            hist_data[color]['fake'].append(hist_fake)

            # Update the channel-specific matrix
            for j in range(num_bins):
                for k in range(num_bins):
                    matrices[color][j, k] += np.sum((real_channel // (256 // num_bins) == j) & (fake_channel // (256 // num_bins) == k))

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
        #plt.colorbar()

        tick_positions = np.arange(num_bins + 1) - 0.5  # This includes one extra for the last edge
        tick_labels = [f'{int(bin_edges[i])}' for i in range(num_bins)] + [
            '255']  # Adding the last label manually

        plt.xticks(tick_positions, tick_labels, rotation=45)
        plt.yticks(tick_positions, tick_labels)
        plt.tick_params(which='minor', size=0)  # Hide minor ticks
        plt.grid(True, which='major', color='white', linewidth=2)
        plt.xlabel('Predicted Intensity')
        plt.ylabel('True Intensity')
        #plt.title(f'Confusion Matrix - {key.capitalize()} Channel')

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

# Example usage
folder_path = r'D:\Chang_files\work_records\swinT\hl_SwinTResnet\test_55\images'
process_images(folder_path)
