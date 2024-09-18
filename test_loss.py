import os
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from models.networks import define_D, GANLoss

# Helper function to load and preprocess single-channel images
def load_image(image_path, transform):
    image = Image.open(image_path).convert('L')  # Convert to grayscale (single channel)
    image = transform(image)
    return image.unsqueeze(0)  # Add batch dimension

def remove_module_prefix(state_dict):
    """Remove the 'module.' prefix from keys if the model was not wrapped in DataParallel during saving."""
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            new_key = key[len("module."):]  # remove 'module.' prefix
        else:
            new_key = key
        new_state_dict[new_key] = value
    return new_state_dict

# Function to calculate losses
def calculate_losses(image_dir, D_weights, log_path, input_nc, ndf, netD, n_layers_D, norm, init_type, init_gain, gan_mode, gpu_ids):
    # Set up the device
    device = torch.device('cuda:{}'.format(gpu_ids[0]) if gpu_ids else 'cpu')

    # Load the discriminator model
    netD = define_D(input_nc, ndf, netD, n_layers_D, norm, init_type, init_gain, gpu_ids)
    state_dict = torch.load(D_weights)
    state_dict = remove_module_prefix(state_dict)
    netD.load_state_dict(torch.load(state_dict))
    netD = netD.to(device)
    netD.eval()

    # Define loss functions
    criterionGAN = GANLoss(gan_mode).to(device)
    criterionL1 = nn.L1Loss().to(device)

    # Set up the image transform (assuming the images are already in the correct size)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])  # Normalize to [-1, 1] for single-channel (grayscale)
    ])

    # Initialize variables to store total losses
    total_L1_loss = 0.0
    total_GAN_loss = 0.0
    num_images = 0

    # Iterate through real and fake images in the same directory
    real_images = [f for f in os.listdir(image_dir) if '_real_B.tif' in f]
    for real_image_name in real_images:
        fake_image_name = real_image_name.replace('_real_B.tif', '_fake_B.tif')

        real_image_path = os.path.join(image_dir, real_image_name)
        fake_image_path = os.path.join(image_dir, fake_image_name)

        if not os.path.exists(fake_image_path):
            print(f"Fake image not found for {real_image_name}, skipping...")
            continue

        # Load images
        real_image = load_image(real_image_path, transform).to(device)
        fake_image = load_image(fake_image_path, transform).to(device)

        # Calculate L1 loss
        L1_loss = criterionL1(fake_image, real_image)

        # Concatenate real_A and fake_B for discriminator
        input_for_D = torch.cat((real_image, fake_image), 1)
        pred_fake = netD(input_for_D)
        GAN_loss = criterionGAN(pred_fake, True)  # The goal is to make the fake image look real

        # Add losses to totals
        total_L1_loss += L1_loss.item()
        total_GAN_loss += GAN_loss.item()
        num_images += 1

        # Print the losses for each pair (optional)
        print(f"Processed {real_image_name}: L1 Loss: {L1_loss.item()}, GAN Loss: {GAN_loss.item()}")

    # Calculate average losses
    avg_L1_loss = total_L1_loss / num_images
    avg_GAN_loss = total_GAN_loss / num_images

    # Write the losses to the log file
    with open(log_path, 'a') as log_file:
        log_file.write(f"Avg L1 Loss: {avg_L1_loss}, Avg GAN Loss: {avg_GAN_loss}\n")

    print(f"Average L1 Loss: {avg_L1_loss}, Average GAN Loss: {avg_GAN_loss}")

if __name__ == '__main__':
    # Define your configuration here:
    config = {
        'image_dir': './results/hl_swinTResnet_dapi_2/test_55/images/',  # Path to your images
        'D_weights': './checkpoints/hl_swinTResnet_dapi_2/55_net_D.pth',  # Path to your discriminator weights
        'log_path': './results/hl_swinTResnet_dapi_2/test_55/loss_log_test.txt',  # Path to save the log file
        'input_nc': 1,  # Number of input channels (1 for grayscale)
        'ndf': 64,  # Number of filters in the discriminator
        'netD': 'basic',  # Discriminator type (e.g., 'basic', 'n_layers', etc.)
        'n_layers_D': 3,  # Number of layers in the discriminator (for 'n_layers' type)
        'norm': 'batch',  # Normalization type
        'init_type': 'normal',  # Initialization type for the network
        'init_gain': 0.02,  # Initialization gain
        'gan_mode': 'lsgan',  # GAN mode ('lsgan', 'vanilla', etc.)
        'gpu_ids': [0]  # List of GPU IDs to use
    }

    # Run the function to calculate losses
    calculate_losses(
        config['image_dir'],
        config['D_weights'],
        config['log_path'],
        config['input_nc'],
        config['ndf'],
        config['netD'],
        config['n_layers_D'],
        config['norm'],
        config['init_type'],
        config['init_gain'],
        config['gan_mode'],
        config['gpu_ids']
    )
