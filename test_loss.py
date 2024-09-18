import os
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from models.networks import define_D, GANLoss
from options.test_options import TestOptions

# Helper function to load and preprocess single-channel images
def load_image(image_path, transform):
    image = Image.open(image_path).convert('L')  # Convert to grayscale (single channel)
    image = transform(image)
    return image.unsqueeze(0)  # Add batch dimension

# Function to calculate losses
def calculate_losses(image_dir, D_weights, log_path, opt):
    # Set up the device
    device = torch.device('cuda:{}'.format(opt.gpu_ids[0]) if opt.gpu_ids else 'cpu')

    # Load the discriminator model
    netD = define_D(opt.input_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, opt.gpu_ids)
    netD.load_state_dict(torch.load(D_weights))
    netD = netD.to(device)
    netD.eval()

    # Define loss functions
    criterionGAN = GANLoss(opt.gan_mode).to(device)
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
    # Parse test options from options/test_options.py
    opt = TestOptions().parse()  # get test options

    # Set log file and directory paths from the command line arguments
    image_dir = opt.dataroot
    D_weights = os.path.join(opt.checkpoints_dir, opt.name, f"{opt.epoch}_net_D.pth")
    log_path = os.path.join(opt.results_dir, opt.name, 'loss_logs.txt')

    calculate_losses(image_dir, D_weights, log_path, opt)
