import os
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
from models.networks import define_D, GANLoss
from data.base_dataset import BaseDataset, get_params, get_transformA, get_transformB

# Helper function to load and preprocess single-channel images
def load_image(image_path, is_A=True, params=None):
    # Load the image using PIL without changing the number of channels
    image = Image.open(image_path)

    if is_A:
        # Get transform parameters and apply transform specific for A images
        transform = get_transformA(params, grayscale=False)
    else:
        # Get transform parameters and apply transform specific for B images
        transform = get_transformB(params, grayscale=True)

    # Apply the transformations to the image
    image = transform(image)

    return image.unsqueeze(0)  # Add batch dimension if not already present

# Function to load the discriminator model's weights
def load_network(net, D_weights, device):
    print(f'Loading the model from {D_weights}')
    state_dict = torch.load(D_weights, map_location=device)
    if isinstance(net, torch.nn.DataParallel):
        net = net.module
    net.load_state_dict(state_dict)
    return net

# Function to calculate losses
def calculate_losses(image_dir, D_weights, log_path, input_nc, ndf, netD, n_layers_D, norm, init_type, init_gain,
                     gan_mode, gpu_ids, lambda_L1=30):
    device = torch.device('cuda:{}'.format(gpu_ids[0]) if gpu_ids else 'cpu')
    netD = define_D(input_nc, ndf, netD, n_layers_D, norm, init_type, init_gain, gpu_ids)
    netD = load_network(netD, D_weights, device)
    netD = netD.to(device)
    netD.eval()

    criterionGAN = GANLoss(gan_mode).to(device)
    criterionL1 = nn.L1Loss().to(device)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])  # Normalize to [-1, 1] for single-channel (grayscale)
    ])

    total_L1_loss = 0.0
    total_GAN_loss = 0.0
    num_images = 0

    fake_images = [f for f in os.listdir(image_dir) if '_fake_B.tif' in f]
    for fake_image_name in fake_images:
        real_image_name_A = fake_image_name.replace('_fake_B.tif', '_real_A.tif')
        real_image_name_B = fake_image_name.replace('_fake_B.tif', '_real_B.tif')

        real_image_path_A = os.path.join(image_dir, real_image_name_A)
        real_image_path_B = os.path.join(image_dir, real_image_name_B)
        fake_image_path = os.path.join(image_dir, fake_image_name)

        if not (os.path.exists(real_image_path_A) and os.path.exists(real_image_path_B)):
            print(f"Matching real images not found for {fake_image_name}, skipping...")
            continue

        real_image_A = load_image(real_image_path_A, is_A=True).to(device)
        real_image_B = load_image(real_image_path_B, is_A=False).to(device)
        fake_image = load_image(fake_image_path, is_A=False).to(device)

        L1_loss = criterionL1(fake_image, real_image_B) * lambda_L1  # Applying the lambda factor

        # Concatenate real_A and fake_B for discriminator
        input_for_D = torch.cat((real_image_A, fake_image), 1)
        pred_fake = netD(input_for_D)
        GAN_loss = criterionGAN(pred_fake, True)

        total_L1_loss += L1_loss.item()
        total_GAN_loss += GAN_loss.item()
        num_images += 1

        print(f"Processed {fake_image_name}: L1 Loss: {L1_loss.item()}, GAN Loss: {GAN_loss.item()}")

    avg_L1_loss = total_L1_loss / num_images
    avg_GAN_loss = total_GAN_loss / num_images

    with open(log_path, 'a') as log_file:
        log_file.write(f"Avg L1 Loss: {avg_L1_loss}, Avg GAN Loss: {avg_GAN_loss}\n")

    print(f"Average L1 Loss: {avg_L1_loss}, Average GAN Loss: {avg_GAN_loss}")

if __name__ == '__main__':
    config = {
        'image_dir': './results/hl_swinTResnet_dapi_2/test_55/images/',
        'D_weights': './checkpoints/hl_swinTResnet_dapi_2/55_net_D.pth',
        'log_path': './results/hl_swinTResnet_dapi_2/test_55/loss_log_test.txt',
        'input_nc': 4,
        'ndf': 64,
        'netD': 'basic',
        'n_layers_D': 3,
        'norm': 'batch',
        'init_type': 'normal',
        'init_gain': 0.02,
        'gan_mode': 'lsgan',
        'gpu_ids': [0],
    }

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
