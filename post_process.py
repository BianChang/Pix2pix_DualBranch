import os
import numpy as np
import csv
from skimage import io
from skimage.io import imread
from skimage.color import rgb2gray
from PIL import Image
import argparse
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio
import math

# this function is for read image,the input is directory name
def process_directory(directory_name, subdirname):
    for filename in os.listdir(r"./"+directory_name):
        if filename.endswith('B.tif'):
            img = imread(directory_name + "/" + filename)
            composite = tif_composite(img)
            outimgdir = './composite_rgb'
            if not os.path.exists(outimgdir + '/' + subdirname):
                os.mkdir(outimgdir + '/' + subdirname)
            composite.save(os.path.join(outimgdir, subdirname, filename))


def tif_composite(img):
    img = ((img + 1.0) / 2.0) * 255

    a = img[:, :, 0] #dapi
    b = img[:, :, 1] #CD3
    c = img[:, :, 2] #Panck
    rgb_img = np.zeros((1024, 1024, 3), 'uint8')
    rgb_img[:, :, 0] = c
    rgb_img[:, :, 1] = b
    rgb_img[:, :, 2] = a
    rgb_img = Image.fromarray(rgb_img)

    return(rgb_img)

def compute_ssim(directory_name):
    csv_path = os.path.join(directory_name, 'score.csv')
    f = open(csv_path, 'w', encoding='utf-8',newline='')
    csv_writer = csv.writer(f)
    csv_writer.writerow(['file_name', 'dapi', 'cd3', 'panck','average'])
    for filename in os.listdir(r"./" + directory_name):
        if filename.endswith('_fake_B.tif'):
            fake_mihc = imread(directory_name + "/" + filename)
            nosuff_name = filename[0:-11]
            real_mihc_name = filename[0:-10]+'real_B.tif'
            real_mihc = imread(directory_name + '/' + real_mihc_name)
            real_dapi =real_mihc[:, :, 0]
            real_cd3 = real_mihc[:, :, 1]
            real_panck = real_mihc[:, :, 2]

            fake_dapi = fake_mihc[:, :, 0]
            fake_cd3 = fake_mihc[:, :, 1]
            fake_panck = fake_mihc[:, :, 2]
            dapi_score = ssim(real_dapi, fake_dapi)
            cd3_score = ssim(real_cd3, fake_cd3)
            panck_score = ssim(real_panck, fake_panck)
            average_score = np.average([dapi_score,cd3_score,panck_score])
            csv_writer.writerow([nosuff_name,dapi_score,cd3_score,panck_score,average_score])
    f.close()


def compute_metrics(directory_name):
    csv_path = os.path.join(directory_name, 'score.csv')

    with open(csv_path, 'w', encoding='utf-8', newline='') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow([
            'file_name', 'ssim', 'pearson', 'psnr'
        ])

        for filename in os.listdir(directory_name):
            if filename.endswith('_fake_B.tif'):
                path_to_file = os.path.join(directory_name, filename)
                fake_image = imread(path_to_file)
                base_name = filename[:-11]
                real_image_name = base_name + '_real_B.tif'
                real_image_path = os.path.join(directory_name, real_image_name)
                real_image = imread(real_image_path)

                # Ensure images are 1-channel (grayscale)
                real_channel = real_image.astype(float)
                fake_channel = fake_image.astype(float)

                tiny = 1e-15  # tiny constant to avoid numerical issues
                real_channel[0, 0] += tiny
                fake_channel[0, 0] += tiny

                # Compute SSIM
                ssim_score = ssim(real_channel, fake_channel, data_range=255)

                # Compute Pearson correlation coefficient
                pearson_corr = np.corrcoef(real_channel.flatten(), fake_channel.flatten())[0, 1]

                # Compute PSNR
                psnr_score = peak_signal_noise_ratio(real_channel, fake_channel)

                # Write results to CSV
                csv_writer.writerow([
                    base_name, ssim_score, pearson_corr, psnr_score
                ])



def compute_dapi_ssim(directory_name):
    csv_path = os.path.join(directory_name, 'score.csv')
    #csv_path = csv_path.replace('\\', '/')
    f = open(csv_path, 'w', encoding='utf-8',newline='')
    csv_writer = csv.writer(f)
    csv_writer.writerow(['file_name', 'dapi'])
    for filename in os.listdir(r"./" + directory_name):
        if filename.endswith('_fake_B.tif'):
            fake_mihc = imread(directory_name + "/" + filename)
            nosuff_name = filename[0:-11]
            real_mihc_name = filename[0:-10]+'real_B.tif'
            #real_mihc_name = filename[0:-11] + '.tif'
            real_mihc = imread(directory_name + '/' + real_mihc_name)
            real_dapi =rgb2gray(real_mihc)
            fake_dapi = rgb2gray(fake_mihc)
            print(real_dapi.shape)
            print(fake_dapi.shape)
            dapi_score = ssim(real_dapi, fake_dapi)
            csv_writer.writerow([nosuff_name, dapi_score])
    f.close()

def compute_cd3_ssim(directory_name):
    csv_path = os.path.join(directory_name, 'score.csv')
    #csv_path = csv_path.replace('\\', '/')
    f = open(csv_path, 'w', encoding='utf-8',newline='')
    csv_writer = csv.writer(f)
    csv_writer.writerow(['file_name', 'cd3'])
    for filename in os.listdir(r"./" + directory_name):
        if filename.endswith('_fake_B.tif'):
            fake_mihc = imread(directory_name + "/" + filename)
            nosuff_name = filename[0:-11]
            real_mihc_name = filename[0:-10]+'real_B.tif'
            #real_mihc_name = filename[0:-11] + '.tif'
            real_mihc = imread(directory_name + '/' + real_mihc_name)
            real_cd3 = rgb2gray(real_mihc)
            fake_cd3 = rgb2gray(fake_mihc)
            cd3_score = ssim(real_cd3, fake_cd3)
            csv_writer.writerow([nosuff_name,cd3_score])
    f.close()

def compute_panck_ssim(directory_name):
    csv_path = os.path.join(directory_name, 'score.csv')
    #csv_path = csv_path.replace('\\', '/')
    f = open(csv_path, 'w', encoding='utf-8',newline='')
    csv_writer = csv.writer(f)
    csv_writer.writerow(['file_name', 'panck'])
    for filename in os.listdir(r"./" + directory_name):
        if filename.endswith('_fake_B.tif'):
            fake_mihc = imread(directory_name + "/" + filename)
            nosuff_name = filename[0:-11]
            real_mihc_name = filename[0:-10]+'real_B.tif'
            #real_mihc_name = filename[0:-11] + '.tif'
            real_mihc = imread(directory_name + '/' + real_mihc_name)
            real_panck = rgb2gray(real_mihc)
            fake_panck = rgb2gray(fake_mihc)
            panck_score = ssim(real_panck, fake_panck)
            csv_writer.writerow([nosuff_name,panck_score])
    f.close()

def validation_train(real_mihc, fake_mihc):
    # Normalize the values between 0 and 1
    real_mihc = (real_mihc + 1.0) / 2.0
    fake_mihc = (fake_mihc + 1.0) / 2.0

    # Convert tensors to NumPy arrays and scale to [0, 255] range
    real_mihc = (real_mihc.cpu().detach().numpy() * 255).astype(np.uint8)
    fake_mihc = (fake_mihc.cpu().detach().numpy() * 255).astype(np.uint8)

    # Squeeze out unnecessary dimensions and ensure channel dimension is handled
    real_mihc = np.squeeze(real_mihc)
    fake_mihc = np.squeeze(fake_mihc)

    # Since you now have only one channel, just compare the entire images directly
    real_channel = real_mihc
    fake_channel = fake_mihc

    # Calculate SSIM (Structural Similarity Index) for the single channel
    channel_score = ssim(real_channel, fake_channel, data_range=255)

    return channel_score

if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--srcdir", type=str, help="process this directory.")
    args = parser.parse_args()
    directory_name = args.srcdir
    compute_metrics(directory_name)
