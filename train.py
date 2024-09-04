"""General-purpose training script for image-to-image translation.

This script works for various models (with option '--model': e.g., pix2pix, cyclegan, colorization) and
different datasets (with option '--dataset_mode': e.g., aligned, unaligned, single, colorization).
You need to specify the dataset ('--dataroot'), experiment name ('--name'), and model ('--model').

It first creates model, dataset, and visualizer given the option.
It then does standard network training. During the training, it also visualize/save the images, print/save the loss plot, and save models.
The script supports continue/resume training. Use '--continue_train' to resume your previous training.

Example:
    Train a CycleGAN model:
        python train.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
    Train a pix2pix model:
        python train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/train_options.py for more training options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import time
import torch
from options.train_options import TrainOptions
from options.test_options import ValOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
from post_process import validation_train
from torch.utils.data import random_split
from util import util
from data.aligned_dataset import AlignedDataset
import csv
from tqdm import tqdm
import os

if __name__ == '__main__':
    opt = TrainOptions().parse()   # get training options

    # Load the full dataset
    dataset_full = AlignedDataset(opt)
    dataset_size = len(dataset_full)  # get the number of images in the dataset.
    print('The total number of images = %d' % dataset_size)

    # Split the dataset into training and validation sets
    val_split_ratio = 0.1  # Adjust this ratio as needed
    n_val = int(val_split_ratio * dataset_size)
    n_train = dataset_size - n_val
    dataset_train, dataset_validation = random_split(dataset_full, [n_train, n_val])

    print('The number of training images = %d' % n_train)
    print('The number of validation images = %d' % n_val)

    dataset = torch.utils.data.DataLoader(
            dataset_train,
            batch_size=opt.batch_size,
            shuffle=not opt.serial_batches,
            num_workers=int(opt.num_threads))
    dataset_val = torch.utils.data.DataLoader(
            dataset_validation,
            batch_size=1,
            shuffle=False,
            num_workers=0)


    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers

    visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
    total_iters = 0                # the total number of training iterations

    f = open('./checkpoints/' + '%s/' % opt.name + 'validation_train.csv', 'w', encoding='utf-8', newline='') # record validation result
    csv_writer = csv.writer(f)
    csv_writer.writerow(['epoch', 'marker'])

    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        visualizer.reset()              # reset the visualizer: make sure it saves the results to HTML at least once every epoch
        for i, data in tqdm(enumerate(dataset), total=len(dataset), desc="Training Epoch %d" % epoch):  # inner loop within one epoch
            print('TRAINING')
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)         # unpack data from dataset and apply preprocessing
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights

            if total_iters % opt.display_freq == 0:   # display images on visdom and save images to a HTML file
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                if opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)
                if opt.display_id == 0:
                    visualizer.novisdom_plot_losses(epoch, float(epoch_iter) / dataset_size, losses, opt.n_epochs + opt.n_epochs_decay)

            if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

            iter_data_time = time.time()
        model.update_learning_rate()
        if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

        if epoch % opt.val_freq == 0:  #run validation on the validation set
            marker = 0
            with torch.no_grad():
                for i, data_val in tqdm(enumerate(dataset_val), total=len(dataset_val), desc="Validating Epoch %d" % epoch):
                    print('VALIDATION')
                    imgs = data_val['A']
                    truemasks = data_val['B']
                    imgs = imgs.to(device='cuda',dtype=torch.float)
                    net = getattr(model, 'net' + 'G')
                    maskpred = net(imgs)
                    #maskpred = maskpred.cpu().numpy()
                    #truemasks = truemasks.cpu().numpy()
                    score = validation_train(truemasks, maskpred)
                    marker += score
                csv_writer.writerow([epoch, marker/n_val])
        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
    f.close()
