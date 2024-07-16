# HEMIT: H&E to Multiplex-immunohistochemistry Image Translation with Dual-Branch Pix2pix Generator

## Description

This repository contains the source code of the proposed network and evaluation scheme for our submission. It has been anonymized for double-blind review purposes. The final version of the project will be updated and made publicly available upon acceptance and preparation of the camera-ready version.

## Training Instructions

This section provides detailed instructions on how to train models using the provided scripts. The training process is flexible and supports various models and datasets.
This repository's structure and training scripts are highly based on the original [pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) project by Jun-Yan Zhu and colleagues. For a comprehensive list of common commands and additional options, please refer to their [documentation](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix).

### Prerequisites

Ensure you have the required dependencies installed. You can install the dependencies via `pip` using the `requirements.txt` file included in the repository:

```
pip install -r requirements.txt
```

### Example Training

```
python train.py --dataroot ./datasets/hemit/ --name hemit_SwinTResnet_New_2 --model pix2pix --direction AtoB --display_id 0 --lr 0.00003 --lambda_L1 30 --no_flip --netG SwinTResnet --n_epochs 50 --n_epochs_decay 30 --lr_policy step --batch_size 2 --loss_type L1 --val_freq 5
```

This command configures the training as follows:
- `--dataroot ./datasets/hemit/`: Sets the dataset directory.
- `--name hemit_SwinTResnet_New_2`: Names the training session.
- `--model pix2pix`: Specifies using the pix2pix model.
- `--direction AtoB`: Defines the direction of image translation.
- `--display_id 0`: Disables visdom visualization to run without a display.
- `--lr 0.00003`: Sets the learning rate.
- `--lambda_L1 30`: Adjusts the weight for the L1 loss.
- `--no_flip`: Disables random flipping of images during training.
- `--netG SwinTResnet`: Chooses a specific generator architecture.
- `--n_epochs 50`: Sets the number of epochs before starting decay.
- `--n_epochs_decay 30`: Specifies the number of epochs to linearly decay the learning rate to zero.
- `--lr_policy step`: Applies a step decay to the learning rate.
- `--batch_size 2`: Sets the batch size.
- `--loss_type L1`: Uses L1 loss for training.
- `--val_freq 5`: Runs validation every 5 epochs.

### Example testing

```
python test.py --dataroot ./datasets/hemit/ --name hemit_SwinTResnet_New --model pix2pix --direction AtoB --epoch 20 --num_test 945 --eval --netG SwinTResnet
```
This command configures the training as follows:
- `--dataroot ./datasets/hemit/`: Specifies the path to the dataset.
- `--name hemit_SwinTResnet_New`: Sets the name of the experiment from which the model weights will be loaded.
- `--epoch 20`: Loads the model saved at epoch 20.
- `--num_test 945`: Limits the number of test images to 945.
- `--eval`: Enables evaluation mode, which may change the behaviour of some layers like dropout.
- `--netG SwinTResnet`:  Specifies the generator architecture used.

```
python post_process.py --srcdir results/hemit_SwinTResnet_New/test_20/
```
This command calculates the evaluation metrics on the generated results and outputs a CSV file for documentation.
