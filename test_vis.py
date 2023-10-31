import os
import torch
from options.test_options import TestOptions
from data import create_dataset
from models import create_model


def test_model(model, test_loader, device, output_dir):
    model.eval()
    outputs = []

    with torch.no_grad():
        print('Test:')
        for i, data in enumerate(test_loader):
            model.set_input(data)  # unpack data from data loader
            model.test()  # run inference
            visuals = model.get_current_visuals()  # get image results
            img_path = model.get_image_paths()  # get image paths
            print('processing (%04d)-th image... %s' % (i, img_path))
            save_outputs(visuals, img_path[0], output_dir)


def save_outputs(output, filename, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    base_filename = os.path.splitext(filename)[0]

    output_image = output['image']

    # Inverse normalization to bring values back to [0, 1]
    output_image = (output_image + 1) / 2.0

    # If you need to save it in the range [0, 255], you can do the following:
    output_image = (output_image * 255).astype(np.uint8)

    output_path = os.path.join(output_dir, f'{base_filename}.tif')

    # Save the image (assuming it's a PyTorch tensor)
    # If it's a numpy array, you should use a different save method (e.g., from PIL or OpenCV)
    torch.save(output_image, output_path)


def main():
    opt = TestOptions().parse()  # get test options
    opt.num_threads = 0
    opt.batch_size = 1
    opt.serial_batches = True
    opt.no_flip = True
    opt.display_id = -1

    dataset = create_dataset(opt)
    model = create_model(opt)
    model.setup(opt)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    output_dir = os.path.join(opt.results_dir, opt.name)
    test_model(model, dataset, device, output_dir)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--test_name', default='test', type=str, help='Name of the test trial')
    parser.add_argument('--test_path', default='./datasets/mIHC_1024/', type=str, help='Path to test data')
    parser.add_argument('--model_path', default='./checkpoints/',
                        type=str, help='Path to the pretrained model')
    parser.add_argument('--output_path', default='./results/', type=str,
                        help='Path to save the outputs')
    parser.add_argument('--visualized_output_path', default='./visualization/mihc_pix2pix_0108_lr0003_cosine_ls35',
                        type=str, help='Path to save the visualized outputs')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for testing')
    parser.add_argument('--direction', type=str, default='AtoB', help='Model transformation direction')
    parser.add_argument('--num_test', type=int, default=400, help='Number of test images')

    args = parser.parse_args()
    main()
