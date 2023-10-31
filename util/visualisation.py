"""
Functionality to visualise multi-channel fluorescent images,
where each channel contains the signal of a different stain.
"""
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def display_composite(img, colours=None):
    """
    Display composite view of multi-channel image.
    Background is black. Each stain is rendered with a different colour.
    The stain's intensity controls colour transparency.
    Parameters:
    -----------
    img: np.array, shape(n_channels, rows, cols), np.float
        A multi-channel image - all intensities are positive
        floats
    colours: np.array(dtype = int), shape(n_channels, 3)
        The RGB colour selected for each stain [0,255].
        If unspecified, default colours are provided for up to
        7 stains.
    Returns:
    --------
    composite_image: PIL Image
    """
    if colours is None:
        colours = np.array([
            [0, 0, 255],  # blue (usually DAPI)
            [255, 51, 153],  # magenta
            [255, 0, 0],  # red
            [255, 255, 0],  # yellow
            [51, 153, 255],  # cyan
            [255, 128, 0],  # orange
            [0, 204, 0]  # green
        ])

    back = np.zeros((img.shape[1], img.shape[2], 4)).astype(np.uint8)
    composite = Image.fromarray(back, 'RGBA')
    back_alpha = Image.new('L', (img.shape[2], img.shape[1]), 255)
    composite.putalpha(back_alpha)
    for i, c in enumerate(img):
        channel_image = np.zeros((img.shape[1], img.shape[2], 4))
        channel_image[..., 0:-1] = colours[i]
        channel_image = Image.fromarray(channel_image.astype(np.uint8), 'RGBA')
        c = Image.fromarray(c.astype(np.uint8))
        c.convert('L')
        channel_image.putalpha(c)
        composite = Image.alpha_composite(composite, channel_image)
    return composite


def montage(image, colours=None):
    """
    Display montage of all channels of multi-channel image.
    Background is black. Each stain is rendered with a different colour.
    Parameters:
    -----------
    img: np.array, shape(n_channels, rows, cols), np.float
        A multi-channel image - all intensities are positive
        floats
    colours: np.array(dtype = int), shape(n_channels, 3)
        The RGB colour selected for each stain [0,255].
        If unspecified, default colours are provided for up to
        7 stains.
    Returns:
    --------
    fig: plt figure
    """
    if colours is None:
        colours = np.array([
            [0, 0, 255],  # blue (usually DAPI)
            [255, 51, 153],  # magenta
            [255, 0, 0],  # red
            [255, 255, 0],  # yellow
            [51, 153, 255],  # cyan
            [255, 128, 0],  # orange
            [0, 204, 0]  # green
        ])

    n = image.shape[0]
    rows = round(n/2)
    cols = 2
    k = 0

    fig, ax = plt.subplots(
        rows,
        cols,
        figsize=(6*cols-0.6,6*rows-0.6),
        gridspec_kw={'wspace': 0, 'hspace': 0}
    )

    for i in range(rows):
        for j in range(cols):
            if k == n:
                ax[i,j].imshow(
                    np.zeros(shape=(image.shape[1], image.shape[2])),
                    cmap='gray'
                )
                ax[i,j].axis('off')
            else:
                c = display_composite(image[k,...][np.newaxis,...],
                                      colours=colours[k,...][np.newaxis,...])
                ax[i,j].imshow(c)
                ax[i,j].axis('off')
            k += 1

    return fig
