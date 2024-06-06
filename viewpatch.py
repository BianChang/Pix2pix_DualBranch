import os
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

# Define the folder path and edge length in the script
folder_path = r'C:\Users\chang\Desktop\CVPR figures\view\1'
edge_length = 100  # Set the edge length as needed

def extract_patches_from_images(folder_path, points, edge_length):
    patches_output_folder = os.path.join(folder_path, 'patches')
    annotated_output_folder = os.path.join(folder_path, 'annotated')
    os.makedirs(patches_output_folder, exist_ok=True)
    os.makedirs(annotated_output_folder, exist_ok=True)

    # Load a font
    font_size = 40
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif')):
            image_path = os.path.join(folder_path, filename)
            image = Image.open(image_path)
            draw = ImageDraw.Draw(image)

            for i, (x, y) in enumerate(points):
                # Extract patch
                patch = image.crop((x, y, x + edge_length, y + edge_length))
                patch_output_path = os.path.join(patches_output_folder, f'patch_{i+1}_{filename}')
                patch.save(patch_output_path)
                print(f'Patch {i+1} saved to {patch_output_path}')

                # Determine color and thickness
                color = "red" if filename.lower() == 'he.tif' else "yellow"
                thickness = 5

                # Annotate original image with a square and order number
                for t in range(thickness):  # thicker edge
                    draw.rectangle([x-t, y-t, x + edge_length+t, y + edge_length+t], outline=color)
                draw.text((x + 5, y + 5), str(i+1), fill=color, font=font)

            annotated_output_path = os.path.join(annotated_output_folder, f'annotated_{filename}')
            image.save(annotated_output_path)
            print(f'Annotated image saved to {annotated_output_path}')

def select_points(image_path, num_points=4):
    points = []

    def onclick(event):
        if len(points) < num_points:
            points.append((int(event.xdata), int(event.ydata)))
            plt.scatter(event.xdata, event.ydata, c='red', s=40)
            plt.draw()
        if len(points) == num_points:
            plt.close()

    image = Image.open(image_path)
    fig, ax = plt.subplots()
    ax.imshow(image)
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()

    return points

def main():
    # Select points from image named 'HE'
    he_image_path = os.path.join(folder_path, 'HE.tif')
    points = select_points(he_image_path)
    print(points)

    extract_patches_from_images(folder_path, points, edge_length)

if __name__ == "__main__":
    main()
