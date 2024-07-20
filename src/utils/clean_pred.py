# Code to extract relevant area from prediction tif, save them as jpg and delete extra data
import glob
import multiprocessing
import os
import re
import shutil
import sys

import numpy as np
import rasterio
from PIL import Image

# zone_of_interest = {"pkks_small": [(fromLEFT, fromTOP)]}
zone_of_interest = {
    "pkks_small": [(3500, 7100)],
    "bantchh_small": [(16000, 15000)],
    "angkor_small": [(15300, 1400)],
    "roluos": [(2100, 300)],
}
IMAGE_SIZE = 1024 * 2


def info_from_path(path):
    """Extracts epoch number and area name from a given file path.

    Args:
        path (str): The file path.

    Returns:
        tuple: A tuple containing the epoch number and area name.
    """
    # Get file name
    file_name = os.path.basename(path)
    # Parse name
    match = re.search(r"eval_(\d+)_(\w+?)_(\d+c?m_)?argmax", file_name)
    if match is None:
        return None, None
    epoch_number = int(match.group(1))
    area_name = match.group(2)

    # matching rule for raster_targets
    # match = re.search(r"(\w+?)_(\d+c?m_)?target", file_name)
    # epoch_number = 0  # int(match.group(1))
    # area_name = match.group(1)
    return epoch_number, area_name


def generate_image(path, epoch, aois):
    """Generate an image by cropping and merging multiple prediction tiff files.

    Args:
        path (str): The path to save the generated image.
        epoch (int): The epoch number.
        aois (dict): A dictionary containing the names and paths of the areas of interest.

    Returns:
        None
    """

    # Increase the limit for image size
    Image.MAX_IMAGE_PIXELS = None

    cropped_images = []
    for name, corners in zone_of_interest.items():
        if name not in aois:
            print(f"Area {name} not found in {path}")
            # Add black image
            cropped_images.append(
                3 * np.ones((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
            )
            continue

        path_tif = aois[name]
        # Open tif file
        image = rasterio.open(path_tif)
        # Read image data as a NumPy array
        image_data = image.read(1)

        for corner in corners:
            # Extract 1024x1024 pixel image starting from corner
            cropped_image_data = image_data[
                corner[1] : corner[1] + IMAGE_SIZE,
                corner[0] : corner[0] + IMAGE_SIZE,
            ]
            cropped_images.append(cropped_image_data)

    def assign_colors(images):
        """Assigns colors to the given images based on their pixel values.

        Parameters:
        images (list): A list of images represented as numpy arrays.

        Returns:
        list: A list of RGB images with assigned colors.
        """
        unique_colors = {
            0: (220, 150, 0),
            1: (20, 120, 220),
            2: (200, 0, 200),
            3: (255, 255, 255),
        }
        black_color = (0, 0, 0)

        rgb_images = []
        for image in images:
            pixels = np.array(
                [
                    unique_colors.get(value, black_color)
                    for value in image.flatten()
                ]
            )
            rgb_image = pixels.reshape(image.shape + (3,))
            rgb_images.append(rgb_image)
        return rgb_images

    # Convert cropped_images to RGB with unique colors
    cropped_images_rgb_class = assign_colors(cropped_images)

    # Save all images as a single JPG
    IMAGE_SIZE_B = IMAGE_SIZE + 10
    merged_image_data = np.zeros(
        (IMAGE_SIZE, IMAGE_SIZE_B * len(cropped_images_rgb_class), 3),
        dtype=np.uint8,
    )
    for i, img in enumerate(cropped_images_rgb_class):
        merged_image_data[
            :, i * IMAGE_SIZE_B : i * IMAGE_SIZE_B + IMAGE_SIZE, :
        ] = img

    # Load reference image
    ref_image_path = None
    possible_paths = [
        os.path.join(path, "../../../ref/pred_ref.png"),
        os.path.join(path, "../../../../ref/pred_ref.png"),
        os.path.join(path, "../../../../../ref/pred_ref.png"),
        os.path.join(path, "../../../../../../ref/pred_ref.png"),
    ]
    for possible_path in possible_paths:
        if os.path.exists(possible_path):
            ref_image_path = possible_path
            break

    if ref_image_path is not None:
        print("Reference image found, error will be highlighted in red")
        ref_image = Image.open(ref_image_path)
        ref_image_data = np.array(ref_image)

        # add red when prediction is different from reference
        difference = np.where(merged_image_data != ref_image_data)
        merged_image_data[difference[0], difference[1], :] = (
            merged_image_data[difference[0], difference[1], :] // 3
        ) * 2 + (85, 0, 0)
    else:
        print("No reference image found")

    merged_image = Image.fromarray(merged_image_data)
    merged_image.save(os.path.join(path, f"pred_{epoch}.png"))


def generate_images(path, file_dict):
    """Generate images using multiprocessing.

    Args:
        path (str): The path to the images.
        file_dict (dict): A dictionary containing epoch and aois information.

    Returns:
        None
    """
    max_processes = min(16, len(file_dict), multiprocessing.cpu_count())
    pool = multiprocessing.Pool(processes=max_processes)
    pool.starmap(
        generate_image,
        [(path, epoch, aois) for epoch, aois in file_dict.items()],
    )
    pool.close()
    pool.join()


def process_dir(path):
    """Process the directory containing prediction files from on training.

    Args:
        path (str): The path to the directory.

    Returns:
        None
    """
    # Create a dict(epoch:dict(name:path)) from all sub .tif files
    print(f"Exploring path {path}")
    file_dict = {}
    for file_path in glob.glob(path + "/*.tif"):
        # print(f"Exploring {file_path}")
        epoch_number, area_name = info_from_path(file_path)
        if epoch_number is None or area_name is None:
            print(f"Invalid file name {file_path}")
            continue
        if epoch_number not in file_dict:
            file_dict[epoch_number] = {}
        file_dict[epoch_number][area_name] = file_path

    if len(file_dict) == 0:
        print("No prediction files found")
        return

    print("Generating images...", end="")
    generate_images(path, file_dict)
    print("Done")
    print("Deleting extra files...", end="")
    # Delete extra prediction files
    # Only best and last prediction as well as generated jpg are kept
    last_epoch = max(int(epoch) for epoch in file_dict.keys())
    # look into ../checkpoints/ for best model epoch
    best_epoch = max(
        [
            int(file_name.split("_")[1].split(".")[0])
            for file_name in os.listdir(os.path.join(path, "../checkpoints/"))
            if len(file_name.split("_")) > 1
        ]
    )
    for file_path in glob.glob(path + "/*.tif"):
        epoch_number, area_name = info_from_path(file_path)
        if epoch_number != last_epoch and epoch_number != best_epoch:
            os.remove(file_path)

    # Delete eval_prepared directory
    eval_prepared_dir = os.path.join(path, "../eval_prepared")
    if os.path.exists(eval_prepared_dir):
        shutil.rmtree(eval_prepared_dir)
    print("Done")


def explore_directories(path):
    """Explore all subdirectories recurcively and process the 'preds' subdirectory if it exists.

    Args:
        path (str): The path to the root directory.

    Returns:
        None
    """
    for root, dirs, files in os.walk(path):
        if "preds" in dirs:
            subdir_path = os.path.join(root, "preds")
            # check if there is already png file
            if not len(glob.glob(subdir_path + "/*.png")) > 0:
                process_dir(subdir_path)


if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) > 0:
        Run_path = args[0]
    else:
        Run_path = "./logs/"
    explore_directories(Run_path)
