import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple

def resize_images(image_paths: List[Path], target_size: Tuple[int, int] = (128, 128)):
    """
    Resizes images to the target size and centers them on a white canvas.

    Parameters:
    - image_paths (List[Path]): List of image file paths. Each path can be a pathlib.Path object.
    - target_size (Tuple[int, int]): The desired output size (width, height). Default is (128, 128).

    This function reads each image from the provided paths, resizes it to fit within the 
    target size while maintaining the aspect ratio, and then centers the resized image 
    on a white canvas of the target size. The final image is saved back to the original path,
    overwriting the existing file.

    Raises:
    - ValueError: If the image cannot be loaded from the provided path.
    """

    for image_path in image_paths:
        # Ensure the image path is converted to a string
        image_path_str = str(image_path)

        # Load the image
        image = cv2.imread(image_path_str)
        
        # Check if the image was successfully loaded
        if image is None:
            print(f"Failed to load image: {image_path}")
            continue

        # Get the original dimensions of the image
        original_height, original_width = image.shape[:2]

        # Calculate the scaling factor to fit the image within the target size
        scale = min(target_size[0] / original_width, target_size[1] / original_height)
        
        # Calculate the new dimensions for the resized image
        new_width = int(original_width * scale)
        new_height = int(original_height * scale)
        
        # Resize the image to the new dimensions
        resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        # Create a white canvas with the target size
        canvas = np.full((target_size[1], target_size[0], 3), 255, dtype=np.uint8)
        
        # Calculate the top-left corner where the resized image will be placed on the canvas
        x_offset = (target_size[0] - new_width) // 2
        y_offset = (target_size[1] - new_height) // 2
        
        # Place the resized image onto the canvas
        canvas[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized_image
        
        # Save the final image, overwriting the original file
        cv2.imwrite(image_path_str, canvas)
