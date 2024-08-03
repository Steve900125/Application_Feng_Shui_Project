from typing import Optional, Tuple
from pathlib import Path
import numpy as np
import sys
import cv2

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  
OUTPUT_PATH_DEFAULT = ROOT / 'fengshui' / 'output'
sys.path.insert(0, str(ROOT))   # for import moduls 

from fengshui.item import Item

def draw_bounding_boxes(image_path: Optional[Path] = None, 
                        image: Optional[np.ndarray] = None, 
                        color: Tuple[int, int, int] = (0, 0, 255), 
                        item: Item = None, 
                        thickness: int = 2) -> np.ndarray:
    """
    Draws bounding box on the image at the specified path or on the provided image array using the coordinates from the given Item.

    Parameters:
    - image_path (Optional[Path]): Path to the image file. Either image_path or image must be provided.
    - image (Optional[np.ndarray]): The image array. Either image_path or image must be provided.
    - item (Item): An instance of the Item class containing bounding box coordinates.
    - color (Tuple[int, int, int]): Color of the bounding box in BGR format. Default is red (0, 0, 255).
    - thickness (int): Thickness of the bounding box lines. Default is 2.

    Returns:
    - np.ndarray: The image with the bounding box drawn on it.

    Raises:
    - ValueError: If neither image_path nor image is provided.
    """
    if image is None and image_path is None:
        raise ValueError("Either image_path or image must be provided.")
    
    if image is None:
        image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError("Image could not be loaded. Check the provided path or image array.")
    
    # Extract bounding box coordinates from the Item instance and convert to integers
    start_point = (int(item.x1), int(item.y1))
    end_point = (int(item.x2), int(item.y2))

    # Draw the rectangle on the image
    cv2.rectangle(image, start_point, end_point, color, thickness)

    return image

def save_to_image(image:np.ndarray, file_name:str='bounding.jpg'):
    """
    Saves the given image to the specified file path.

    Parameters:
    - image (np.ndarray): The image to be saved.
    - file_name (Optional[str]): The name of the file to save the image as. Default is 'bounding.jpg'.

    Returns:
    - No return
    """
    
    OUTPUT_PATH_DEFAULT.mkdir(parents=True, exist_ok=True)
    file_path = OUTPUT_PATH_DEFAULT / file_name
    cv2.imwrite(str(file_path), image)
