from typing import Optional
from pathlib import Path
import numpy as np
import sys
import cv2

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  
SAVE_PATH_DEFAULT = ROOT / 'fengshui' / 'output'
sys.path.insert(0, str(ROOT))   # for import moduls 

from fengshui.item import Item

def draw_bounding_boxes(image_path:Path, item:Item, color:tuple=(0, 0, 255), thickness:int=2)->np.ndarray:
    """
    Draws bounding box on the image at the specified path using the coordinates from the given Item.

    Parameters:
    - image_path (Path): Path to the image file.
    - item (Item): An instance of the Item class containing bounding box coordinates.
    - color (tuple): Color of the bounding box in BGR format. Default is red (255, 0, 0).
    - thickness (int): Thickness of the bounding box lines. Default is 2.

    Returns:
    - image (numpy.ndarray): The image with the bounding box drawn on it.
    """
    image = cv2.imread(image_path)

    # Extract bounding box coordinates from the Item instance and convert to integers
    start_point = (int(item.x1), int(item.y1))
    end_point = (int(item.x2), int(item.y2))

    cv2.rectangle(image, start_point , end_point, color, thickness)

    return image

def save_to_image(image:np.ndarray, file_name:str='bounding.img'):
    """
    Saves the given image to the specified file path.

    Parameters:
    - image (np.ndarray): The image to be saved.
    - file_name (Optional[str]): The name of the file to save the image as. Default is 'bounding.jpg'.

    Returns:
    - No return
    """
    
    SAVE_PATH_DEFAULT.mkdir(parents=True, exist_ok=True)
    file_path = SAVE_PATH_DEFAULT / file_name
    cv2.imwrite(str(file_path), image)
