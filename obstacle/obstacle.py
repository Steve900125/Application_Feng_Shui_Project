from pathlib import Path
from typing import List, Tuple, Dict
import numpy as np
from PIL import Image
import cv2
import sys

# Path arrangement
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
sys.path.insert(0, str(ROOT))  # for import modules

from fengshui.item import Item  # Core class, very important

def apply_white_boxes(floor_plan: np.ndarray, items: List[Item]) -> np.ndarray:
    """
    Apply white boxes on the floor plan image for each item's bounding box with a 2% margin.

    Parameters:
    - floor_plan (np.ndarray): The binarized floor plan image where obstacles are to be highlighted.
    - items (List[Item]): List of items whose bounding boxes are to be expanded and applied as white boxes on the floor plan.

    Returns:
    - np.ndarray: The updated floor plan image with white boxes applied over the specified items.
    """
    height, width = floor_plan.shape[:2]

    for item in items:
        # Get the bounding box coordinates from the Item object
        x_min, y_min = int(item.x1), int(item.y1)
        x_max, y_max = int(item.x2), int(item.y2)

        # Calculate a 2% margin to expand the bounding box
        x_margin = int((x_max - x_min) * 0.02)
        y_margin = int((y_max - y_min) * 0.02)

        # Adjust the coordinates to include the margin, ensuring they stay within image boundaries
        x_min = max(0, x_min - x_margin)
        y_min = max(0, y_min - y_margin)
        x_max = min(width, x_max + x_margin)
        y_max = min(height, y_max + y_margin)

        # Set the pixels in the expanded bounding box area to white
        floor_plan[y_min:y_max, x_min:x_max] = 255

    return floor_plan


def floor_plan_binarization(image_path: Path) -> np.ndarray:
    """
    Binarize the floor plan image.

    Parameters:
    - image_path (Path): Path to the image file.

    Returns:
    - np.ndarray: Binarized image.
    """
    try:
        pil_image = Image.open(image_path)
        image = np.array(pil_image)
    except Exception as e:
        raise ValueError(f"Failed to load image from {image_path}: {e}")

    if image is None:
        raise ValueError(f"Failed to load image from {image_path}")

    # Check if the image is already grayscale or not
    if len(image.shape) == 3:  # If image has 3 channels, it's a color image
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply bilateral filter
    blur = cv2.bilateralFilter(image, 10, 100, 1000)

    # Apply morphological operations
    kernel = np.ones((3, 3), np.uint8)
    img_erode = cv2.erode(blur, kernel)
    img_dilate = cv2.dilate(img_erode, kernel)

    # Apply threshold to binarize the image
    ret, result = cv2.threshold(img_dilate, 50, 255, cv2.THRESH_BINARY)

    return result


def bresenham_line(x0: int, y0: int, x1: int, y1: int) -> List[Tuple[int, int]]:
    """
    Generate points in a line from (x0, y0) to (x1, y1) using Bresenham's algorithm.

    Parameters:
    - x0, y0, x1, y1 (int): Coordinates of the start and end points.

    Returns:
    - List[Tuple[int, int]]: List of points in the line.
    """
    points = []
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    while True:
        points.append((x0, y0))
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy
    return points

def points_check(floor_plan: np.ndarray, points_line: List[Tuple[int, int]], scan_range: int, orientation: str) -> int:
    """
    Check points along the line for obstacles and count black points.

    Parameters:
    - floor_plan (np.ndarray): Binarized floor plan image.
    - points_line (List[Tuple[int, int]]): List of points in the line.
    - scan_range (int): Range to scan around each point.
    - orientation (str): Orientation of the scan ('vertical' or 'horizontal').

    Returns:
    - int: Maximum number of black points found along the scan range.
    """
    half_range = round(scan_range / 2)
    max_black_point = 0

    for point in points_line:
        black_point_counter = 0
        if orientation == 'vertical':
            left = max(point[0] - half_range, 0)
            right = min(point[0] + half_range, floor_plan.shape[1])
            for n in range(left, right + 1):
                if floor_plan[point[1], n] == 0:
                    black_point_counter += 1
        elif orientation == 'horizontal':
            up = max(point[1] - half_range, 0)
            down = min(point[1] + half_range, floor_plan.shape[0])
            for n in range(up, down + 1):
                if floor_plan[n, point[0]] == 0:
                    black_point_counter += 1
        max_black_point = max(max_black_point, black_point_counter)
    return max_black_point

def items_obstacle_detect(image_path: Path, items: List[Item]) -> Dict[str, any]:
    """
    Detect obstacles between two items on the floor plan.

    Parameters:
    - image_path (Path): Path to the floor plan image.
    - items (List[Item]): List of two items to check between.

    Returns:
    - Dict[str, Any]: Dictionary containing the binarized image array, points line, and obstacle rate.
    """

    result_dic ={
        'items' : [],
        'points_line' : [],
        'bin_image_np_arrary': None,
        'rate': 0.0
    }

    floor_plan = floor_plan_binarization(image_path)
    start = items[0].get_center()
    end = items[1].get_center()
    points_line = bresenham_line(int(start['center_X']), int(start['center_Y']), int(end['center_X']), int(end['center_Y']))

    if items[0].orientation != items[1].orientation:
        raise ValueError("Items do not have the same orientation")

    # Clean Items area to white
    floor_plan = apply_white_boxes(floor_plan=floor_plan, items=items)

    # Check obstacle
    scan_range = max(items[0].get_length_value(), items[1].get_length_value())
    max_black_point = points_check(floor_plan, points_line, scan_range, items[0].orientation)

    rate =  max_black_point / scan_range if scan_range > 0 else 0

    result_dic['items'] = items
    result_dic['bin_image_np_arrary'] = floor_plan
    result_dic['points_line'] = points_line
    result_dic['rate'] = rate

    return result_dic

if __name__ == "__main__":

    path = ROOT / 'images' / 'FloorPlan (2).jpg'
    items= [ Item (4.2847514152526855, 103.40266418457031, 83.41145324707031, 206.95797729492188,'entrance', 'horizontal'), 
            Item (335.0831604003906, 69.35748291015625, 541.446044921875, 300.3077392578125,'kitchen', 'horizontal')]
    res = items_obstacle_detect(items=items, image_path=path)

    print(res['rate'])

