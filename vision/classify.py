from ultralytics.engine.results import Results # type: ignore
from typing import List, Optional
from ultralytics import YOLO # type: ignore
from pathlib import Path
import re
import sys

# Path arrangement
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  
sys.path.insert(0, str(ROOT))  

from vision.resize import resize_images



def extract_suffix(file_name):
    # Find the numeric suffix using regex
    match = re.search(r'(\d+)\.jpg$', file_name)
    return int(match.group(1)) if match else -1

def get_class_name(result:Results):
    return result.names[result.probs.top1]

def object_orientation_classify(root: Path, model_path: Path, object_name: str, result: Results) -> Optional[List]:

    # To find item image crops path from YOLOv8
    item_crops_path = root / Path(result.save_dir) / 'crops' / object_name
    
    # Check if the object has been detected
    not_empty = False
    for object_id in result.boxes.cls.tolist():
        if result.names[object_id] == object_name:
            not_empty = True

    if not_empty is False:
        return None

    source_path = Path(result.path)
    image_base_name = source_path.stem  # stem gives the base name without suffix

    # < WARRING > In macos m1 result order is not same as windows system
    # Find matching image files
    matching_files = list(item_crops_path.glob(f"{image_base_name}*.jpg"))

    # Sort files by the numeric suffix at the end of the filename
    matching_files_sorted = sorted(matching_files, key=lambda x: extract_suffix(x.name))

    # Resize image
    #resize_images_files = [Path(file) for file in matching_files_sorted]
    #resize_images(image_paths=resize_images_files)
    
    if item_crops_path.exists and not_empty:
        model = YOLO(model_path)  # pretrained YOLOv8 cls model
        results = model.predict(matching_files_sorted)
        object_orientation_list = [get_class_name(item) for item in results]

        return object_orientation_list
    else:
        return None

def orientation_classify(images_paths: Path, model_path: Path):
    model = YOLO(model_path) 
    results = model.predict(images_paths)
    return results
    
