from ultralytics.engine.results import Results
from typing import List, Optional
from ultralytics import YOLO
from pathlib import Path

def get_class_name(result:Results):
    return result.names[result.probs.top1]

def object_orientation_classify(root:Path, model_path:Path, object_name:str, result:Results)->Optional[List]:

    # To find item image crops path from YOLOv8
    item_crops_path = root / Path(result.save_dir) / 'crops' / object_name
    # Check if the object has been detected
    not_empty = True if len(result.boxes.cls)>0 else False

    if item_crops_path.exists and not_empty:
        model = YOLO(model_path)  # pretrained YOLOv8 cls model
        results = model.predict(item_crops_path)
        object_orientation_list = [get_class_name(item) for item in results]

        return object_orientation_list
    else:
        return None

    