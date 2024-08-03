from ultralytics.engine.results import Results # type: ignore
from ultralytics import YOLO # type: ignore
from pathlib import Path
from typing import List, Optional


def floor_plan_detect(images_path: Path, model_path: Path) -> Optional[List[Results]]:
    """
        This function uses YOLOv8 to detect objects in the floor plan.

        Args:
            images_path (str or Path): The path to the directory containing the images.
            model_path (str or Path): The path to the YOLOv8 model file.

        Returns:
            results: The results of the YOLOv8 model's prediction, otherwise None.
    """

    if model_path.exists() and images_path.exists():
        model = YOLO(model_path)  
        results = model.predict(images_path, save=True, save_txt=True, save_crop=True, exist_ok=True)  
        return results
    else:
        return None
        #raise FileNotFoundError("Model path or images path does not exist.")
        
