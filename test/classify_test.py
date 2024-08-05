from ultralytics import YOLO # type: ignore
from pathlib import Path
import sys  

# Path arrangement
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  
sys.path.insert(0, str(ROOT))   # for import moduls 

CLASSIFY_MODEL_PATH = ROOT / 'models' / 'classify_yolov8.pt'
def get_class_name(result):
    return result.names[result.probs.top1]

matching_files_sorted = ROOT /  'runs' / 'detect' /'predict' / 'crops' /'entrance' / 'entrance.jpg'
model = YOLO(CLASSIFY_MODEL_PATH)  # pretrained YOLOv8 cls model
results = model.predict(matching_files_sorted)

object_orientation_list = [get_class_name(item) for item in results]

print(results[0])
print(results[0].probs)