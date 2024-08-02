from ultralytics.engine.results import Results
from typing import List, Optional
from pathlib import Path
import shutil
import sys  
import os 

# Path arrangement
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  
sys.path.insert(0, str(ROOT))   # for import moduls 

# Resource
IMAGES_PATH = ROOT / "images"

# Clean folder
CLEAN_IMAGES_FOLDER = False

# Vision
from vision.detect import floor_plan_detect  
from vision.classify import object_orientation_classify
YOLO_RESULTS_PATH = ROOT / 'runs' 
DETECT_MODEL_PATH = ROOT / 'models' / 'detect_yolov8.pt'
CLASSIFY_MODEL_PATH = ROOT / 'models' / 'classify_yolov8.pt'

# Fengshui class
from fengshui.item import Item

# Draw
from draw.draw_item import draw_bounding_boxes
from draw.draw_item import save_to_image

def show_results(results:list):
    print(len(results[0].boxes.cls))
    print(results[0].boxes.cls)
    #print(results[0].boxes)
    print(results[0])
    print(type(results[0].boxes)) 

def clean_folder(folder_path:Path):
    '''
        'clean_folder' is using in server to clean temporary files
    '''
    if folder_path.exists():
        shutil.rmtree(folder_path)
        print(f"{folder_path} has been deleted.")
        os.makedirs(folder_path)

def extract_target_xyxy_data(object_name:str, result:Results)->Optional[List[List[float]]]:
    """
    Extracts the bounding box coordinates for the specified object name from the given Results object.

    Parameters:
    - object_name (str): The name of the object to filter for.
    - result (Results): The results object containing detected objects and their bounding boxes.

    Returns:
    - Optional[List[List[float]]]: A list of bounding boxes corresponding to the specified object name, otherwise None.
    Example : [[702.2841796875, 584.446044921875, 785.9964599609375, 680.3638916015625], [], ...]
    """

    # Is result empty (no data)
    if len(result.boxes.cls)<=0:
        return None

    # result.names
    # {0: 'bathroom', 1: 'bed', 2: 'bedroom', 3: 'door', 4: 'entrance', 5: 'kitchen', 6: 'living-room', 7: 'window'}
    data_list = zip(result.boxes.xyxy, result.boxes.cls)

    # Note : xyxy and cls type is Tensor
    target_data_list = []
    for xyxy, cls in data_list:
        if result.names[cls.item()] == object_name:
            target_data_list.append(xyxy.tolist())     

    return target_data_list

def init_one_target(object_name:str, result:Results)->Optional[List[Item]]:
    """
    Initializes a list of Item objects for the specified object name from the given Results object.

    Parameters:
    - object_name (str): The name of the object to filter and initialize items for.
    - result (Results): The results object containing detected objects and their bounding boxes.

    Returns:
    - Optional[List[Item]]: A list of Item objects if the lengths of orientation_list and xyxy_list match, otherwise None.
    Example : Item (702.2841796875, 584.446044921875, 785.9964599609375, 680.3638916015625,'door', 'vertical')
    """
        
    # Get what data we want and create "Item" list
    orientation_list = object_orientation_classify(root=ROOT, model_path=CLASSIFY_MODEL_PATH, object_name=object_name, result=result)
    xyxy_list =  extract_target_xyxy_data(object_name=object_name, result=result)
    item_list = []

    if len(orientation_list) == len(xyxy_list):  # (error detect)
        for index in range(len(orientation_list)):
            item = Item(x1=xyxy_list[index][0],
                        y1=xyxy_list[index][1],
                        x2=xyxy_list[index][2],
                        y2=xyxy_list[index][3],
                        orientation=orientation_list[index],
                        name=object_name)
            item_list.append(item)      
        return item_list        
    else:
        return None # The data don't correspond in length

def object_to_object(objects_name:List[str], result:Results):
    ''' 
        FengShui oject to object analysis 
        (IMPORTANT!! If "result" doesn't have any data we return None)
        Step1 : Check 2 target is same name or not
        Step2 : Formate the data which we can easily use ("Future formatable input data location.")
        Step3 : Check they are overlap or not
        Step4 : Check if the path is clear if these two objects overlap
    '''
    # Is result empty (no data)
    if len(result.boxes.cls)<=0:
        return None

    # Step1 
    if len(objects_name) == 1:
        # Stept2
        item_list = init_one_target(object_name=objects_name[0], result=result)
        print(item_list[0])
        image = draw_bounding_boxes(image_path=result.path,item=item_list[0])
        save_to_image(image=image, file_name='test.jpg')


    elif len(objects_name) == 2:
        pass
    else:
        return None


def run():
    """
        Main function for Feng Shui conflict detection.

        The function performs the following tasks:
        1. Object Detection: Identifies objects in the floor plan and determines their locations.
        2. Conflict Detection: Projects and compares the overlap between two objects to detect conflicts.
        3. Obstacle Detection: Checks for obstacles between the paths of two objects.

        Returns:
            results: The results of the detection processes, including object locations, conflicts, and obstacles.
    """
    # Delete previous user data
    clean_folder(YOLO_RESULTS_PATH)
    
    # Object detection
    results = floor_plan_detect(images_path=IMAGES_PATH, model_path=DETECT_MODEL_PATH)
    show_results(results=results)
    
    # Object to object analysis
    door_to_door = ['door']
    entrance_to_kitchen = ['entrance', 'kitchen']
    for result in results:
        object_to_object(objects_name=door_to_door, result=result)
        #object_to_object(objects_name=entrance_to_kitchen, result=result)
    

    
    

if __name__ == "__main__":
    run()