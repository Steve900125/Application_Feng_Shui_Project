from ultralytics.engine.results import Results # type: ignore
from typing import List, Optional, Dict
from pathlib import Path
import cv2
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

# Fengshui 
from fengshui.item import Item  # Core class vary important
OUTPUT_PATH = ROOT / 'fengshui' / 'output' # Note: In "draw" dir have same default path

# Draw
from draw.draw_item import draw_bounding_boxes
from draw.draw_item import save_to_image

# Overlape
from overlape.overlape import overlape_rate
OVERLAPE_THRESHOLD = 0.5 # 50% overlape range

def show_results(results: List[Results]):
    print(len(results[0].boxes.cls))
    print(results[0].boxes.cls)
    #print(results[0].boxes)
    print(results[0])
    print(type(results[0].boxes)) 

def clean_folder(folder_path: Path):
    '''
        'clean_folder' is using in server to clean temporary files
    '''
    if folder_path.exists():
        shutil.rmtree(folder_path)
        print(f"{folder_path} has been deleted.")
        os.makedirs(folder_path)

def extract_target_xyxy_data(object_name: str, result: Results) -> Optional[List[List[float]]]:
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

def init_one_target(object_name: str, result: Results) -> Optional[List[Item]]:
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
    
    print('orientation_list',len(orientation_list))
    print('xyxy_list',len(xyxy_list))

    print(orientation_list)
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

def save_overlap_to_jpg(overlape_results:Dict[str, dict], result: Results):

    # Selet target
    image = cv2.imread(str(result.path))
    for res in overlape_results:
        image = draw_bounding_boxes(item=res['items'][0], image=image)
        image = draw_bounding_boxes(item=res['items'][1], image=image)
    
    # Init file name (save at ROOT / fengshui / *.jpg)
    result_path = Path(result.path)
    sub_name = res['items'][0].name + res['items'][1].name
    file_name = 'overlape_' + sub_name + str(result_path.name)

    save_to_image(image= image, file_name= file_name)

def filter_overlape_rate(overlape_results: List[Dict[str, dict]]) -> List[Dict[str, dict]]:
    """
    Checks the overlap rate and full coverage status for a list of results, 
    and returns a list of eligible results.

    Parameters:
    - overlape_results (List[Dict[str, dict]]): A list of dictionaries containing overlap analysis results.

    Returns:
    - List[Dict[str, dict]]: A list of eligible results where full coverage is achieved 
                             or the overlap rate is above the defined threshold.
    """
    eligibility_list = []
    for res in overlape_results:
        if res['full_coverage'] or res['rate'] >= OVERLAPE_THRESHOLD:
            eligibility_list.append(res)

    return eligibility_list

def get_overlap_results_one_item(item_list: List[Item]) -> List[Dict[str, any]]:
    overlape_results = []
    # Step 3: Compare each pair of items for overlap
    for out_index in range(len(item_list)):
        for inner_index in range(out_index+1, len(item_list)):
            items = [item_list[out_index], item_list[inner_index]]
            # {"items": List[Item, Item],"rate": float,"full_coverage": bool}
            overlape_results.append(overlape_rate(items=items))
    return overlape_results

def get_overlap_results_two_item(type_one_item_list: List[Item], type_two_item_list: List[Item])-> List[Dict[str, any]]:
    overlape_results = []
    # Step 3: Compare each pair of items for overlap
    for type_one_item in type_one_item_list:
        for type_two_item in type_two_item_list:
            items = [type_one_item, type_two_item]
            # {"items": List[Item, Item],"rate": float,"full_coverage": bool}
            overlape_results.append(overlape_rate(items=items))
    return overlape_results

def change_orientation(item_list: List[Item], orientation: str)-> List[Item]:
    for item in item_list:
        item.orientation = orientation
    return item_list

# Core function
def object_to_object(objects_name: List[str], result: Results, orient_check: Dict[str, bool]):
    ''' 
    FengShui object to object analysis.
    
    Parameters:
    - objects_name (List[str]): List of object names to analyze.
    - result (Results): The results object containing detected objects and their bounding boxes.
    
    Returns:
    - Optional[dict]: Analysis result dictionary or None if no data is available.
    
    Steps:
    <Check if result is empty (no data)>
    1. Check if the number of target objects is valid (1 or 2).
    2. Format the data for easier use.
    3. Caculate the overlap rate
    4. Filter the objects by the threshold and save target to jpg.
    5. Check if the path is clear if the objects overlap.
    '''

    overlape_results = []

    # Is result empty (no data)
    target_cls_list = result.boxes.cls.tolist()
    objects_name_id = []

    # Find it's id from result.names
    for name in objects_name:
        for cls_id in result.names:
                if result.names[cls_id] == name:
                    objects_name_id.append(cls_id)

    # Check exist in target_cls_list
    for name_id in objects_name_id :
        if target_cls_list.count(name_id) <= 0:
            return None

    # Step1 : Check the number of target objects
    if len(objects_name) == 1:
        
        # Stept2 : Format the data for one target
        item_list = init_one_target(object_name=objects_name[0], result=result)

        if item_list is None:
            print("The item_list is None")
            return None
        
        #if orient_check[objects_name] == False:
        # Step 3: Compare each pair of items for overlap
        if orient_check[objects_name[0]]:
            overlape_results = get_overlap_results_one_item(item_list=item_list)
        else:
            # No orientation check means that process need to check both orientation.
            item_list = change_orientation(item_list= item_list, orientation= 'horizontal')
            hor_overlape_results = get_overlap_results_one_item(item_list=item_list)

            item_list = change_orientation(item_list= item_list, orientation= 'vertical')
            ver_overlape_results = get_overlap_results_one_item(item_list=item_list)

            overlape_results = hor_overlape_results + ver_overlape_results
            


    elif len(objects_name) == 2:
        # Stept2 : Format the data for one target
        type_one_item_list = init_one_target(object_name=objects_name[0], result=result)
        type_two_item_list = init_one_target(object_name=objects_name[1], result=result)

        print(type_one_item_list)
        print(type_two_item_list)

        # Both need to check orientation
        if orient_check[objects_name[0]] and orient_check[objects_name[1]]:
            print("Both need to check orientation")
            overlape_results = get_overlap_results_two_item(type_one_item_list=type_one_item_list, 
                                                            type_two_item_list=type_two_item_list)
        # Both don't need to check orientation
        elif not orient_check[objects_name[0]] and not orient_check[objects_name[1]]:
            print("don't need to check orientation")
            type_one_item_list = change_orientation(item_list= type_one_item_list, orientation= 'horizontal')
            type_two_item_list = change_orientation(item_list= type_two_item_list, orientation= 'horizontal')
            hor_overlape_results = get_overlap_results_two_item(type_one_item_list=type_one_item_list, 
                                                            type_two_item_list=type_two_item_list)
            
            type_one_item_list = change_orientation(item_list= type_one_item_list, orientation= 'vertical')
            type_two_item_list = change_orientation(item_list= type_two_item_list, orientation= 'vertical')
            ver_overlape_results = get_overlap_results_two_item(type_one_item_list=type_one_item_list, 
                                                                type_two_item_list=type_two_item_list)  

            overlape_results = hor_overlape_results + ver_overlape_results

        # One of them need check orientation
        else:
            print(' One of them need check orientation')
            if orient_check[objects_name[0]]:
                main_list = type_one_item_list
                dependence＿list = type_two_item_list
            if orient_check[objects_name[1]]:
                main_list = type_two_item_list
                dependence＿list = type_one_item_list

            
            dependence＿list = change_orientation(item_list= dependence＿list, orientation= 'horizontal')
            hor_overlape_results = get_overlap_results_two_item(type_one_item_list= main_list, 
                                                                type_two_item_list= dependence＿list)
            
            dependence＿list = change_orientation(item_list= dependence＿list, orientation= 'vertical')
            ver_overlape_results = get_overlap_results_two_item(type_one_item_list= main_list, 
                                                                type_two_item_list= dependence＿list)
            overlape_results = hor_overlape_results + ver_overlape_results 

            #save_overlap_to_jpg(overlape_results, result=result) 
            print("overlape_results ",overlape_results)
    else:
        return None
    
    # Step4 : Filter the objects by the threshold and save target to jpg.

    # Filter by OVERLAPE_THRESHOLD
    have_overlape_list = filter_overlape_rate(overlape_results)
    
    # Note: For extract oringinal path need to input "result"  
    if len(have_overlape_list) > 0:
        save_overlap_to_jpg(have_overlape_list, result=result)  

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
    clean_folder(OUTPUT_PATH)
    
    # Object detection
    results = floor_plan_detect(images_path=IMAGES_PATH, model_path=DETECT_MODEL_PATH)
    
    # Object to object analysis
    door_to_door = ['door']
    window_to_window = ['window']
    entrance_to_kitchen = ['entrance', 'kitchen']
    orient_check = {'entrance': True, 'kitchen': False, 'door': True}
    for result in results:
        object_to_object(objects_name=door_to_door, result=result, orient_check=orient_check)
        #object_to_object(objects_name=window_to_window, result=result)
        object_to_object(objects_name=entrance_to_kitchen, result=result, orient_check=orient_check)

if __name__ == "__main__":
    run()