from ultralytics.engine.results import Results # type: ignore
from typing import List, Optional, Dict
import copy
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
from draw.draw_item import draw_points_line

# Obstical
from obstacle.obstacle import items_obstacle_detect
OBSTACLE_THRESHOLD = 0.5 # 50%

# Overlap
from overlap.overlap import overlap_rate
OVERLAP_THRESHOLD = 0.5 # 50% overlap range

# 這裡應該要是 list 對應 oto result 主程式要再修正
def show_analy_results(result: Dict[str, List]):

    if result is None:
        return None

    for res in result['overlap_result']:
        # Print the names of the two overlapping items
        print("Item 1 name: {}, Item 2 name: {}".format(res['items'][0].name, res['items'][1].name))

    # items, points_line, bin_image_np_arrary, rate
    for res in result['obstacle_result']:
        print("Item 1 name: {}, Item 2 name: {}".format(res['items'][0].name, res['items'][1].name))

       

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

def save_overlap_to_jpg(overlap_results: Dict[str, any], result: Results):

    # Selet target
    image = cv2.imread(str(result.path))
    for res in overlap_results:
        image = draw_bounding_boxes(item=res['items'][0], image=image)
        image = draw_bounding_boxes(item=res['items'][1], image=image)
    
    # Init file name (save at ROOT / fengshui / *.jpg)
    result_path = Path(result.path)
    sub_name = res['items'][0].name + '_to_' +res['items'][1].name + '_'
    file_name = 'overlap_' + sub_name + str(result_path.name)

    save_to_image(image= image, file_name= file_name)

def save_obstacle_to_jpg(obstacle_results: Dict[str, any], result: Results):

    # Selet target
    image = cv2.imread(str(result.path))
    for res in obstacle_results:
        image = draw_bounding_boxes(item=res['items'][0], image=image)
        image = draw_bounding_boxes(item=res['items'][1], image=image)
        image = draw_points_line(points_line=res['points_line'], image=image)

    # Init file name (save at ROOT / fengshui / *.jpg)
    result_path = Path(result.path)
    sub_name = res['items'][0].name + '_to_' +res['items'][1].name + '_'
    file_name = 'obstacle_' + sub_name + str(result_path.name)

    save_dir = save_to_image(image= image, file_name= file_name)

    return save_dir 

def filter_overlap_rate(overlap_results: List[Dict[str, any]]) -> List[Dict[str, any]]:
    """
    Checks the overlap rate and full coverage status for a list of results, 
    and returns a list of eligible results.

    Parameters:
    - overlap_results (List[Dict[str, dict]]): A list of dictionaries containing overlap analysis results.

    Returns:
    - List[Dict[str, dict]]: A list of eligible results where full coverage is achieved 
                             or the overlap rate is above the defined threshold.
    """
    eligibility_list = []
    for res in overlap_results:
        if res['full_coverage'] or res['rate'] >= OVERLAP_THRESHOLD:
            eligibility_list.append(res)

    return eligibility_list

def filter_obstacle_rate(obstacle_results: List[Dict[str, any]]):
    eligibility_list = []
    for res in obstacle_results:
        if res['rate'] <= OBSTACLE_THRESHOLD:
            eligibility_list.append(res)

    return eligibility_list

def get_overlap_results_one_item(item_list: List[Item]) -> List[Dict[str, any]]:
    overlap_results = []
    # Step 3: Compare each pair of items for overlap
    for out_index in range(len(item_list)):
        for inner_index in range(out_index+1, len(item_list)):
            items = [item_list[out_index], item_list[inner_index]]
            # {"items": List[Item, Item],"rate": float,"full_coverage": bool}
            overlap_results.append(overlap_rate(items=items))
    return overlap_results

def get_overlap_results_two_item(type_one_item_list: List[Item], type_two_item_list: List[Item])-> List[Dict[str, any]]:
    overlap_results = []
    # Step 3: Compare each pair of items for overlap
    for type_one_item in type_one_item_list:
        for type_two_item in type_two_item_list:
            items = [type_one_item, type_two_item]
            # {"items": List[Item, Item],"rate": float,"full_coverage": bool}
            overlap_results.append(overlap_rate(items=items))
    return overlap_results

def change_orientation(item_list: List[Item], orientation: str)-> List[Item]:
    # 若不是使用複製則會導致 item 方向會被第二次覆蓋掉物件原始數值導致方向錯誤
    # 針對以檢測出來的結果方向會同步更動受到影響
    new_item_list = copy.deepcopy(item_list)
    for item in new_item_list:
        item.orientation = orientation
    return new_item_list

# Core function
def total_object_to_object(objects_name: List[str], result: Results, orient_check: Dict[str, bool]):
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
    6. Caculate the obstacle rate
    '''

    overlap_results = []

    # Is result empty (no data)
    target_cls_list = result.boxes.cls.tolist()
    objects_name_id = []

    # Find item id from result.names
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
            overlap_results = get_overlap_results_one_item(item_list=item_list)
        else:
            # No orientation check means that process need to check both orientation.
            item_list = change_orientation(item_list= item_list, orientation= 'horizontal')
            hor_overlap_results = get_overlap_results_one_item(item_list=item_list)

            item_list = change_orientation(item_list= item_list, orientation= 'vertical')
            ver_overlap_results = get_overlap_results_one_item(item_list=item_list)

            overlap_results = hor_overlap_results + ver_overlap_results
            
    elif len(objects_name) == 2:
        # Stept2 : Format the data for one target
        type_one_item_list = init_one_target(object_name=objects_name[0], result=result)
        type_two_item_list = init_one_target(object_name=objects_name[1], result=result)

        #print(type_one_item_list)
        #print(type_two_item_list)
        if type_one_item_list is None or type_two_item_list is None:
            return None

        # Both need to check orientation
        if orient_check[objects_name[0]] and orient_check[objects_name[1]]:
            print("Both need to check orientation")
            overlap_results = get_overlap_results_two_item(type_one_item_list=type_one_item_list, 
                                                            type_two_item_list=type_two_item_list)
        # Both don't need to check orientation
        elif not orient_check[objects_name[0]] and not orient_check[objects_name[1]]:
            print("don't need to check orientation")
            type_one_item_list = change_orientation(item_list= type_one_item_list, orientation= 'horizontal')
            type_two_item_list = change_orientation(item_list= type_two_item_list, orientation= 'horizontal')
            hor_overlap_results = get_overlap_results_two_item(type_one_item_list=type_one_item_list, 
                                                            type_two_item_list=type_two_item_list)
            print("hor_overlap_results :",hor_overlap_results)
            type_one_item_list = change_orientation(item_list= type_one_item_list, orientation= 'vertical')
            type_two_item_list = change_orientation(item_list= type_two_item_list, orientation= 'vertical')
            ver_overlap_results = get_overlap_results_two_item(type_one_item_list=type_one_item_list, 
                                                                type_two_item_list=type_two_item_list)  
            print('ver_overlap_results :',ver_overlap_results )

            overlap_results = hor_overlap_results + ver_overlap_results
            #print("k_to_e",overlap_results)

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
            hor_overlap_results = get_overlap_results_two_item(type_one_item_list= main_list, 
                                                                type_two_item_list= dependence＿list)
            
            dependence＿list = change_orientation(item_list= dependence＿list, orientation= 'vertical')
            ver_overlap_results = get_overlap_results_two_item(type_one_item_list= main_list, 
                                                                type_two_item_list= dependence＿list)
            overlap_results = hor_overlap_results + ver_overlap_results 
    else:
        return None
    
    # Step4 : Filter the objects by the threshold and save target to jpg.
    # Filter by OVERLAP_THRESHOLD
    # result_dic = {'items': items,'rate': 0.0,'full_coverage': False}
    have_overlap_list = filter_overlap_rate(overlap_results)

    # Note: For extract oringinal path need to input "result"  
    if len(have_overlap_list) > 0:
        save_overlap_to_jpg(have_overlap_list, result=result)  
    
    obstacle_results = []
    # Step5 : check obstical rate
    # target
    for target in have_overlap_list:
        obstacle_result = items_obstacle_detect(items= target['items'], image_path= result.path)
        obstacle_results.append(obstacle_result)
    
    pass_obstacle_results = filter_obstacle_rate(obstacle_results=obstacle_results)

    image_result_dir = None
    if len(pass_obstacle_results) > 0:
        image_result_dir = save_obstacle_to_jpg(obstacle_results= pass_obstacle_results, result=result)
    else:
        return None

    all_results = {
        'overlap_result'  : have_overlap_list,
        'obstacle_result' : pass_obstacle_results,
        'image_path' : image_result_dir
    }

    return all_results
        
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
    entrance_to_kitchen = ['entrance', 'kitchen']
    orient_check = {'entrance': False, 'kitchen': False, 'door': True, 'window':True}

    chatbot_images = {
        'door_to_door':[],
        'entrance_to_kitchen':[],
    }

    for result in results:
        door_result = total_object_to_object(objects_name=door_to_door, result=result, orient_check=orient_check)
        if door_result is not None:
            chatbot_images['door_to_door'].append(door_result['image_path'])
        
        entr_result = total_object_to_object(objects_name=entrance_to_kitchen, result=result, orient_check=orient_check)
        if entr_result is not None:
            chatbot_images['entrance_to_kitchen'].append( entr_result['image_path'])
    
    return  chatbot_images


if __name__ == "__main__":
    res = run()
    print(res)
