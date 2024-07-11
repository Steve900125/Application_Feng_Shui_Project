import detectors.yolo_detector as yolo_dect
import symmetry.symmetry_detector as sym_dect
import symmetry.obstacle_decetor as obs_dect
from pathlib import Path
import shutil
import time
import sys


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLO root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
# if platform.system() != 'Windows':
#     ROOT = ROOT.relative_to(Path.cwd())


def feng_shui_door(source, floor_plan_model, orientation_model,result):
    
    '''
        Feng_Shui Door Symmetry Detection Block
        About : Determine whether two doors are symmetrical and whether there are no geographical obstacles in between.
    '''
    item = 'door'
    judge_rule = 0.5 # Maching over 50% area in symmetry rate

    # Find 
    item_orient_list = yolo_dect.orientation_classify( ROOT=ROOT, item=item, source_path=result.path, model=orientation_model)
    
    # Can't find any data or classification failed
    if item_orient_list is None:
        return 0

    # Get item's xyxy information from "result"
    item_xyxy_list = [
        sub_item.boxes.xyxy.tolist() 
        for sub_item in result
        if result.names[sub_item.boxes.cls.item()] == item
    ]
    # Flatten boxes [ [[x1,y1,x2,y2]] , ... ]  -> [[x1,y1,x2,y2] , ... ]
    item_xyxy_list = [item_xyxy[0] for item_xyxy in item_xyxy_list]

    
    # [item_A , item_B , {'symmetry':bool, 'cross_area_rate':float, 'full_contain':bool}]
    door_sym_list = sym_dect.door_symmetry_detect(  item_xyxy_list = item_xyxy_list , 
                                                    item_orient_list = item_orient_list,
                                                    sym_true_filter = True
                                                    )   

    # Filter the 'cross_area_rate' over the certain range
    if door_sym_list is None:
        print("Hi 我是薛丁格的 bug")
        # 薛丁格的 bug
        return 0

    door_sym_list = [ result for result in door_sym_list if result[2]['cross_area_rate'] > judge_rule]

    problem_count = 0

    # Check obstacle 
    for sym_data in door_sym_list:
        obs_rate = obs_dect.item_obstacle_decete(
            source = result.path,
            item_A = sym_data[0],
            item_B = sym_data[1],
            orientation = sym_data[0].orientation
        )
        print(obs_rate)
        if obs_rate < judge_rule:
            problem_count+=1

            # # Draw
            # img = cv2.imread(result.path)
            # image_with_box_1 = sym_data[0].draw_box_on_image(image=img)
            # image_with_box_2 = sym_data[1].draw_box_on_image(image=image_with_box_1)  
            # cv2.imwrite('output.jpg', img)

    return problem_count

def run(
        source = ROOT / 'images',
        floor_plan_model = ROOT / 'models' / 'yolov8_floor_plan_dect.pt',  # detection (Yolov8)
        orientation_model = ROOT / 'models' / 'yolov8_orientation_cls.pt',  # classification (Yolov8)
    ):
    
    # Correspondent type result
    final_result= {}

    results = yolo_dect.floor_plan_detcte(source = source,  model = floor_plan_model)
    # stop running
    if results is None : 
        return 0

    door_total_problems = 0
    for result in results:
        problem_count = feng_shui_door( source = source , 
                                        floor_plan_model = floor_plan_model ,
                                        orientation_model = orientation_model ,
                                        result = result )
        print("門對門的形況出現 : ", problem_count)
        door_total_problems += problem_count

    # Remove all predict file (resource)
    predict_path = ROOT / 'runs' / 'detect' / 'predict'
    shutil.rmtree(predict_path)

    final_result['door'] = door_total_problems

    return final_result

if __name__ == "__main__":
    start = time.time()
    run()
    end = time.time()
    print("執行時間：%f 秒" % (end - start))
    
