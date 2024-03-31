import detect_functions as d_f 
import os

# Initial basic information
source_file = './source_floor_plan/*.jpg' # 來源的所有資料
item_name = 'door' # 要針對辨識方向的物件名稱
org_img_name = '' # 原始圖片的名稱
cls_model_dir = 'yolov8_model_door_cls.pt' # 辨識垂直會水平的模型名稱


# 將圖片進行 Yolo 平面圖影像辨識物件
results = d_f.object_detction(source_file)

# 針對每張圖片個別進行處理
for result in results:
    # 取得此原始辨識圖片的名稱
    org_img_name = os.path.splitext(os.path.basename(result.path))[0]

    # 辨別門(item_name 該物件)的方向是水平還是垂直
    cls_list = d_f.orientation_classify(item_name , org_img_name , cls_model_dir)

    # 找不到門的物件
    if len(cls_list) == 0:
        continue
    
    # 取得門(item_name 該物件)的xyxy數值
    # Convert tensor to integer for indexing, assuming item.boxes.cls is a 1-element tensor
    target_list = [
        item.boxes.xyxy.tolist()[0][:] for item in result 
        if result.names[item.boxes.cls.item()] == item_name 
    ]

    # 回傳所有針對物件對稱辨識的結果包含:
    # 1. 比較的兩個物件資料 (Class  FengShuiItem : define by detect_functions.py)
    # 2. 比較結果 : 'symmetry' : 是否對稱 ,  'cross_area_rate' : 重疊比例 (0~1) , 'full_contain' : 是否被完全包含
    symmetry_results = d_f.identify_door_symmetry( target_list , cls_list )
    
    # 檢查中間有沒有障礙物比例
    symmetry_true = [is_sys for is_sys in symmetry_results if is_sys[2]['symmetry'] == True] 
    
    # 針對已經發現對稱的物件檢查中間的障礙物
    for sym_item in symmetry_true:
        obstacle_rate = d_f.check_obstacle_rate( result.path , sym_item[0] , sym_item[1] , sym_item[0].orientation)
        sym_item[2]["obstacle_rate"] = obstacle_rate # In the dic we add obstacle_rate values
        print(sym_item[2])





