from ultralytics import YOLO
import os
import re
import glob
import cv2
import numpy as np


class FengShuiItem:
    def __init__(self, x1, y1, x2, y2, orientation):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.orientation = orientation  # ex: door : 'vertical' or 'horizontal'
    
    def get_center(self):
        """Calculate the center of the door."""
        centerX = (self.x1 + self.x2) / 2
        centerY = (self.y1 + self.y2) / 2
        return (centerX, centerY)

    def get_area(self):
        """Calculate the area of the door."""
        width = self.x2 - self.x1
        height = self.y2 - self.y1
        return width * height
    
    def __repr__(self):
        return f"Item ({self.x1}, {self.y1}, {self.x2}, {self.y2}, '{self.orientation}')"

# Call by main directly
# Object : 針對圖片進行物件偵測
# Input : 將圖片所屬的資料夾路徑放置於此函數
# Output: 產生平面圖辨識結果到 ./runs/detect/predict(N) 位置 
# 包含標註好的圖片，個物件的位置表，以及物件本身圖片
def object_detction(images_folder):
 
    # Load a model
    model = YOLO('yolov8_model_v3.pt')  # pretrained YOLOv8n model

    results = model.predict(images_folder , save=True, save_txt=True , save_crop = True)  # return a list of Results objects

    return results 


# Object : 找到當前預測資料夾
# Input : 根目錄的檔案路徑 , 隨著次數增加的子目錄路徑
# Output : 當前最新的目錄
def find_latest_folder(base_path, base_name="predict"):
    # 列出所有的檔案和檔案夾
    items = os.listdir(base_path)
    
    # 創建一個正則表達式，匹配 'predict' 和後面可能跟隨的數字
    pattern = re.compile(f"^{base_name}(\\d*)$")
    
    # 儲存匹配的檔案夾和對應的數字
    matches = []
    for item in items:
        match = pattern.match(item)
        if match and os.path.isdir(os.path.join(base_path, item)):
            num = match.group(1)
            num = int(num) if num.isdigit() else 0
            matches.append((item, num))
    
    # 根據數字排序並找到最大的一個
    if matches:
        latest_folder = sorted(matches, key=lambda x: x[1], reverse=True)[0][0]
        return os.path.join(base_path, latest_folder)
    else:
        return None

# Call by main directly
# Object : 利用分類器去分類指定圖片源內的物件
# Input : 物件的名稱 , 原始資料的名稱(被crops裁切前的原始檔案) , 分類器的 model 路徑
# Output : 辨識完類別的 list 如果找不到會回傳 None
def orientation_classify(item_name , org_img_name , cls_model_dir):
    # ================================================================================================#
    # Yolo v8 預設的偵測存放路徑
    predict_file_root = './runs/detect/'
    # 預測結果的檔案命名基礎 predict + n ['' ~ int N]
    predict_file_name = 'predict' 
    # 找出預測的最新資料夾
    latest_predict = find_latest_folder(base_path= predict_file_root , base_name= predict_file_name )
    
    # Check point
    if latest_predict == None:
        print("Warning : can't not find the [predict] floder " , latest_predict)
        print("It should look like : ./runs/detect/predict")
        return None 
    # ================================================================================================#
    # 合併到 crops 資料夾路徑加上 target (物件名稱)
    target_path = os.path.join(latest_predict , 'crops' , item_name)
    
    # Check point
    if os.path.isdir(target_path):
        #print(target_path)
        pass
    else:
        print("Warning : can't not find the floder " , target_path)
        return None
    # ================================================================================================#
    # 尋找對應原始資料所裁切下來的所有目標子檔案
    # EX : ./runs/detect/predict4/crops/door org_img_name = jojo
    # ./runs/detect/predict4/crops/door/jojo.jpg
    # ./runs/detect/predict4/crops/door/jojo1.jpg
    search_pattern = os.path.join(target_path, f'{org_img_name}*.jpg')
    matching_files = glob.glob(search_pattern)
    
    # Check point
    if len(matching_files) == 0:
        print("Warning : can't not find any crops image in this floder")
        return None
    # ================================================================================================#
    # 根據 cls_model_dir 模型辨識出想要的結果回傳 list 
    model = YOLO(cls_model_dir)  # pretrained YOLOv8n model
    results = model.predict(matching_files)
    cls_list = []
    for item in results:
        cls_list.append(item.names[item.probs.top1])
    return cls_list

# Object : 辨識面對面的百分比用聯集取交集
# Input : class FengShuiItem A and B + orientation
# Output : symmetry (bool :True/False) , bounding_area (float) , full_contain (bool :True/False)
def check_door(item_A , item_B , orientation) -> dict:
    result_dic = {}
    if orientation == 'vertical':
        cent_x_A,_ = item_A.get_center()
        cent_x_B,_ = item_B.get_center()

        # 如果 door_A 門中心大於 door_B 門中心就把 A 門設在右方 B 門設在左方
        if cent_x_A >= cent_x_B:
            right_door = item_A
            left_door = item_B
        else:
            right_door = item_B
            left_door = item_A
        
        # 當左方門的最大值仍然小於右方門的最小值代表沒有交集
        if left_door.x2 <= right_door.x1:
            result_dic['symmetry'] = False
            result_dic['bounding_area'] = 0
            result_dic['full_contain'] = False
            return result_dic
            # End
        else :
            data = [left_door.x1 , left_door.x2 , right_door.x1 , right_door.x2]  
            data.sort()
            # Get max range
            # 最大減最小確定聯集範圍
            union = data[3] - data[0]
            intersection = data[2] - data[1]
            bounding_area = intersection / union

            # full contain check
            if  left_door.x1 <= right_door.x1  and right_door.x2 <= left_door.x2:
                result_dic['full_contain'] = True
            else :
                result_dic['full_contain'] = False
            
            result_dic['symmetry'] = True
            result_dic['bounding_area'] = bounding_area
            
            return result_dic
    
    if orientation == 'horizontal':
        _,cent_y_A = item_A.get_center()
        _,cent_y_B = item_B.get_center()

         # 如果 door_A 門y中心大於 door_B 門中心就把 A 門設在下方 B 門設在上方 （翻轉概念）
        if cent_y_A >= cent_y_B:
            up_door = item_B
            down_door = item_A
        else:
            up_door = item_A
            down_door = item_B
        
        if up_door.y2 <= down_door.y1:
            result_dic['symmetry'] = False
            result_dic['bounding_area'] = 0
            result_dic['full_contain'] = False
            return result_dic
            # End
        else :
            data = [up_door.y1 , up_door.y2 , down_door.y1 , down_door.y2]  
            data.sort()
            # Get max range
            # 最大減最小確定聯集範圍
            union = data[3] - data[0]
            intersection = data[2] - data[1]
            bounding_area =  intersection / union

            # full contain check
            if up_door.y1 <= down_door.y1 and down_door.y2 <=  up_door.y2:
                result_dic['full_contain'] = True
            else :
                result_dic['full_contain'] = False
            
            result_dic['symmetry'] = True
            result_dic['bounding_area'] = bounding_area
            return result_dic
            # End
    
# Call by main directly
# Input : Item xyxy data list , Item classification result list
# Output : symmetry_results : Class FengShuiItem 2 items  dic: {'symmetry' : 是否對稱 ,  'bounding_area' : 重疊比例 (0~1) , 'full_contain' : 是否被完全包含}
# Logic : 分成垂直跟水平的 list -> 針對不同方向作排序(越接近越容易面對面) -> C n 取 2 比較所有物件
def identify_door_symmetry(door_xyxy_list , door_cls_list):
    vertical_list = []
    horizontal_list = []
    
    if len(door_cls_list) != len(door_xyxy_list):
        print("Warning : The length doesn't campare with each other (door_xyxy_list and door_cls_list length number)")
        return None

    # 初始化門的物件類型分成水平跟垂直
    for num in range(len(door_cls_list)):
        if door_cls_list[num] == 'vertical':
            # xyxy set
            item = FengShuiItem( door_xyxy_list[num][0], 
                                 door_xyxy_list[num][1], 
                                 door_xyxy_list[num][2], 
                                 door_xyxy_list[num][3], 
                                 'vertical')
            vertical_list.append(item)           
        else:
            # xyxy set
            item = FengShuiItem( door_xyxy_list[num][0], 
                                 door_xyxy_list[num][1], 
                                 door_xyxy_list[num][2], 
                                 door_xyxy_list[num][3], 
                                 'horizontal')
            horizontal_list.append(item)
        #print(item.__repr__())
    
    # Order by the X center. 
    # This means if two items face each other, their coordinates on the X axis are close.
    sorted_ver_list = sorted(vertical_list, key=lambda item: item.get_center()[0])
    # Order by the Y center 
    # This means if two items face each other, their coordinates on the Y axis are close.
    sorted_hor_list = sorted(horizontal_list, key=lambda item: item.get_center()[1])

    symmetry_results = []
    for i in range(len(sorted_ver_list)-1):
        for j in range( i+1 , len(sorted_ver_list)):
            symmetry_results.append([sorted_ver_list[i] , sorted_ver_list[j] , check_door(sorted_ver_list[i] , sorted_ver_list[j] , 'vertical')])
            
    for i in range(len(sorted_hor_list)-1):
        for j in range( i+1 , len(sorted_hor_list)):
            symmetry_results.append([sorted_hor_list[i] , sorted_hor_list[i] , check_door(sorted_hor_list[i] , sorted_hor_list[j] , 'horizontal')])
        
    return symmetry_results

# Input : 平面圖檔案路徑
# Output : 黑白 CV image 陣列 (rows , cols , BRG)
# 將原始圖片轉分析出牆體並2值化成黑白圖
# 去除雜點只取出牆的部分像素
def floor_plan_edit(image_dir):
    image = cv2.imread(image_dir)
    image= cv2.cvtColor(image , cv2.COLOR_BGR2GRAY)
    blur = cv2.bilateralFilter(image, 10, 100, 1000)
    kernel = np.ones((3,3), np.uint8)
    img_erode = cv2.erode(blur, kernel)     # 先侵蝕，將白色小圓點移除
    img_dilate = cv2.dilate(img_erode, kernel)    # 再膨脹，白色小點消失
    ret, result = cv2.threshold(img_dilate , 50 , 255, cv2.THRESH_BINARY)
    return result

# 找出兩點間的像素路徑
# Input: 2 point (x , y)
# Output : 路徑 point list
def bresenham_line(x0, y0, x1, y1):
    x0, y0, x1, y1 = round(x0), round(y0), round(x1), round(y1)
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

# Call by main directly
# Input: 原始平面圖路徑 , 物件2個 , 比較方向
# Output : 再路徑中被阻擋的最大範圍數值 0 ~ 1 (黑色像素)
# Logic : 把圖片作牆體偵測處理 -> 算出兩物件連線路徑 -> 確認方向 -> 以線條計算黑點數量 -> 回傳計算值最大遮罩比例
def check_obstacle_rate(image_dir, item_A, item_B , orientation):

    floor_plan = floor_plan_edit(image_dir)
    start = item_A.get_center()
    end = item_B.get_center()
    line_points = bresenham_line(int(start[0]), int(start[1]), int(end[0]), int(end[1]))

    # Chose the max bounding block on line
    max_black_point = 0

    if orientation == 'vertical':
        scan_range = max((item_A.x2 - item_A.x1), (item_B.x2 - item_B.x1))
        half_range = round(scan_range/2) # Change to int

        # point (x , y)
        for point in line_points:
            # base point on line (point[0], point[1])
            # If is 'vertical' we check horizontal side area so the Y axis would not change
            # __ ^
            # __ |
            left = abs(point[0] - half_range)
            right = abs(point[0] + half_range)
            black_point_counter = 0

            for n in range(left , right + 1):
                # If we found black point on the horizontal side area
                if floor_plan[point[1]][n] == 0:
                    black_point_counter+=1

            max_black_point = max(max_black_point , black_point_counter)
        
    if orientation == 'horizontal':
        scan_range = max((item_A.y2 - item_A.y1), (item_B.y2 - item_B.y1))  
        half_range = round(scan_range/2) # Change to int

        for point in line_points:
            # base point on line (point[0], point[1])
            # If is 'horizontal' we check vertical side area so the X axis would not change
            # |||||| 
            # ----->
            up =  abs(point[1] - half_range)
            down = abs(point[1] +  half_range) # In image down is bigger
            black_point_counter = 0

            for n in range(up , down + 1):
                 # If we found black point on the vertical side area
                if floor_plan[n][point[0]] == 0:
                    black_point_counter+=1

    # Count the max block area
    if scan_range > 0:
        return max_black_point / scan_range
    else:
        return 0
