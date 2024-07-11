from .symmetry_detector import FengShuiItem # 薛丁格的 bug
import numpy as np
import cv2

def floor_plan_binarization(image_dir):
    image = cv2.imread(image_dir)
    image= cv2.cvtColor(image , cv2.COLOR_BGR2GRAY)
    blur = cv2.bilateralFilter(image, 10, 100, 1000)
    kernel = np.ones((3,3), np.uint8)
    img_erode = cv2.erode(blur, kernel)     
    img_dilate = cv2.dilate(img_erode, kernel)  
    ret, result = cv2.threshold(img_dilate , 50 , 255, cv2.THRESH_BINARY)
    return result

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


def item_obstacle_decete(source, item_A, item_B , orientation):

    floor_plan = floor_plan_binarization(source)
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
            left = point[0] - half_range
            right = point[0] + half_range
            black_point_counter = 0

            # Avoid out of range
            if left < 0:
                left = 0
            if right > floor_plan.shape[1] :
                right = floor_plan.shape[1]

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
            up =  point[1] - half_range
            down = point[1] +  half_range 
            black_point_counter = 0

            # Avoid out of range
            if up < 0:
                up = 0
            if down > floor_plan.shape[0] :
                down = floor_plan.shape[0]

            for n in range(up , down + 1):
                 # If we found black point on the vertical side area
                if floor_plan[n][point[0]] == 0:
                    black_point_counter+=1

    # Count the max block area
    if scan_range > 0:
        return max_black_point / scan_range
    else:
        return 0
