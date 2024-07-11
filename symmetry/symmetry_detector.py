
import cv2

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

def door_symmetry_caculate(item_A: FengShuiItem, item_B: FengShuiItem, orientation: str ) -> dict:
    """Check symmetry based on orientation and determine overlap and containment properties."""
    
    # Initialize result dictionary
    result_dic = {
        'symmetry': False, 
        'cross_area_rate': 0,
        'full_contain': False
    }
    
    # Get centers and coordinates based on orientation
    if orientation == 'vertical':
        cent_A, _ = item_A.get_center()
        cent_B, _ = item_B.get_center()
        coord_index = 0  # x coordinate index for vertical orientation
    elif orientation == 'horizontal':
        _, cent_A = item_A.get_center()
        _, cent_B = item_B.get_center()
        coord_index = 1  # y coordinate index for horizontal orientation
    else:
        return result_dic  # If orientation is not recognized, return default results
    
    # Determine order based on centers
    first_item, second_item = (item_A, item_B) if cent_A >= cent_B else (item_B, item_A)
    
    # Extract coordinates for simpler access
    first_min = getattr(first_item, 'xy'[coord_index] + '1')
    first_max = getattr(first_item, 'xy'[coord_index] + '2')
    second_min = getattr(second_item, 'xy'[coord_index] + '1')
    second_max = getattr(second_item, 'xy'[coord_index] + '2')

    # Check if there is no overlap
    if first_min >= second_max:
        return result_dic  # Return default results if no overlap

    # Calculate intersection and union
    intersection = min(first_max, second_max) - max(first_min, second_min)
    union = max(first_max, second_max) - min(first_min, second_min)
    cross_area_rate = intersection / union if union != 0 else 0
    
    # Update dictionary with calculated values
    result_dic['symmetry'] = True
    result_dic['cross_area_rate'] = cross_area_rate
    result_dic['full_contain'] = (second_min >= first_min and second_max <= first_max) or (second_min <= first_min and second_max >= first_max)
    
    return result_dic
    
#return : [[item_A, item_B, {'symmetry': ..., 'cross_area_rate': ...,'full_contain': ...}] , ]
def generate_symmetry_results(door_dict, orientation):
    """Generates symmetry results for a list of sorted items."""
    sorted_items = door_dict[orientation]
    results = []
    for i in range(len(sorted_items) - 1):
        for j in range(i + 1, len(sorted_items)):
            results.append([
                sorted_items[i], sorted_items[j],
                door_symmetry_caculate(sorted_items[i], sorted_items[j], orientation)
            ])
    # [[item_A, item_B, {'symmetry': ..., 'cross_area_rate': ...,'full_contain': ...}] , ]
    return results

def door_symmetry_detect(item_xyxy_list : list , item_orient_list : list , sym_true_filter : bool) -> list:

    # Create a dictionary for initialize doors information
    door_dict = { "vertical":[] , "horizontal":[] }

    if len(item_xyxy_list) != len(item_orient_list):
        return None
    
    # Set the basic information on FengShuiItem
    for xyxy, orient in zip( item_xyxy_list , item_orient_list):
        item = FengShuiItem(xyxy[0], xyxy[1], xyxy[2], xyxy[3], orient)
        door_dict[orient].append(item) # Append by it's "orientation" types
    
    # Sorting the lists within the dictionary by their centers
    # For vertical items, we sort by the x-center, for horizontal items, by the y-center
    door_dict["vertical"] = sorted(door_dict["vertical"], key=lambda item: item.get_center()[0])
    door_dict["horizontal"] = sorted(door_dict["horizontal"], key=lambda item: item.get_center()[1])


    # Calculate symmetry results for both orientations
    symmetry_results = []
    if 'vertical' in door_dict:
        symmetry_results.extend(generate_symmetry_results(door_dict, 'vertical'))
    if 'horizontal' in door_dict:
        symmetry_results.extend(generate_symmetry_results(door_dict, 'horizontal'))
    
    #result[2] is symmetry_results dic
    if sym_true_filter:
        symmetry_results = [result for result in symmetry_results if result[2]['symmetry'] == True]
    # [[item_A, item_B, {'symmetry': ..., 'cross_area_rate': ...,'full_contain': ...}] , ]
    return symmetry_results

    