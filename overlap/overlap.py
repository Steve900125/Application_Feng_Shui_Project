from pathlib import Path
from typing import List, Dict
import sys  

# Path arrangement
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  
sys.path.insert(0, str(ROOT))  # for import modules 

# Fengshui class
from fengshui.item import Item

from draw.draw_item import draw_bounding_boxes
from draw.draw_item import save_to_image

def order_points(items: List[Item]) -> Dict[str, dict]:
    """
    Order points by projection value, and return a dictionary with 4 ordered values.
    
    Parameters:
    - items (List[Item]): The list of items to be ordered.
    
    Returns:
    - Dict[str, dict]: A dictionary with 4 ordered points, each entry contains {value: float, item: Item}.
                       'value' is the projection value used for calculating rate,
                       'item' is the associated item used to check full coverage.
    
    Example:
    items = [
        Item(1, 2, 5, 6, name="Item A", orientation="vertical"),
        Item(3, 4, 7, 8, name="Item B", orientation="horizontal")
    ]
    result = order_points(items)
    {
        'first_point': {'value': 1, 'item': Item (1, 2, 5, 6, 'Item A', 'vertical')},
        'second_point': {'value': 2, 'item': Item (1, 2, 5, 6, 'Item A', 'vertical')},
        'third_point': {'value': 3, 'item': Item (3, 4, 7, 8, 'Item B', 'horizontal')},
        'fourth_point': {'value': 4, 'item': Item (3, 4, 7, 8, 'Item B', 'horizontal')}
    }
    """
    order_list = []

    for item in items:
        proj_dic = item.get_projection_values()
        order_list.append({'value': proj_dic['min'], 'item': item})
        order_list.append({'value': proj_dic['max'], 'item': item})

    # Sort by projection value
    order_list = sorted(order_list, key=lambda x: x['value'])

    order_dic = {}
    order_dic['first_point'] = order_list[0]
    order_dic['second_point'] = order_list[1]
    order_dic['third_point'] = order_list[2]
    order_dic['fourth_point'] = order_list[3]

    return order_dic

def check_full_coverage(order_dic: Dict[str, dict]) -> bool:
    """
    Checks if the first and fourth points in the order dictionary are from the same item,
    indicating full coverage.
    
    Parameters:
    - order_dic (Dict[str, dict]): A dictionary with 4 ordered points.
    
    Returns:
    - bool: True if the first and fourth points are from the same item, indicating full coverage;
            otherwise, False.
    """
    if order_dic['first_point']['item'] == order_dic['fourth_point']['item']:
        return True
    else:
        return False

def cal_inter_rate(order_dic: Dict[str, dict]) -> float:
    """
    Calculate the intersection-over-union (IoU) rate based on the ordered points.
    
    Parameters:
    - order_dic (Dict[str, dict]): A dictionary with 4 ordered points.
    
    Returns:
    - float: The intersection-over-union rate. If there is no intersection, returns 0.
    """
    # Have intersection I__|_I__| |_I_I_|
    if order_dic['first_point']['item'] != order_dic['second_point']['item']:
        union_range = order_dic['fourth_point']['value'] - order_dic['first_point']['value']
        inter_range = order_dic['third_point']['value'] - order_dic['second_point']['value']
        
        if inter_range >= 0 and union_range >0:
            return inter_range / union_range
        else:
             return 0.0
    else:  # No intersection I___I  |___|
        return 0.0

def overlap_rate(items: List[Item]) -> Dict[str, dict]:
    """
    Calculate the overlap rate and check full coverage for a list of items.
    
    Parameters:
    - items (List[Item]): The list of items to be checked.
    
    Returns:
    - Dict[str, dict]: A dictionary containing the items, overlap rate, and full coverage status.
    """
    result_dic = {
        'items': items,
        'rate': 0.0,
        'full_coverage': False
    }

    if items[0].orientation == items[1].orientation:  # Same orientation
        order_dic = order_points(items=items)  # Get order by projection value
        result_dic['full_coverage'] = check_full_coverage(order_dic)
        result_dic['rate'] = cal_inter_rate(order_dic)
        
        if result_dic['rate'] == 1:
            result_dic['full_coverage'] = True
        

    return result_dic

if __name__ == '__main__':
    # Define the items
    items = [
        Item(702.2831420898438, 584.4522705078125, 785.9961547851562, 680.3616943359375, 'door', 'horizontal'),
        Item(106.75611877441406, 106.28368377685547, 223.01071166992188, 212.30514526367188, 'door', 'horizontal'),
        Item(785.08642578125, 497.03765869140625, 879.02880859375, 583.862548828125, 'door', 'vertical'),
        Item(704.1438598632812, 406.7515869140625, 786.35986328125, 498.7106018066406, 'door', 'vertical'),
        Item(784.2705688476562, 854.5673217773438, 869.21044921875, 933.338623046875, 'door', 'horizontal'),
        Item(459.5932312011719, 219.94772338867188, 544.4046630859375, 314.1328430175781, 'door', 'vertical'),
        Item(614.6255493164062, 415.4562683105469, 689.44873046875, 496.7417297363281, 'door', 'horizontal'),
        Item(550.9129638671875, 139.6213836669922, 641.0330200195312, 223.3002471923828, 'door', 'horizontal'),
        Item(624.34619140625, 322.4889831542969, 691.6366577148438, 393.45184326171875, 'door', 'vertical')
    ]
    test = [
        Item(4.2847514152526855, 103.40266418457031, 83.41145324707031, 206.95797729492188,'entrance', 'horizontal'),
        Item(335.0831604003906, 69.35748291015625, 541.446044921875, 300.3077392578125,'kitchen', 'horizontal')
    ]

    items = [items[0],items[3]]
    items =  test


    image_path = ROOT / 'images' / 'FloorPlan (2).jpg'
    image = draw_bounding_boxes(image_path=image_path, item=items[0])
    image = draw_bounding_boxes(image=image, item=items[1])
    save_to_image(image=image,file_name='test.jpg')

    result = overlap_rate(items)
    print(result)

    order_result = order_points(items)
    print("Order Result:")
    print(order_result)

    full_coverage_result = check_full_coverage(order_result)
    print("Full Coverage Result:")
    print(full_coverage_result)

    inter_rate_result = cal_inter_rate(order_result)
    print("Intersection Rate Result:")
    print(inter_rate_result)