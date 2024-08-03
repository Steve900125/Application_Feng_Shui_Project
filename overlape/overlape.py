from pathlib import Path
from typing import List, Dict
import sys  

# Path arrangement
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  
sys.path.insert(0, str(ROOT))  # for import modules 

# Fengshui class
from fengshui.item import Item

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

def overlape_rate(items: List[Item]) -> Dict[str, dict]:
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

    return result_dic

if __name__ == '__main__':
    item1 = Item(x1=1, y1=2, x2=5, y2=6, name='Item A', orientation='horizontal')
    item2 = Item(x1=3, y1=4, x2=4, y2=8, name='Item B', orientation='horizontal')

    items = [item1, item2]

    result = overlape_rate(items)
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