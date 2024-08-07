
from typing import List, Optional, Dict
from pathlib import Path
import sys  

# Path arrangement
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  
sys.path.insert(0, str(ROOT))   # for import moduls 

# Fengshui 
from fengshui.item import Item  # Core class vary important

# Obstical
from obstacle.obstacle import items_obstacle_detect
OBSTACLE_THRESHOLD = 0.5 # 50%

# Overlap
from overlap.overlap import overlap_rate
OVERLAP_THRESHOLD = 0.5 # 50% overlap range

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

def object_to_object(items:List[Item], image_path: Path):

    overlap_results = []
    overlap_results.append(overlap_rate(items= items))
    have_overlap_list = filter_overlap_rate(overlap_results= overlap_results)

    obstacle_results = []
    target =  have_overlap_list[0]
    obstacle_result = items_obstacle_detect(items= target['items'], image_path= image_path)
    obstacle_results.append(obstacle_result)

    result = {
        'overlap_result'  : target,
        'obstacle_result' : obstacle_result
    }
    
    return result
