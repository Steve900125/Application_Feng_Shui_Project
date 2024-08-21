from pathlib import Path
import sys  

# Path arrangement
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  
sys.path.insert(0, str(ROOT))   # for import moduls 

# Fengshui 
from fengshui.item import Item  # Core class vary important
from fengshui.indep_assess import object_to_object

TEST_IMAGE = ROOT / 'test' / 'images' /'FloorPlan (2).jpg'

if __name__ == "__main__":
    test = [
        Item(4.2847514152526855, 103.40266418457031, 83.41145324707031, 206.95797729492188,'entrance', 'horizontal'),
        Item(335.0831604003906, 69.35748291015625, 541.446044921875, 300.3077392578125,'kitchen', 'horizontal')
    ]
    result = object_to_object(items= test, image_path= TEST_IMAGE)

    print('overlap rate : ',result['overlap_result']['rate'])
    print('obstacle rate : ',result['obstacle_result']['rate'])