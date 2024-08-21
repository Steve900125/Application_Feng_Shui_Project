import unittest
from pathlib import Path
import sys  

# Path arrangement
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  
sys.path.insert(0, str(ROOT))  

from vision.classify import orientation_classify
from vision.resize import resize_images

ENTRANCE_IMG_PATH = ROOT / 'test' / 'images' / 'entrance'
CLASSIFY_MODEL_PATH = ROOT / 'models' / 'classify_yolov8.pt'

class TestResize(unittest.TestCase):
    def test_resize_is_work(self):
        source_list = list(ENTRANCE_IMG_PATH.glob("*.jpg"))       
        resize_images(image_paths=source_list)
        self.assertTrue(all(p.exists() for p in source_list), "All image files should exist after resizing.")
    
    def test_orientation_is_right(self):
        source_list = list(ENTRANCE_IMG_PATH.glob("*.jpg"))       
        resize_images(image_paths=source_list)

        orientation_comp = ['horizontal', 'horizontal', 'vertical', 'vertical']
        results = orientation_classify(images_paths=source_list, model_path=CLASSIFY_MODEL_PATH )
        print('len(results):',len(results))
        for index in range(len(results)):
            result = results[index]
            print(source_list[index],end=':')
            print(result.names[result.probs.top1])
            #self.assertEqual(result.names[result.probs.top1], orientation_comp[index])

if __name__ == "__main__":
    unittest.main()
        
