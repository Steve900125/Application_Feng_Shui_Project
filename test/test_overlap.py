import unittest
from pathlib import Path
import sys  

# Path arrangement
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  
sys.path.insert(0, str(ROOT))   # for import moduls 

# Fengshui 
from fengshui.item import Item  # Core class vary important
from overlap.overlap import overlap_rate


class TestOverlapeCal(unittest.TestCase):
    def test_int_val(self):
        # orientation : same
        # name : same
        # values : int
        items = [
            Item(x1=100, y1=100, x2=200, y2=200, orientation='vertical', name='door'),
            Item(x1=150, y1=300, x2=250, y2=800, orientation='vertical', name='door'),
        ]
        result = overlap_rate(items= items)
        union_range = 250 - 100
        inter_range = 200 - 150
        rate = inter_range / union_range
        self.assertEqual(result['rate'], rate)
    
    def test_float_val_ver(self):
        # orientation : same
        # name : same
        # values : float
        items = [
            Item(x1=100.001, y1=100.0, x2=200.3910, y2=200.0, orientation='vertical', name='door'),
            Item(x1=150.004, y1=300.0, x2=250.3041, y2=800.0, orientation='vertical', name='door'),
        ]
        result = overlap_rate(items= items)
        union_range = 250.3041 - 100.001
        inter_range = 200.3910 - 150.004
        rate = inter_range / union_range
        self.assertEqual(result['rate'], rate)
        self.assertEqual(result['full_coverage'], False)
    
    def test_float_val_hor(self):
        # orientation : same
        # name : same
        # values : float
        items = [
            Item(x1=100.001, y1=100.001, x2=200.3910, y2=240.011, orientation='horizontal', name='door'),
            Item(x1=150.004, y1=130.01, x2=250.3041, y2=400.0, orientation='horizontal', name='door'),
        ]
        result = overlap_rate(items= items)
        union_range = 400.0 - 100.001
        inter_range = 240.011 - 130.01
        rate = inter_range / union_range
        self.assertEqual(result['rate'], rate)
        self.assertEqual(result['full_coverage'], False)

    def test_diff_ori(self):
        # orientation : diff
        # name : same
        # values : float
        items = [
            Item(x1=100.001, y1=100.0, x2=200.3910, y2=200.0, orientation='horizontal', name='door'),
            Item(x1=150.004, y1=300.0, x2=250.3041, y2=800.0, orientation='vertical', name='door'),
        ]
        result = overlap_rate(items= items)
        self.assertEqual(result['rate'], 0)
    
    def test_diff_name(self):
        # orientation : same
        # name : same
        # values : float
        items = [
            Item(x1=100.001, y1=100.0, x2=200.3910, y2=200.0, orientation='vertical', name='door'),
            Item(x1=150.004, y1=300.0, x2=250.3041, y2=800.0, orientation='vertical', name='dog'),
        ]
        result = overlap_rate(items= items)
        union_range = 250.3041 - 100.001
        inter_range = 200.3910 - 150.004
        rate = inter_range / union_range
        self.assertEqual(result['rate'], rate)
    
    def test_all_same(self):
        # orientation : same
        # name : same
        # values : float
        items = [
            Item(x1=100.001, y1=100.0, x2=200.3910, y2=200.0, orientation='vertical', name='door'),
            Item(x1=100.001, y1=100.0, x2=200.3910, y2=200.0, orientation='vertical', name='door')
        ]
        result = overlap_rate(items= items)
        union_range = 200.3910 - 100.001
        inter_range = 200.3910 - 100.001
        rate = inter_range / union_range
        self.assertEqual(result['rate'], rate)
        self.assertEqual(result['full_coverage'], True)

    def test_empty_items(self):
        items = [Item(), Item()]  # Doesn't initial Items
        with self.assertRaises(ValueError):
            overlap_rate(items=items)

if __name__ == "__main__":
    unittest.main()
