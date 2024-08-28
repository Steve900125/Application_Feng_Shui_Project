import unittest
from pathlib import Path
import sys
import shutil
import cv2
from PIL import Image
import numpy as np

# Path arrangement
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
sys.path.insert(0, str(ROOT))  # for import modules

from obstacle.obstacle import floor_plan_binarization
from obstacle.obstacle import apply_white_boxes

IMAGES_PATH = ROOT / 'test' / 'val_images'
SAVE_BIN_DIR = ROOT / 'test' / 'bin_images'


class TestBina(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Ensure SAVE_BIN_DIR exists and clear it before tests
        if SAVE_BIN_DIR.exists():
            shutil.rmtree(SAVE_BIN_DIR)
        SAVE_BIN_DIR.mkdir(parents=True, exist_ok=True)

    def test_binarization(self):
        for image_path in IMAGES_PATH.glob('*.*'):  # Adjust the pattern if necessary
            with self.subTest(image=image_path):
                
                # Run the binarization process
                bin_image = floor_plan_binarization(image_path=image_path)

                # Define the save path with the same filename in the new directory
                save_path = SAVE_BIN_DIR / image_path.name

                # Save the binarized image using PIL (to avoid cv2.imwrite issues with paths)
                pil_image = Image.fromarray(bin_image)
                pil_image.save(str(save_path))

                # Check if the file was saved correctly
                self.assertTrue(save_path.exists())

if __name__ == "__main__":
    unittest.main()
