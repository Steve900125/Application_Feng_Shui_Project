from typing import List, Optional, Dict

class Item:
    """
    Defines an object's name and its position on the floor plane for "overlap calculation" and "path obstacle detection".
    """
    def __init__(self, x1= 0.0, y1= 0.0, x2= 0.0, y2= 0.0, name= None, orientation= None):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.name = name
        self.orientation = orientation

    def get_center(self) -> Dict[str, int]:
        """ Calculate the center of the item. (pixel) """
        center_X = int((self.x1 + self.x2) / 2)
        center_Y = int((self.y1 + self.y2) / 2)
        return {'center_X':center_X, 'center_Y':center_Y}
    
    def get_projection_values(self) -> Dict[str, float]:
        # {min: float, max: float}
        if self.orientation == 'vertical':
            return {'min': self.x1, 'max': self.x2}
        elif self.orientation == 'horizontal':
            return {'min': self.y1, 'max': self.y2}
        else:
            raise ValueError(f"Invalid orientation '{self.orientation}'. Orientation must be 'vertical' or 'horizontal'.")
    
    def get_length_value(self) -> int:
        if self.orientation == 'vertical':
            return round(self.x2 - self.x1)
        elif self.orientation == 'horizontal':
            return round(self.y2 - self.y1)
        else:
            raise ValueError(f"Invalid orientation '{self.orientation}'. Orientation must be 'vertical' or 'horizontal'.")

    def __repr__(self):
        return  f"Item ({self.x1}, {self.y1}, {self.x2}, {self.y2},'{self.name}', '{self.orientation}')"