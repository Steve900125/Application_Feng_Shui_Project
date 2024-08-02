class Item:
    """
    Defines an object's name and its position on the floor plane for "overlap calculation" and "path obstacle detection".
    """
    def __init__(self, x1=0, y1=0, x2=0, y2=0, name=None, orientation=None):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.name = name
        self.orientation = orientation

    def get_center(self):
        """ Calculate the center of the item. (pixel) """
        center_X = int((self.x1 + self.x2) / 2)
        center_Y = int((self.y1 + self.y2) / 2)
        return {'center_X':center_X, 'centerY':center_Y}
    
    def __repr__(self):
        return  f"Item ({self.x1}, {self.y1}, {self.x2}, {self.y2},'{self.name}', '{self.orientation}')"