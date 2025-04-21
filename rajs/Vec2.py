import numpy as np

class Vec2:

    def __init__(self, x=0 ,y=0 ):
        self.x = x
        self.y = y


    def __add__(self, other):
        # should check if Vec2
        _x = self.x + other.x
        _y = self.y + other.y
        return Vec2(_x, _y)
    
    def __sub__(self, other):
        # should check if Vec2
        _x = self.x - other.x
        _y = self.y - other.y
        return Vec2(_x, _y)
    
    def __str__(self):
        return "Vec2({}, {})".format(self.x, self.y)
    
    @staticmethod
    def seperate_vectors(points):
        """
            Converts a given array of Vec2's to two arrays; the first consisting of
            all the x values and the second one all the y values.

            Makes plotting them with matplotlib a bit easier
        """

        _x = [p.x for p in points]
        _y = [p.y for p in points]

        return _x, _y


    @staticmethod
    def Distance(v1, v2 ):
        return np.sqrt( (v1.x - v2.x)**2 + (v1.y - v2.y)**2)
