from Vec2 import Vec2
from ray import Ray
import numpy as np


class Mesh:

    def __init__(self):
        # will be an array of x,y coordinates (Vec2)
        self.points : list[Vec2] = []

    def generate_line(self, line_width: float , point_amount : int):
        self.points = []
        step_size = line_width / (point_amount-1)

        for x in range(point_amount):
            self.points.append(Vec2(step_size * x, 0))
    
    def generate_seesaw(self, line_width: float , point_amount : int, zigzag_factor : float):
        self.points = []
        step_size = line_width / (point_amount-1)

        for x in range(point_amount):
            self.points.append(Vec2(step_size * x, zigzag_factor * (-1)**x))
    
    def generate_sine(self, line_width : float, freq : float, amplitude : float, point_amount : int):
        self.points = []
        step_size = line_width / (point_amount - 1)

        for x in range(point_amount):
            self.points.append(Vec2(step_size * x, amplitude * np.sin(step_size * x * freq) ))

    def __str__(self):
        s = "Mesh["
        for p in self.points:
            s += str(p) + ","

        return s + "]"
    
    def add_offset(self, offset : Vec2):
        for x in range(len(self.points)):
            self.points[x] += offset

    def rayify(self) -> list[Ray]:
        """
            Converts the entire mesh into an array of rays.

            Returns:
                List of rays representing the mesh
        """
        _rays = []
        for x in range(len(self.points) - 1):
            n_ray = Ray(self.points[x].x,self.points[x].y)
            n_ray.SetDirectionByPoint(self.points[x+1])
            n_ray.ExtendFromSource(1e-3)
            _rays.append(n_ray)
        return _rays





