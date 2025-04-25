import matplotlib.pyplot as plt
from Interface import InterfaceBorder


from Vec2 import Vec2
from ray import Ray

class GraphX:

    @staticmethod
    def set_size(x_min : float, x_max : float, y_min : float, y_max : float):
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
    
        # plt.close(plt.figure(figsize=(14, 6)))  # width=8 inches, height=6 inches
        

    @staticmethod
    def grid_on():
        plt.grid(True)
    
    @staticmethod
    def show():
        plt.show()

    @staticmethod
    def draw_interface_border(border : InterfaceBorder, color):
        _x, _y = Vec2.seperate_vectors(border._mesh.points)
        plt.plot(_x, _y, color=color)
    

    @staticmethod
    def draw_ray(ray : Ray, color="red", thickness=1):
        plt.plot([ray.source_pos.x, ray.end_pos.x], [ray.source_pos.y, ray.end_pos.y], color=color, linewidth = thickness)
    
    @staticmethod
    def draw_rays(rays : list[Ray], color="red", thickness=1):
        for ray in rays:
            plt.plot([ray.source_pos.x, ray.end_pos.x], [ray.source_pos.y, ray.end_pos.y], color=color, linewidth = thickness, linestyle="--")
    
    @staticmethod
    def draw_point(point : Vec2, marker = "o" , color ="red"):
        plt.plot(point.x, point.y, marker=marker, color=color)  

    @staticmethod
    def draw_pos_history(points : list[Vec2], ray_color='green', point_color='black'):
        for x in range(len(points) - 1):
            _r = Ray(points[x].x,points[x].y)
            _r.SetDirectionByPoint(points[x+1])

            GraphX.draw_ray(_r, ray_color)
            GraphX.draw_point(points[x], "o", point_color)