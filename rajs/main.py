from graphx import GraphX
from Interface import InterfaceBorder, Interface

from Vec2 import Vec2
from ray import Ray, RayCollision
from LightRay import LightRay, LightRayCollision
from Material import Material

# window setup
# =================================
GraphX.set_size(-5,35,-5,80)
GraphX.grid_on()

# interface setup
# =================================
InterfaceBorder.BorderType.SINE_SETTINGS(0, 8, 500)
int_top = InterfaceBorder(InterfaceBorder.BorderType.SINE, 30, "TopBorder")
int_top.move_up(50)

InterfaceBorder.BorderType.SINE_SETTINGS(0.8, 4, 500)
int_middle = InterfaceBorder(InterfaceBorder.BorderType.SINE, 30, "MiddleBorder")
int_middle.move_up(10)

InterfaceBorder.BorderType.SINE_SETTINGS(0.3, 5, 550)
int_bot = InterfaceBorder(InterfaceBorder.BorderType.SINE, 30, "BottomBorder")

interface = Interface()

# materials setup
# =================================
mat_air = Material("Air")
mat_air.ReadFromFile("NK/correct_air_big.csv")

print(mat_air.get_complex_n(0.8e-6))


interface.AddLayer(None, int_top, Material("Air", 1, 7))
interface.AddLayer(int_top, int_middle, Material("Diamond", 1.2, 2))
interface.AddLayer(int_middle, int_bot, Material("Diamond2", 1, 0))
interface.AddLayer(int_bot, None, Material("Moly", 1, 0))

print(str(interface))

interface.ConnectBorders()

GraphX.draw_interface_border(int_top, "red")
GraphX.draw_interface_border(int_middle, "green")
GraphX.draw_interface_border(int_bot, "blue")





start_ray = LightRay(Vec2(15, 60), 270)

GraphX.draw_ray(start_ray._ray, color='black')

start_col = start_ray.Send(interface)

print("collision from initial beam at : " + str(start_col.col.point))
print(start_col.transmitted_angle)

GraphX.draw_point(start_col.col.point)


# transray = LightRay(Vec2(15, 50), direction = 270)

trans_ray = LightRay(start_col.col.point, start_col.transmitted_angle)
reflec_ray = LightRay(start_col.col.point, start_col.reflected_angle)

GraphX.draw_ray(trans_ray._ray, color='purple')
# GraphX.draw_ray(reflec_ray._ray, color='yellow')

trans_col =trans_ray.Send(interface)
GraphX.draw_point(trans_col.col.point, color='black')

T_R = LightRay(trans_col.col.point, trans_col.reflected_angle)
GraphX.draw_ray(T_R._ray)




GraphX.show()



