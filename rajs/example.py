from graphx import GraphX
from Interface import InterfaceBorder, Interface

from Vec2 import Vec2
from ray import Ray, RayCollision
from LightRay import LightRay, LightRayCollision
from Material import Material


# window setup
# =================================

thickness_top_layer = 100
thickness_bottom_layer = 500
width = 30000

# setting size of the graph that will be show
GraphX.set_size(-width/2*1.05, width/2*1.05, -thickness_bottom_layer * 0.05, thickness_top_layer + thickness_bottom_layer *1.05)
GraphX.grid_on()

# interface setup
# =================================
InterfaceBorder.BorderType.SINE_SETTINGS(11, 8, 500)                         # generating a sine border
int_top = InterfaceBorder(InterfaceBorder.BorderType.SINE, width, "TopBorder") # creating the actual border
int_top.move_up(thickness_top_layer + thickness_bottom_layer)               # correct height
int_top.move_right(-width/2)                                                # align properly

InterfaceBorder.BorderType.SINE_SETTINGS(0.8, 4, 500)
int_middle = InterfaceBorder(InterfaceBorder.BorderType.SINE, width, "MiddleBorder")
int_middle.move_up(thickness_bottom_layer)
int_middle.move_right(-width/2)

InterfaceBorder.BorderType.SINE_SETTINGS(0.3, 5, 500)
int_bot = InterfaceBorder(InterfaceBorder.BorderType.SINE, width, "BottomBorder")
int_bot.move_right(-width/2)

interface = Interface()

# materials setup
# =================================
lab_olive_mat = Material("Olive", "./lab-olive.csv", 800)
air_mat = Material("Air", "", 800, Material.MateterialTypes.SINGLE_NK)
air_mat.set_single_nk(1, 0)

# creating the interface by specifing the enclosing borders and their respective materials
# must be done in the correct order 
interface.AddLayer(None, int_top, air_mat)
interface.AddLayer(int_top, int_middle, lab_olive_mat)
interface.AddLayer(int_middle, int_bot, lab_olive_mat)
interface.AddLayer(int_bot, None, air_mat)

interface.ConnectBorders()
interface.build_trimesh()

# telling matplotlib to draw the interface ==> must still call GraphX.show()
GraphX.draw_interface_border(int_top, "red")
GraphX.draw_interface_border(int_middle, "green")
GraphX.draw_interface_border(int_bot, "blue")


import numpy as np
_n = 0


record_bounces = 0
record_history = []

while True:

    _pos, _, _ = interface.TraceOneRay(Vec2(0, thickness_bottom_layer + thickness_top_layer - 50),
                                       360-5, 1.2 * 1e-6, 800, debug=False)
    _n +=1
    
    if len(_pos) > record_bounces:
        record_bounces = len(_pos)
        record_history = _pos
    
    if _n > 1000:
        if record_bounces == 0:
            print("WHAT T HE FLIPPIY DID FUKC")
            break
        GraphX.draw_pos_history(record_history)
        print(record_bounces)
        break



GraphX.show()



