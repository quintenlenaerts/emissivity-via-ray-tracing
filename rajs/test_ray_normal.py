from graphx import GraphX
from Interface import InterfaceBorder

from Vec2 import Vec2
from ray import Ray, RayCollision

GraphX.set_size(-1,15,-1,15)
GraphX.grid_on()

# creating a single line
Interface_Line = InterfaceBorder(InterfaceBorder.BorderType.LINE, 30)
Interface_Line.move_up(8) #setting it about halfway acros the screen
Interface_Line.move_right(-1)


def gen_ray_from_center(center : Vec2, angle : float) -> Ray:
    r = Ray(center.x, center.y, angle, 5) # ray.end_pos now contains a point at the correct incidence
    new_source_pos = r.end_pos

    # switching source and end positions
    r.end_pos = Vec2(r.source_pos.x, r.source_pos.y)
    r.source_pos = Vec2(new_source_pos.x, new_source_pos.y)

    r.ExtendFromSource(1)

    GraphX.draw_ray(r)
    return r
    


GraphX.draw_interface_border(Interface_Line, color='blue')
GraphX.draw_point(Vec2(8,8), color='black')

test_succes = True

angles_to_check = [ x for x in range(200)]

for angle in angles_to_check:
    if (angle > 360):
        angle -= 360

    if (angle == 180 or angle == 0 or angle == 360):
        continue
    
    incide = 90 - min(angle, 180 - angle) if angle <= 180 else 180 - 90 + min(angle, 180 - angle)

    ray = gen_ray_from_center(Vec2(8,8), angle)
    col = Interface_Line.Collision(ray)

    if (col != None):
        print(col.normal)
        test_succes = test_succes and abs(col.normal - incide) <= 0.3
        print("Normal/Angle/180 - min(Angle, 180 - ANgle) : {}/{}/{}".format(col.normal, angle, incide))

    else:
        print("NOT GETTING A HIT LMAO")
        test_succes = False

print("Normal detection is {}".format(test_succes))



GraphX.show()