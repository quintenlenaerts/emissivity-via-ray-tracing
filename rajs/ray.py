from Vec2 import Vec2
import numpy as np

class RayCollision:
    def __init__(self, point: Vec2, angle: float, normal: float):
        self.point = point
        self.angle = angle % 360       # angle between the two rays (0–180°)
        self.normal = normal%36    # angle of incidence w.r.t. surface normal (0–90°)

        self.surface_ray : Ray = None

        self.reflected_angle : float = 0 

    def __repr__(self):
        return f"RayCollision(point={self.point}, angle={self.angle:.2f}°, normal={self.normal:.2f}°)"


class Ray:

    def __init__(self, source_X : float = 0, source_Y : float = 0, angle_dir : float = 0, radius_dir : float = 50):
        # where the ray will start
        self.source_pos = Vec2(source_X, source_Y)

        # where the ray "ends"; will be set by
        # SetDirectionByAngle or SetDirectionByPoint
        self.end_pos = Vec2()
        self.SetDirectionByAngle(angle_dir, radius_dir)
    
    def SetDirectionByAngle(self, angle_degrees : float, radius: float):
        self.end_pos.x = self.source_pos.x + radius * np.cos(np.radians(angle_degrees))
        self.end_pos.y = self.source_pos.y + radius * np.sin(np.radians(angle_degrees))
    

    def SetDirectionByPoint(self, point : Vec2):
        self.end_pos = Vec2(point.x, point.y)

    # def GetAngle(self) -> float:
    #     dx = self.end_pos.x - self.source_pos.x
    #     dy = self.end_pos.y - self.source_pos.y
    #     angle_rad = np.arctan2(dy, dx)
    #     angle_deg = np.degrees(angle_rad)
    #     return angle_deg
    
    # def GetAngle(self) -> float:
    #     """
    #         Returns the angle of the ray with respect to the y=0 line of world-space. Output in the range 0 - 360
    #     """
    #     dx = self.end_pos.x - self.source_pos.x
    #     dy = self.end_pos.y - self.source_pos.y
    #     _r = np.degrees(np.arctan(dy/dx))

    #     if dx > 0 and dy < 0:
    #         return _r + 360
    #     elif dx < 0:
    #         return _r + 180
    #     return _r

    def GetAngle(self) -> float:
        """
        Returns the angle of the ray relative to the x-axis (0-360°),
        handling vertical directions without divide-by-zero errors.
        """
        dx = self.end_pos.x - self.source_pos.x
        dy = self.end_pos.y - self.source_pos.y
        # Handle vertical lines explicitly to avoid division by zero
        if abs(dx) < 1e-12:
            return 90.0 if dy > 0 else 270.0
        # Compute base angle using arctan(dy/dx)
        angle = np.degrees(np.arctan(dy / dx))
        # Adjust quadrant
        if dx < 0:
            angle += 180
        elif dy < 0:
            angle += 360
        return angle % 360


    def ExtendFromSource(self, amount: float):
        angle = self.GetAngle()
        self.SetDirectionByAngle(angle, self.Length() + amount)

    def MultiplyFromSource(self, amount: float):
        angle = self.GetAngle()
        self.SetDirectionByAngle(angle, self.Length() * amount)

    
    def Length(self) -> float:
        dx = self.end_pos.x - self.source_pos.x
        dy = self.end_pos.y - self.source_pos.y
        return np.hypot(dx, dy)


    

    @staticmethod
    def Collision(ray1, ray2) -> RayCollision:
        """
            handmade by chatgpt 


            Ray should be considerd the surface-ray; in the InterfaceBorder class there is a function called
            Collision(ray); this splits the border up into smaller rays and then performs ray.Col(incoming ray, border rays)


        """
        x1, y1 = ray1.source_pos.x, ray1.source_pos.y
        x2, y2 = ray1.end_pos.x, ray1.end_pos.y
        x3, y3 = ray2.source_pos.x, ray2.source_pos.y
        x4, y4 = ray2.end_pos.x, ray2.end_pos.y

        den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if den == 0:
            return None  # parallel

        # Intersection point
        px = ((x1*y2 - y1*x2)*(x3 - x4) - (x1 - x2)*(x3*y4 - y3*x4)) / den
        py = ((x1*y2 - y1*x2)*(y3 - y4) - (y1 - y2)*(x3*y4 - y3*x4)) / den
        intersection = Vec2(px, py)

        def within_segment(p, a, b, tol=1e-8):
            return min(a, b) - tol <= p <= max(a, b) + tol

        if (within_segment(px, x1, x2) and within_segment(py, y1, y2) and
            within_segment(px, x3, x4) and within_segment(py, y3, y4)):

            angle1 = ray1.GetAngle()
            angle2 = ray2.GetAngle()
            angle_diff = (angle1 - angle2) % 360

            dx_surf = x4 - x3
            dy_surf = y4 - y3

            if abs(dx_surf) < 1e-8:  # vertical surface
                normal_angle = 180 if dy_surf > 0 else 0
            elif abs(dy_surf) < 1e-8:  # horizontal surface
                normal_angle = 90 if dx_surf > 0 else 270
            else:
                normal_vec = Vec2(-dy_surf, dx_surf)
                normal_angle = np.degrees(np.arctan2(normal_vec.y, normal_vec.x)) % 360

            # Ensure reflected ray is on correct side of normal
            incident_vec = Vec2(x2 - x1, y2 - y1)
            in_len = np.hypot(incident_vec.x, incident_vec.y)
            unit_in = Vec2(incident_vec.x / in_len, incident_vec.y / in_len)

            normal_vec = Vec2(np.cos(np.radians(normal_angle)), np.sin(np.radians(normal_angle)))
            dot = unit_in.x * normal_vec.x + unit_in.y * normal_vec.y

            refl_vec = Vec2(
                unit_in.x - 2 * dot * normal_vec.x,
                unit_in.y - 2 * dot * normal_vec.y
            )

            reflected_angle = np.degrees(np.arctan2(refl_vec.y, refl_vec.x)) % 360

            r = RayCollision(intersection, angle_diff, normal_angle)
            r.surface_ray = ray2
            r.reflected_angle = reflected_angle
            return r
    

        return None


        

