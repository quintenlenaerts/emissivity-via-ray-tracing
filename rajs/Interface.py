from Mesh import Mesh
from Vec2 import Vec2
from ray import Ray, RayCollision
from Material import Material

class InterfaceBorder:

    class BorderType:
        LINE = 0
        PERLIN = 1
        ZIGZAG = 2
        SINE = 3

        _sine_amp = 0.3
        _sine_freq = 2
        _sine_points = 100

        @staticmethod
        def SINE_SETTINGS(amplitude : float, frequency : float, points : int=100):
            InterfaceBorder.BorderType._sine_amp = amplitude
            InterfaceBorder.BorderType._sine_freq = frequency
            InterfaceBorder.BorderType._sine_points = points



    def __init__(self, border_type : int, border_width : float, border_name : str = "default_border"):
        self._type = border_type
        self._width = border_width
        self._name = border_name

        self._mesh = Mesh()

        if self._type == InterfaceBorder.BorderType.LINE:
            self._mesh.generate_line(self._width, 1000)
        elif self._type == InterfaceBorder.BorderType.ZIGZAG:
            self._mesh.generate_seesaw(self._width, int(self._width), 0.3)
        elif self._type == InterfaceBorder.BorderType.SINE:
            self._mesh.generate_sine(
                self._width, 
                InterfaceBorder.BorderType._sine_freq,
                InterfaceBorder.BorderType._sine_amp,
                InterfaceBorder.BorderType._sine_points)
        

        self._mesh_rays : list[Ray] = None
    
    def move_up(self, height):
        self._mesh.add_offset(Vec2(0, height))
        self._mesh_rays = None # should be recalculated  

    def move_right(self, offset):
        self._mesh.add_offset(Vec2(offset, 0))
        self._mesh_rays = None # should be recalculated
        

    def Collision2(self, ray : Ray) -> RayCollision:
        """
            Splits the mesh up into small rays and performs Ray.Collision() between
            these and the given ray

            Returns:
                None if there is not collision, otherwise a Vec2 with the point of collision.
        """

        if (self._mesh_rays == None):
            self._mesh_rays = self._mesh.rayify()

        # want to get the closest possible collision
        record_distance = 1e10
        record_col = None

        for r in self._mesh_rays:
            col = Ray.Collision(ray, r)
            if (col != None):
                dist = Vec2.Distance(ray.source_pos, col.point)
                if dist <= record_distance:
                    record_distance = dist
                    record_col = col

                    print("Ray 1 angle : " + str(ray.GetAngle()))
                    print("Surface ray anglze : " + str(r.GetAngle()))

            # if (Ray.Collision(ray, r) != None):
            #     return Ray.Collision(ray, r)
        


        return record_col
    
    def Collision(self, ray: Ray) -> RayCollision:
        """
            Splits the mesh into small rays and performs Ray.Collision() between
            these and the given ray, returning the nearest valid collision.
        """
        if self._mesh_rays is None:
            self._mesh_rays = self._mesh.rayify()

        record_distance = float('inf')
        record_col = None

        for surface_ray in self._mesh_rays:
            col = Ray.Collision(ray, surface_ray)
            if col is None:
                continue
            # Distance from ray origin to intersection
            dist = Vec2.Distance(ray.source_pos, col.point)
            # Skip self-collisions at or extremely close to the source
            if dist < 1e-6:
                continue
            if dist < record_distance:
                record_distance = dist
                record_col = col
                # print("Ray incoming angle:", ray.GetAngle())
                # print("Surface segment angle:", surface_ray.GetAngle())

        return record_col

    def __repr__(self):
        return "Border(name : {}, width : {} µm)".format(self._name, self._width)


class Interface:

    def __init__(self):

        self._borders : list[InterfaceBorder] = []
        self._materials : list[Material] = []

    def AddLayer(self, top_border : InterfaceBorder, bottom_border : InterfaceBorder, material : Material):
        if top_border == None:
            # First Layer
            self._borders.append(bottom_border)
        # elif bottom_border == None:
        #     # Last Layer
        #     self._materials.append(material)
        elif bottom_border != None and top_border != None:
            self._borders.append(bottom_border)

        self._materials.append(material)
    
    def __str__(self):
        s = "Interface : \n"

        for x in range(len(self._borders)):
            s+=("BORDER\t from : {}\t to\t {}".format(
                self._materials[x],
                self._materials[x+1]
            )) + "\n"
        
        return s


    def GetBorderByDirectionFromPoint(self, point : Vec2, angle : float = 90, scanning_height : float = 1000) -> int:
        """
            Returns the index of the interfaceborder (_borders array) in a direction from the given point

            Parameters:
                point : Vec2 object with the position
                scanning_height : float dictating how far we look from the point, for most setups, 1000 µm will be enough.
            Returns:
                Index of the interface border (None is no border collision is found)
        """

        # creating a ray pointing up
        up_ray = Ray(point.x, point.y, angle, scanning_height)

        # for determining which interface collision is closest
        record_distance : float = 1e10
        record_index : int = -1

        # going through all our borders and checking if we get collisions
        for x in range(len(self._borders)):
            # print('checking border : ' + str(self._borders[x]))
            col = self._borders[x].Collision(up_ray)

            if (col != None):
                dist = Vec2.Distance(point, col.point)
                if (dist <= record_distance):
                    record_distance = dist
                    record_index = x

        if (record_index == -1):
            return None
        
        return record_index 

    def GetTopBorder(self, point : Vec2, scanning_height : float = 1000) -> int:
        return self.GetBorderByDirectionFromPoint(point, 90, scanning_height)
    
    def GetBotBorder(self, point : Vec2, scanning_height : float = 1000) -> int:
        return self.GetBorderByDirectionFromPoint(point, 270, scanning_height)
    

    def ConnectBorders(self):
        """
            Connects all the borders inside the interface with each other. 
        """

        for x in range(len(self._borders) - 1):

            # these two poitns we will use to 'close' of the current border
            first_point_of_next = Vec2(
                self._borders[x+1]._mesh.points[0].x,
                self._borders[x+1]._mesh.points[0].y
            )

            _mesh_length = len(self._borders[x+1]._mesh.points)

            last_point_of_next = Vec2(
                self._borders[x+1]._mesh.points[_mesh_length - 1].x,
                self._borders[x+1]._mesh.points[_mesh_length - 1].y
            ) 

            # adding them to the current border's mesh
            self._borders[x]._mesh.points.insert(0, first_point_of_next)
            self._borders[x]._mesh.points.append(last_point_of_next)

            # recalculate the rays for this mesh
            self._borders[x]._mesh_rays = None

    def GetMaterial(self, point : Vec2, scanning_height : float = 1000) -> Material:
        """
            Returns the material of the interface at the given point. Works by sending out 2 rays , 90° up and 90° down
            using the GetTopBorder and GetBotBorder functions.

            Returns:
                The material of the interface at the given point. Should always return a value; Returns None in the case
                something goes wrong lmao.
        """

        topBorder : InterfaceBorder = self.GetTopBorder(point, scanning_height)
        botBorder : InterfaceBorder = self.GetBotBorder(point, scanning_height)
        
        if topBorder == None:
            return self._materials[0]
        elif botBorder == None:
            return self._materials[len(self._materials) - 1]
        else:
            for x in range(len(self._borders)):
                if (topBorder._name == self._borders[x]._name):
                    return self._materials[x]
        return None






