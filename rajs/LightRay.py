from Vec2 import Vec2
from ray import Ray, RayCollision
from Material import Material
import numpy as np
from optics import planck_spectral_radiance, fresnel_coefs, schlick_reflectance


def vector_to_angle(vec: Vec2) -> float:
    return np.degrees(np.arctan2(vec.y, vec.x)) % 360


def angle_to_vector(angle_deg: float) -> Vec2:
    angle_rad = np.radians(angle_deg)
    return Vec2(np.cos(angle_rad), np.sin(angle_rad))

class LightRayCollision:
    def __init__(self, collision: RayCollision, incoming_ray: Ray, surface_ray: Ray,
                 mat_new: Material, mat_old: Material, wavelength_meters: float):

        self.col     = collision
        self.mat_new = mat_new
        self.mat_old = mat_old

        # --- 1) compute the incident unit‐vector in 2D ---
        in_vec = incoming_ray.end_pos - incoming_ray.source_pos
        in_len = np.hypot(in_vec.x, in_vec.y)
        unit_in = Vec2(in_vec.x / in_len, in_vec.y / in_len)

        # --- 2) compute the surface‐tangent and normal (just like you already do) ---
        surf_vec  = surface_ray.end_pos - surface_ray.source_pos
        surf_len  = np.hypot(surf_vec.x, surf_vec.y)
        unit_tan  = Vec2(surf_vec.x / surf_len, surf_vec.y / surf_len)
        unit_norm = Vec2(-unit_tan.y, unit_tan.x)
        # flip if it’s pointing the wrong way
        if (unit_in.x*unit_norm.x + unit_in.y*unit_norm.y) > 0:
            unit_norm = Vec2(-unit_norm.x, -unit_norm.y)

        # --- 3) *now* compute the true reflected‐vector off that normal: ---
        dot = unit_in.x*unit_norm.x + unit_in.y*unit_norm.y
        refl = Vec2(
          unit_in.x - 2*dot*unit_norm.x,
          unit_in.y - 2*dot*unit_norm.y
        )
        self.reflected_angle = vector_to_angle(refl)

        # --- 4) compute transmitted‐vector & Fresnel as you already do: ---
        n1, k1 = mat_old.get_complex_index(wavelength_meters*1e6)
        n2, k2 = mat_new.get_complex_index(wavelength_meters*1e6)
        eta     = n1/n2

        cos_i = -np.clip(unit_in.x*unit_norm.x + unit_in.y*unit_norm.y, -1.0, 1.0)
        k_sq  = 1 - eta**2 * (1 - cos_i**2)
        k_sq  = max(k_sq, 1e-12)

        trans_vec = Vec2(
          eta*unit_in.x + (eta*cos_i - np.sqrt(k_sq))*unit_norm.x,
          eta*unit_in.y + (eta*cos_i - np.sqrt(k_sq))*unit_norm.y
        )
        self.transmitted_angle = vector_to_angle(trans_vec)

        # fresnel coefficients (power)
        R, T = fresnel_coefs(n1 + k1*1j, n2 + k2*1j, cos_i)
        # self.reflected_coef = schlick_reflectance(n1, n2, cos_i)
        self.reflected_coef = R
        self.transmitt_coef  = T
    # def __init__(self, collision: RayCollision, incoming_ray: Ray, surface_ray: Ray,
    #              mat_new: Material, mat_old: Material, wavelength_meters: float):
        
    #     self.col = collision
    #     self.mat_new = mat_new
    #     self.mat_old = mat_old

    #     self.reflected_angle = collision.reflected_angle
    #     self.transmitted_angle = None

    #     # --- Vector-based refraction calculation ---
    #     # 1) Build unit incident vector
    #     in_vec = incoming_ray.end_pos - incoming_ray.source_pos
    #     in_len = np.hypot(in_vec.x, in_vec.y)
    #     unit_in = Vec2(in_vec.x / in_len, in_vec.y / in_len)

    #     # 2) Build unit surface tangent
    #     surf_vec = surface_ray.end_pos - surface_ray.source_pos
    #     surf_len = np.hypot(surf_vec.x, surf_vec.y)
    #     unit_tangent = Vec2(surf_vec.x / surf_len, surf_vec.y / surf_len)

    #     # 3) Compute unit normal by rotating tangent 90° CCW
    #     unit_norm = Vec2(-unit_tangent.y, unit_tangent.x)
    #     # Flip normal if it points in the same general direction as the incident ray
    #     if (unit_in.x * unit_norm.x + unit_in.y * unit_norm.y) > 0:
    #         unit_norm = Vec2(-unit_norm.x, -unit_norm.y)

    #     # 4) Ratio of refractive indices

    #     n1, k1 = mat_old.get_complex_index(wavelength_meters * 1e6)
    #     n2, k2 = mat_new.get_complex_index(wavelength_meters * 1e6)
        
    #     eta = n1 / n2

    #     # 5) Compute cos(theta_i)
    #     cos_i = -(unit_in.x * unit_norm.x + unit_in.y * unit_norm.y)
    #     cos_i = np.clip(cos_i, -1.0, 1.0)

    #     # 6) Compute and store angle of incidence
    #     # will use this for the fresnel equations
    #     self.incidence_angle = np.degrees(np.arccos(cos_i))  # angle in degrees

    #     # 7) Compute k = 1 - eta^2 * (1 - cos_i^2)
    #     k = 1 - eta**2 * (1 - cos_i**2)
    #     if k < 0:
    #         # Total internal reflection: fudge to almost zero transmission
    #         k = 1e-12

    #     # 7) Compute transmitted vector
    #     trans_vec = Vec2(
    #         eta * unit_in.x + (eta * cos_i - np.sqrt(k)) * unit_norm.x,
    #         eta * unit_in.y + (eta * cos_i - np.sqrt(k)) * unit_norm.y
    #     )
    #     self.transmitted_angle = vector_to_angle(trans_vec)

    #     # 8 setting the fresnel coeffs
    #     R,T = fresnel_coefs(n1 + k1 * 1j, n2 + k2 * 1j, cos_i)
    #     self.reflected_coef = R
    #     # dont actually have to calculate T ==> can use Schlikreflectance approx
    #     self.transmitt_coef = T


class LightRay:

    def __init__(self, start_pos : Vec2, direction : float):
        """
            Parameters:
                direction : in degrees
        """

        self._ray : Ray = Ray(start_pos.x, start_pos.y, direction, 1000)
        self._angle = direction

        # used by the Interface class to keep track of where the light
        # ray is in the stack
        # self._layer_identifier : int  = 0
        
        # For the actual emissivity calculations
        self.throughput = 1.0            # how much fractional power is left
        self.emitted_radiance = 0.0     # accumulated W/(m²·sr·m)
        
        self.wavelength : float = 0.9e-6
        self.temperature : float = 300 # kelvin

    def _send(self, interface) -> LightRayCollision:
        # 1) form 3D origin at half the extrusion height
        ox = self._ray.source_pos.x * 1e-6
        oy = self._ray.source_pos.y * 1e-6
        oz = interface._extrusion_height * 0.5
        origin3 = np.array([[ox, oy, oz]])
        θ = np.radians(self._angle)
        dir3 = np.array([[np.cos(θ), np.sin(θ), 0.0]])

        # push the start point a tiny bit along the ray direction so we don't re-hit the same face
        eps = 1e-5
        origin3 += dir3 * eps

        # 2) Embree intersect
        locations, _, tri_idx_arr = interface._intersector.intersects_location(
            ray_origins=origin3,
            ray_directions=dir3,
            multiple_hits=False
        )
        if len(locations) == 0:
            return None
        tri_idx = int(tri_idx_arr[0])

        # 3) ignore top/bottom caps
        normal3 = interface._trimesh.face_normals[tri_idx]
        if abs(normal3[2]) > 1e-6:
            return None

        # 4) map face->segment and retrieve border index
        seg_idx = interface._tm_face_to_seg.get(tri_idx, None)
        if seg_idx is None:
            return None
        border_idx = interface._tm_seg_border[seg_idx]

        # 5) get precomputed surface_ray
        surface_ray = interface._tm_surface_rays[seg_idx]

        # 6) convert hit back to 2D Vec2 (µm)
        hit3 = locations[0]
        hit2 = Vec2(hit3[0] * 1e6, hit3[1] * 1e6)

        # 7) build RayCollision and compute Fresnel via LightRayCollision
        col = RayCollision(hit2, 0.0, 0.0)
        col.surface_ray = surface_ray
        mat_old = interface._materials[border_idx]
        mat_new = interface._materials[border_idx + 1]
        lr = LightRayCollision(col, self._ray, surface_ray, mat_new, mat_old, self.wavelength)
        return lr

    # def _send(self, interface ) -> LightRayCollision:
    #     """
    #         Will send out a light from the initial position and angle in the interface.
            
    #         1. Determine what border the light ray will hit in this direction   
    #             use interface.GetBorderByDirectionFromPoint
    #             if no border gets hit ==> light ray outside material
    #             discard for now

    #         2. Construct the LightRayCollisionobject
    #             calculate R,T coeffs (1) for now
    #             calculate reflected angle and transmitted angle using snel

    #         3. Return light ray collision object

            
    #         Returns:
    #             None is no border is hit (light ray outside material). LightRayCollsion object if a border is hit
    #     """

    #     border_index : int = interface.GetBorderByDirectionFromPoint(self._ray.source_pos, self._angle, 50000)
    #     if (border_index == None):
    #         return None
        
    #     border = interface._borders[border_index]

    #     # can optimize this; this is already being done once in GetBorderByDireciotn
    #     col : RayCollision = border.Collision(self._ray)
    #     if col == None:
    #         return None

    #     # can also be optimized (we can kinda gues in what material the light ray should be instead of
    #     # doing a shitton of calculations)
    #     new_mat , old_mat = self._get_materials(interface, border_index, col) 

    #     # updating this light rays settings
    #     lr = LightRayCollision(col, self._ray, col.surface_ray, new_mat, old_mat, self.wavelength)
    #     # self._update_throughput(lr)

    #     return lr

    # def _update_throughput(self, LR_col : LightRayCollision):
    #     """
    #         Uses the given lightraycollision object to correctly adjust the throughput and spectral radiance of this LightRay.
    #         Is called by the Send() function at the end.
    #     """
        
    #     # figuring out the distance that this light ray travlled in the layer
    #     d = Vec2.Distance(self._ray.source_pos, LR_col.col.point)

    #     # getting the absorption coefficient of the material we just traversed through
    #     alpha = LR_col.mat_old.get_absorption_coefficient(self.wavelength)

    #     # calculating the attenuteitn and emissivity of the layer
    #     attenuation = np.exp(-1 * d * alpha)
    #     emis_layer = 1 - attenuation
    #     B = planck_spectral_radiance(self.wavelength * 1e-6, self.temperature)
    #     emit = emis_layer * B

    #     # adjusting our variables
    #     self.emitted_radiance += self.throughput * emit
    #     self.throughput *= attenuation

    def _get_materials(self, interface, border_index : int, col : RayCollision) -> tuple[Material]:
        """
            Returns:
                A tuple consisitng of: the new material, the current material 
        """
        _surface_ray_angle = col.surface_ray.GetAngle()

        if _surface_ray_angle != -90 and _surface_ray_angle != 90:
            surface_ray = col.surface_ray
            surface_vec = surface_ray.end_pos - surface_ray.source_pos

            # Surface normal (90° CCW rotation)
            normal_vec = Vec2(-surface_vec.y, surface_vec.x)
            normal_len = np.hypot(normal_vec.x, normal_vec.y)
            unit_normal = Vec2(normal_vec.x / normal_len, normal_vec.y / normal_len)

            # Incoming direction
            incoming_vec = self._ray.end_pos - self._ray.source_pos
            incoming_len = np.hypot(incoming_vec.x, incoming_vec.y)
            unit_incoming = Vec2(incoming_vec.x / incoming_len, incoming_vec.y / incoming_len)

            # Determine direction with dot product
            dot = unit_incoming.x * unit_normal.x + unit_incoming.y * unit_normal.y

            if dot < 0:
                # Ray is entering new material
                return interface._materials[border_index], interface._materials[border_index + 1]
            else:
                # Ray is exiting current material
                return interface._materials[border_index + 1], interface._materials[border_index]

        # wanner we een verticale muur raken moeten we eig checken ofdat we in het materiaal zitten
        # aangezien dit bepaald welke transititei we ondergaan
        # kunnen dit voorlopig skippen aangezien we mogen aannemen dat we altijd in ons materiaal gaan zitten
        # en dus geen zijwaartse lichtinval gaan hebben.

        # dus eig moete we hier checken 
        _is_inside_material = True
        if (_is_inside_material):
            return interface._materials[0], interface._materials[border_index+1]