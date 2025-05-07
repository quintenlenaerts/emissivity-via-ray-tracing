from Vec2 import Vec2
from ray import Ray, RayCollision
from Material import Material
import numpy as np
from optics import planck_spectral_radiance, fresnel_coefs_full, schlick_reflectance
from numba import njit

def vector_to_angle(vec: Vec2) -> float:
    return np.degrees(np.arctan2(vec.y, vec.x)) % 360

def angle_to_vector(angle_deg: float) -> Vec2:
    angle_rad = np.radians(angle_deg)
    return Vec2(np.cos(angle_rad), np.sin(angle_rad))


@njit(cache=True)
def optimized_lrcol(
    in_src_x: float, in_src_y: float,
    in_end_x: float, in_end_y: float,
    surf_src_x: float, surf_src_y: float,
    surf_end_x: float, surf_end_y: float,
    n1: float, k1: float,
    n2: float, k2: float
) -> tuple:
    """
    Numba-optimized replacement for LightRayCollision.__init__ calculations.

    Parameters:
        Incoming ray start and end coordinates,
        Surface ray start and end coordinates,
        Real/imag parts of old and new refractive indices.

    Returns:
        (reflected_angle, transmitted_angle, R, T)
    """
    # --- incident unit vector ---
    dx = in_end_x - in_src_x
    dy = in_end_y - in_src_y
    mag = np.hypot(dx, dy)
    ux = dx / mag
    uy = dy / mag

    # --- surface tangent and normal ---
    sdx = surf_end_x - surf_src_x
    sdy = surf_end_y - surf_src_y
    smag = np.hypot(sdx, sdy)
    tx = sdx / smag
    ty = sdy / smag

    # unit normal
    nx = -ty
    ny = tx
    dp = ux * nx + uy * ny
    if dp > 0.0:
        nx = -nx
        ny = -ny
        dp = -dp

    # --- reflected vector ---
    dot = ux * nx + uy * ny
    rx = ux - 2.0 * dot * nx
    ry = uy - 2.0 * dot * ny
    # angle in degrees [0,360)
    ref_ang = np.arctan2(ry, rx) * 180.0 / np.pi
    if ref_ang < 0.0:
        ref_ang += 360.0

    # --- transmitted vector ---
    cos_i = -dp
    # clamp cos_i
    if cos_i < -1.0:
        cos_i = -1.0
    elif cos_i > 1.0:
        cos_i = 1.0

    eta = n1 / n2
    k_sq = 1.0 - eta * eta * (1.0 - cos_i * cos_i)
    if k_sq < 1e-12:
        k_sq = 1e-12
    sqrt_k = np.sqrt(k_sq)
    tx2 = eta * ux + (eta * cos_i - sqrt_k) * nx
    ty2 = eta * uy + (eta * cos_i - sqrt_k) * ny
    trans_ang = np.arctan2(ty2, tx2) * 180.0 / np.pi
    if trans_ang < 0.0:
        trans_ang += 360.0

    # --- Fresnel coefficients ---
    c_old = n1 + k1 * 1j
    c_new = n2 + k2 * 1j
    R, T, rs, ts, rp, tp = fresnel_coefs_full(c_old, c_new, cos_i)

    return ref_ang, trans_ang, R, T, rs, rp, ts, tp

class LightRayCollision:
    def __init__(self, collision: RayCollision, incoming_ray: Ray, surface_ray: Ray,
                 mat_new: Material, mat_old: Material, wavelength_meters: float):
        
        self.col     = collision
        self.mat_new = mat_new
        self.mat_old = mat_old

        n1, k1 = mat_old.get_complex_index(wavelength_meters*1e6)
        n2, k2 = mat_new.get_complex_index(wavelength_meters*1e6)

        # single Numba call replaces all unit‐vector, reflection, transmission
        # calculations + Fresnel:
        # refl_ang, trans_ang, R, T, rs, rp, ts, tp = optimized_lrcol(
        #     # incoming ray source & end (µm)
        #     incoming_ray.source_pos.x,
        #     incoming_ray.source_pos.y,
        #     incoming_ray.end_pos.x,
        #     incoming_ray.end_pos.y,
        #     # surface segment source & end (µm)
        #     surface_ray.source_pos.x,
        #     surface_ray.source_pos.y,
        #     surface_ray.end_pos.x,
        #     surface_ray.end_pos.y,
        #     # complex indices
        #     n1, k1,
        #     n2, k2
        # )

        # self.reflected_angle   = refl_ang
        # self.transmitted_angle = trans_ang
        # self.reflected_coef    = R
        # self.transmitt_coef    = T

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
        R, T, rs, ts, rp, tp = fresnel_coefs_full(n1 + k1*1j, n2 + k2*1j, cos_i)
        # self.reflected_coef = schlick_reflectance(n1, n2, cos_i)
        self.reflected_coef = R
        self.transmitt_coef  = T

        self.rs_amp = rs
        self.rp_amp = rp

        self.ts_amp = ts
        self.tp_amp = tp

        
   

class LightRay:

    def __init__(self, start_pos : Vec2, direction : float):
        """
            Parameters:
                direction : in degrees
        """
        self._ray : Ray = Ray(start_pos.x, start_pos.y, direction, 1000)
        self._angle = direction
        
        # For the actual emissivity calculations
        self.throughput = 1.0            # how much fractional power is left
        self.emitted_radiance = 0.0     # accumulated W/(m²·sr·m)
        
        self.wavelength : float = 0.9e-6
        self.temperature : float = 300 # kelvin

    def _send(self, interface, col_points_offset) -> LightRayCollision:
        # 1) form 3D origin at half the extrusion height
        ox = self._ray.source_pos.x * 1e-6
        oy = self._ray.source_pos.y * 1e-6
        oz = interface._extrusion_height * 0.5
        origin3 = np.array([[ox, oy, oz]])
        θ = np.radians(self._angle)
        dir3 = np.array([[np.cos(θ), np.sin(θ), 0.0]])

        # push the start point a tiny bit along the ray direction so we don't re-hit the same face
        eps = col_points_offset
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