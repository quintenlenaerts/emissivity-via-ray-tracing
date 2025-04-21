import numpy as np
import trimesh
import matplotlib.pyplot as plt
# Assuming your SingleLayerSubstrate class is importable
# from core.layer import SingleLayerSubstrate # Adjust import if necessary
# Assuming RayMeshIntersector is available
from trimesh.ray.ray_pyembree import RayMeshIntersector

from core.core_optical_functions import planck_spectral_radiance, attenuation, schlick_reflectance, reflect2, refract2
# --- Helper Functions ---

def normalize(v):
    """Normalizes a 2D or 3D vector."""
    norm = np.linalg.norm(v)
    # Handle zero vector case to avoid division by zero
    if norm < 1e-12:
        return v * 0.0 # Return zero vector of the same shape
    return v / norm

def reflect(incident_dir, normal):
    """Calculates the reflection vector."""
    # Ensure inputs are normalized
    incident_dir = normalize(incident_dir)
    normal = normalize(normal)
    # Reflection formula: R = I - 2 * dot(I, N) * N
    return incident_dir - 2 * np.dot(incident_dir, normal) * normal

def refract(incident_dir, normal, n_ratio):
    """
    Calculates the refraction vector using Snell's Law.
    Handles Total Internal Reflection (TIR).

    Args:
        incident_dir (np.ndarray): Normalized incident direction vector.
        normal (np.ndarray): Normalized surface normal vector pointing outwards from the surface.
        n_ratio (float): Ratio of refractive indices (n_incident / n_transmitted).

    Returns:
        np.ndarray or None: Normalized refracted direction vector, or None if TIR occurs.
    """
    # Ensure inputs are normalized
    incident_dir = normalize(incident_dir)
    normal = normalize(normal)

    cos_i = -np.dot(normal, incident_dir) # Cosine of incident angle
    # Clamp cos_i to avoid potential floating point issues near grazing angles
    cos_i = np.clip(cos_i, -1.0, 1.0)

    sin_t2 = n_ratio**2 * (1.0 - cos_i**2) # sin^2(theta_transmitted) using Snell's Law

    # Check for Total Internal Reflection (TIR)
    if sin_t2 > 1.0:
        return None # TIR occurs

    cos_t = np.sqrt(1.0 - sin_t2)
    refracted_dir = n_ratio * incident_dir + (n_ratio * cos_i - cos_t) * normal
    return normalize(refracted_dir)

def schlick_reflectance(n1, n2, cos_i):
    """
    Calculates reflectance using Schlick's approximation for unpolarized light.

    Args:
        n1 (float): Refractive index of the incident medium.
        n2 (float): Refractive index of the transmitting medium.
        cos_i (float): Cosine of the angle of incidence (must be >= 0).

    Returns:
        float: Reflectance probability (0 to 1).
    """
    # Ensure cos_i is non-negative (angle between 0 and 90 degrees)
    cos_i = max(0.0, cos_i)
    # Calculate R0, the reflectance at normal incidence
    r0_num = (n1 - n2)
    r0_den = (n1 + n2)
    if abs(r0_den) < 1e-12: # Avoid division by zero if indices are equal
        r0 = 0.0
    else:
        r0 = (r0_num / r0_den)**2
    # Schlick's approximation formula
    return r0 + (1.0 - r0) * ((1.0 - cos_i)**5)

# --- Main Ray Tracer Class ---

class RayTracer2D:
    """
    Traces a 2D ray within a geometry defined by a SingleLayerSubstrate,
    handling reflections, refractions, and absorption. Uses extrusion to 3D
    for robust intersection finding via trimesh.RayMeshIntersector.
    """
    
    EXTRUSION_HEIGHT = 1e-3
    def __init__(self, substrate):
        """
        Initializes the tracer with the substrate geometry.

        Args:
            substrate (SingleLayerSubstrate): An instance containing the Path2D
                                             geometry and material properties.
        """
        # Validate input is the expected type (replace with your actual class name)
        # if not isinstance(substrate, SingleLayerSubstrate):
        #     raise TypeError("substrate must be an instance of SingleLayerSubstrate")
        self.substrate = substrate
        # Pre-build the 3D mesh and intersector for efficiency
        print("Initializing RayTracer2D: Building intersector...")
        self.substrate._build_intersector() # Ensure 3D mesh and intersector are ready
        print("Intersector ready.")

        # Define properties of the surrounding medium (e.g., air)
        self.surrounding_n = 1.0
        self.surrounding_k = 0.0 # Assuming no absorption/emission outside
        
        # pre‑allocate the 1×3 arrays we’ll reuse every bounce
        self._ray_origins_3d = np.zeros((1,3), dtype=float)
        self._ray_dirs_3d    = np.zeros((1,3), dtype=float)
        # cache material (n,k) lookups per wavelength
        self._nkt_cache = {}

    def _get_medium_properties(self, position_2d):
        """
        Determines the refractive index (n) and extinction coefficient (k)
        at a given 2D position based on whether it's inside the substrate.

        Args:
            position_2d (np.ndarray): The (x, y) position to check.

        Returns:
            tuple: (n, k, is_inside) where is_inside is True if the point
                   is within the substrate's path boundary.
        """
        # Check if position is inside the substrate polygon using the Path2D method
        # Note: contains_points expects shape (N, 2)
        
        # Ray is spawned inside the material anyway by constraints, and it is killed when it leaves so technically it will always be inside. 
        is_inside = True # Default to outside
        
        # try:
        #     # Ensure the 3D mesh is available
        #     if self.substrate._mesh_3d is None:
        #          print("Error: Substrate 3D mesh not available for containment check.")
        #          # Attempt to build it? Or just assume outside? Assume outside for safety.
        #          return self.surrounding_n, self.surrounding_k, False

        #     # --- Use 3D mesh containment check ---
        #     # Create a 3D point from the 2D position, placing it mid-way along the extrusion height
        #     position_3d = np.array([[position_2d[0], position_2d[1], self.EXTRUSION_HEIGHT / 2.0]])

        #     # Check if the 3D point is inside the mesh volume
        #     # mesh.contains returns a boolean array for the input points
        #     is_inside_array = self.substrate._mesh_3d.contains(position_3d)
        #     is_inside = is_inside_array[0] # Get the boolean result for the single point
            
            
        # except AttributeError as e:
        #      print(f"Error: Missing required attribute for mesh containment check ({e}). Assuming outside.")
        #      is_inside = False
        # except Exception as e:
        #      print(f"Warning: Mesh containment check failed for position {position_2d}. Error: {e}. Assuming outside.")
        #      is_inside = False
            
        if is_inside:                        
            n, k = self.substrate.material.get_complex_index(self.current_wavelength_meters * 1e6) # Convert to microns for lookup
            medium_temperature = self.substrate.material.temperature # Get temperature from material
            # ----------------------------------------------------
            return n, k, True, medium_temperature  
        else:
            # Outside the substrate, use surrounding medium properties
            return self.surrounding_n, self.surrounding_k, False, 293

    def _get_surface_normal(self, location_3d, triangle_index):
        """
        Estimates the 2D surface normal at the intersection point.
        Relies on the normal of the hit triangle in the extruded 3D mesh.

        Args:
            location_3d (np.ndarray): The 3D intersection point.
            triangle_index (int): Index of the triangle hit in the 3D mesh.

        Returns:
            np.ndarray: Normalized 2D normal vector, or None if ambiguous.
        """
        # Get the 3D normal of the hit triangle face
        try:
            normal_3d = self.substrate._mesh_3d.face_normals[triangle_index]
        except IndexError:
             print(f"Error: Invalid triangle index {triangle_index} provided.")
             return np.array([1.0, 0.0]) # Return default normal

        # If the hit face is vertical (side of extrusion), normal is in XY plane
        if np.abs(normal_3d[2]) < 1e-6: # Tolerance for vertical face check
             normal_2d = normalize(normal_3d[:2])
             return normal_2d
        else:
             # Hit top or bottom face - normal is +/- Z.
             # This means the ray hit the flat cap created by extrusion, likely
             # indicating it hit exactly parallel to an original 2D edge or vertex.
             # Determining the correct 2D normal is ambiguous here using this method.

             # --- Fallback Strategy: Find nearest 2D segment normal ---
             # This is less accurate and slower. Consider improving geometry or intersection method if this occurs frequently.
             print(f"Warning: Ray hit near horizontal face (tri_idx={triangle_index}, normal_3d={normal_3d}). Using fallback: finding closest 2D segment normal.")
             path_2d = self.substrate.path
             vertices = path_2d.vertices
             location_2d = location_3d[:2]
             min_dist_sq = np.inf
             best_normal = None

             if not hasattr(path_2d, 'segments'):
                  print("Error: Path object has no 'segments' attribute for fallback normal calculation.")
                  return np.array([1.0, 0.0]) # Default

             for i0, i1 in path_2d.segments:
                 p0 = vertices[i0]
                 p1 = vertices[i1]
                 seg_vec = p1 - p0
                 seg_len_sq = np.dot(seg_vec, seg_vec)

                 if seg_len_sq < 1e-12: continue # Skip zero-length segments

                 # Project point onto line defined by segment
                 t = np.dot(location_2d - p0, seg_vec) / seg_len_sq
                 t = np.clip(t, 0.0, 1.0) # Clamp projection to segment bounds
                 closest_pt_on_segment = p0 + t * seg_vec
                 dist_sq = np.sum((location_2d - closest_pt_on_segment)**2)

                 if dist_sq < min_dist_sq:
                      min_dist_sq = dist_sq
                      # Calculate perpendicular vector for normal
                      raw_normal = np.array([-seg_vec[1], seg_vec[0]])
                      best_normal = normalize(raw_normal)

             if best_normal is None:
                  print("Error: Could not determine fallback normal.")
                  return np.array([1.0, 0.0]) # Default

             # --- Ensure normal points outwards (heuristic) ---
             # This is complex. A simple check: does normal point towards outside?
             # Test point slightly along normal from intersection point
             # test_pt_out = location_2d + best_normal * 1e-6
             # _, _, is_test_pt_inside = self._get_medium_properties(test_pt_out)
             # if is_test_pt_inside:
             #      best_normal = -best_normal # Flip if it points inwards

             # print(f"Fallback normal calculated: {best_normal}")
             return best_normal
             # -----------------------------------------------------


    # Modified trace method signature and logic
    def trace(self, origin, direction, wavelength_microns, # Added temperature
              max_depth=50, energy_threshold=0.01, DEBUG=False):
        """
        Traces a single 2D ray, calculating accumulated radiance from emission.

        Args:
            origin (array-like): (x, y) starting position (meters).
            direction (array-like): (dx, dy) initial direction vector.
            wavelength_meters (float): Wavelength of the ray in meters.
            temperature_kelvin (float): Baseline temperature for emission calculations (Kelvin).
                                         (Can be overridden by material temperature if available).
            max_depth (int): Maximum number of bounces.
            energy_threshold (float): Stop tracing if ray throughput falls below this fraction.

        Returns:
            tuple: (path_history, accumulated_radiance, final_throughput)
                   path_history (list): List of np.ndarray points [(x,y), ...].
                   accumulated_radiance (float): Sum of emitted spectral radiance contributions [W / (m^2 * sr * m)].
                   final_throughput (float): Remaining throughput fraction of the ray (0 to 1).
        """
        # --- Initialization ---
        current_pos = np.array(origin, dtype=float)
        current_dir = normalize(np.array(direction, dtype=float))
        
        ray_throughput = 1.0 # Renamed from current_energy
        accumulated_radiance = 0.0 # Initialize radiance accumulator
        
        self.current_wavelength_meters = wavelength_microns * 1e-6 # Convert to meters for calculations
        self.wavelength_microns = wavelength_microns # Store for later use

        if np.linalg.norm(current_dir) < 1e-9:
             print("Error: Initial ray direction is zero vector.")
             return [current_pos.copy()], 0.0, 0.0 # path, radiance, throughput

        path_history = [current_pos.copy()]
        if DEBUG: print(f"Starting trace: pos={current_pos}, dir={current_dir}, lambda={self.wavelength_microns:.3g}µm")

        # --- Main Tracing Loop ---
        for bounce in range(max_depth):

            # Check throughput threshold
            if ray_throughput < energy_threshold:
                if DEBUG: print(f"Bounce {bounce+1}: Throughput ({ray_throughput:.3g}) below threshold ({energy_threshold}). Stopping.")
                break

            # Determine properties of the current medium
            key = wavelength_microns
            if key not in self._nkt_cache:
                self._nkt_cache[key] = self._get_medium_properties(current_pos)
            current_n, current_k, is_inside, medium_temp = self._nkt_cache[key]
            
            # _, _, is_inside, medium_temp = self._get_medium_properties(current_pos)  # only temp & “inside” still vary
            # Use provided temperature if material doesn't specify one? Or always use material temp if inside?
            # Let's assume material temp overrides if available and inside.
            # if not is_inside: medium_temperature = temperature_kelvin # Use background temp if outside

            # reuse our arrays
            off = current_pos + current_dir * 1e-9
            self._ray_origins_3d[0,0] = off[0]        
            self._ray_origins_3d[0,1] = off[1]      # z stays zero
            self._ray_dirs_3d[0,0]    = current_dir[0]
            self._ray_dirs_3d[0,1]    = current_dir[1]

            # Find the next intersection
            if self.substrate._intersector is None: print("Error: Intersector not available."); break
            
            try:
                locations_3d, index_ray, index_tri = self.substrate._intersector.intersects_location(    
                    ray_origins=self._ray_origins_3d,
                    ray_directions=self._ray_dirs_3d,
                    multiple_hits=False)
            except Exception as e: print(f"Error during intersection: {e}"); break

            # --- Handle No Intersection (Ray Escapes) ---
            if len(locations_3d) == 0:
                if DEBUG: print(f"Bounce {bounce+1}: Ray escaped - no intersection found.")
                # Optional: Add final segment emission/absorption if escaping into non-vacuum?
                break

            # --- Process Intersection ---
            hit_location_3d = locations_3d[0]
            hit_triangle_idx = index_tri[0]
            hit_location_2d = hit_location_3d[:2]
            distance = np.linalg.norm(hit_location_2d - current_pos)

            # --- Apply Absorption and Emission for the Segment ---
            if distance > 1e-12: # Only process if distance is meaningful

                #  alpha = (4.0 * np.pi * current_k) / self.current_wavelength_meters # Absorption coefficient in m⁻¹
                    alpha = self.substrate.material.get_alpha(self.wavelength_microns) # Convert to microns for lookup
                    # alpha is in cm-1, distance is in m, convert alpha to m-1
                    alpha_m = alpha * 1e2
                    attenuation = np.exp(-alpha_m * distance)
                    emissivity = 1.0 - attenuation

                    # Calculate emitted radiance from this segment
                    # Using the temperature determined for the current medium
                    B_lambda_T = planck_spectral_radiance(wavelength_microns, medium_temp)
                    emitted_radiance_segment = emissivity * B_lambda_T # W / (m^2 * sr * m)

                    # Add segment's emission, weighted by throughput reaching it
                    accumulated_radiance += ray_throughput * emitted_radiance_segment

                    # Update ray throughput by absorption
                    ray_throughput *= attenuation
                # else: medium is non-absorbing (k=0), throughput doesn't change, emissivity is 0.

            # Update position and path history
            if not np.all(np.isfinite(hit_location_2d)):
                 if DEBUG: print(f"Bounce {bounce+1}: Invalid hit location ({hit_location_2d}). Stopping.")
                 break
            current_pos = hit_location_2d.copy()
            path_history.append(current_pos.copy())

            # --- Handle Interaction at Interface ---
            normal_2d = self._get_surface_normal(hit_location_3d, hit_triangle_idx)
            if normal_2d is None or np.linalg.norm(normal_2d) < 1e-9:
                 if DEBUG: print(f"Bounce {bounce+1}: Invalid normal. Stopping trace.")
                 break

            # Ensure normal points against incident ray
            dot_prod = np.dot(current_dir, normal_2d)
            if dot_prod > 1e-9: normal_2d = -normal_2d
            elif abs(dot_prod) < 1e-9 and DEBUG: print(f"Warning: Grazing incidence at bounce {bounce+1}.")

            # Determine properties of the next medium
            test_point = current_pos + normal_2d * 1e-6
            next_n, next_k, next_is_inside, _ = self._get_medium_properties(test_point) # Don't need temp here

            # Calculate reflection probability R
            cos_i = -np.dot(current_dir, normal_2d); cos_i = np.clip(cos_i, 0.0, 1.0)
            R = schlick_reflectance(current_n, next_n, cos_i)
            if np.random.random() < R:
                d0x, d0y = reflect2(current_dir[0], current_dir[1], normal_2d[0], normal_2d[1])
            else:
                ok, rx, ry = refract2(current_dir[0], current_dir[1], normal_2d[0], normal_2d[1], current_n/next_n)
                if not ok:
                    d0x, d0y = reflect2(current_dir[0], current_dir[1], normal_2d[0], normal_2d[1])
                else:
                    d0x, d0y = rx, ry

            current_dir = normalize(np.array([d0x, d0y]))
            if np.linalg.norm(current_dir) < 1e-9 and DEBUG:
                 print(f"Bounce {bounce+1}: Zero direction after interaction. Stopping."); break

        # --- End of Loop ---
        else:
            pass 
            if DEBUG: print(f"Max depth ({max_depth}) reached. Final throughput: {ray_throughput:.3g}")

        # Return path, accumulated radiance, and final throughput
        return path_history, accumulated_radiance, ray_throughput

    def visualize_trace(self, path_history, title="Ray Trace"):
         """
         Visualizes the traced path on top of the substrate geometry using matplotlib.

         Args:
             path_history (list): List of np.ndarray points [(x,y), ...] from trace().
             title (str): Title for the plot.
         """
         if not path_history or len(path_history) < 2:
             print("Warning: Insufficient path history to visualize.")
             # Optionally plot just the substrate?
             fig, ax = plt.subplots(figsize=(10, 8))
             self._plot_substrate(ax) # Helper to plot substrate
             ax.set_title("Substrate Geometry (No Ray Path)")
             plt.show()
             return

         fig, ax = plt.subplots(figsize=(10, 8))

         # Plot substrate boundary
         self._plot_substrate(ax)

         # Plot the ray path
         path_array = np.array(path_history)
         ax.plot(path_array[:, 0], path_array[:, 1], color='red', linestyle='-', marker='.', markersize=5, label='Ray Path', zorder=3)
         # Mark start point
         ax.scatter([path_array[0, 0]], [path_array[0, 1]], color='lime', marker='o', s=70, zorder=5, label='Start', edgecolors='k')
         # Mark end point
         ax.scatter([path_array[-1, 0]], [path_array[-1, 1]], color='black', marker='x', s=70, zorder=5, label='End')


         # Final plot adjustments
         ax.set_xlabel("X coordinate (m)") # Assuming meters based on trace logic
         ax.set_ylabel("Y coordinate (m)")
         ax.set_title(f"{title} ({len(path_history)-1} bounces)")
         ax.set_aspect('equal', adjustable='box')
         ax.legend()
         ax.grid(True, linestyle=':', alpha=0.6)

         plt.tight_layout() # Adjust layout
         plt.show()

    def _plot_substrate(self, ax):
        """Helper function to plot the substrate boundary on given axes."""
        try:
            path = self.substrate.path
            vertices = path.vertices
            plotted_label_boundary = False
            if hasattr(path, 'entities') and path.entities:
                 for entity in path.entities:
                     if len(entity.points) > 1:
                         pts = vertices[entity.points]
                         label = 'Layer Boundary' if not plotted_label_boundary else ""
                         ax.plot(pts[:, 0], pts[:, 1], color='blue', linewidth=1.5, label=label, zorder=2)
                         plotted_label_boundary = True
            elif hasattr(path, 'segments'): # Fallback
                 segments = path.segments
                 for i0, i1 in segments:
                     p0 = vertices[i0]
                     p1 = vertices[i1]
                     label = 'Layer Boundary' if not plotted_label_boundary else ""
                     ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color='blue', linewidth=1.5, label=label, zorder=2)
                     plotted_label_boundary = True

            if not plotted_label_boundary: # Add dummy entry if path was empty/unplottable
                ax.plot([], [], color='blue', linewidth=1.5, label='Layer Boundary')
        except Exception as e:
             print(f"Error plotting substrate boundary: {e}")
             ax.plot([], [], color='blue', linewidth=1.5, label='Layer Boundary (Error)')




import numpy as np
import math
from numba import njit

# Inline Planck function JIT-compiled for nopython mode
@njit
def planck_lambda(wl, T):
    # wl: wavelength in microns, T: temperature in K
    # Physical constants
    c1 = 3.741771e-16  # W·m^2 (first radiation constant)
    c2 = 1.438776e-2   # m·K   (second radiation constant)
    lam_m = wl * 1e-6  # convert microns to meters
    # Planck's law: spectral radiance per unit wavelength
    return (c1 / (lam_m**5)) / (math.exp(c2 / (lam_m * T)) - 1.0)

# Utility normalize for 2D vectors
@njit
def normalize2(dx, dy):
    norm = math.hypot(dx, dy)
    if norm > 0.0:
        return dx / norm, dy / norm
    return 0.0, 0.0


class RayTracer2DOptimized:
    """
    JIT-accelerated 2D ray tracer with Fresnel reflection/refraction.
    """
    def __init__(self, substrate, wls):
        path = substrate.path
        
        self.substrate = substrate
        # Vertex array
        self.verts = np.asarray(path.vertices, dtype=np.float64)
        # Build segment list
        segs = []
        if hasattr(path, 'entities') and path.entities:
            for ent in path.entities:
                pts = ent.points
                for i in range(len(pts) - 1):
                    segs.append((pts[i], pts[i+1]))
        elif hasattr(path, 'segments') and path.segments:
            segs = list(path.segments)
        else:
            raise ValueError("Path2D has no entities or segments.")
        self.segs = np.asarray(segs, dtype=np.int32)
        # Precompute refractive indices and absorption coefficients per integer micron
        

        substrate.material.make_tables(wls)
        self.n_table = np.asarray(substrate.material.n_table, dtype=np.float64)
        self.alpha_table = np.asarray(substrate.material.a_table, dtype=np.float64)
        
        self.wls = wls
        self.surrounding_n = 1.0
        # Material temperature
        self.temp = float(substrate.material.temperature)

    def trace(self, origin, direction, wavelength_microns,
              max_depth=50, energy_threshold=0.01):
        path_x = np.zeros(max_depth+1, dtype=np.float64)
        path_y = np.zeros(max_depth+1, dtype=np.float64)
        dx, dy = normalize2(direction[0], direction[1])
        count, acc_rad, throughput = _ray_trace_jit(
            float(origin[0]), float(origin[1]), dx, dy,
            float(wavelength_microns), self.wls,
            self.n_table, self.surrounding_n,
            self.segs, self.verts,
            self.alpha_table, self.temp,
            max_depth, float(energy_threshold),
            path_x, path_y
        )
        path = [(path_x[i], path_y[i]) for i in range(count)]
        return path, acc_rad, throughput
    
    def visualize_trace(self, path_history, title="Ray Trace"):
         """
         Visualizes the traced path on top of the substrate geometry using matplotlib.

         Args:
             path_history (list): List of np.ndarray points [(x,y), ...] from trace().
             title (str): Title for the plot.
         """
         if not path_history or len(path_history) < 2:
             print("Warning: Insufficient path history to visualize.")
             # Optionally plot just the substrate?
             fig, ax = plt.subplots(figsize=(10, 8))
             self._plot_substrate(ax) # Helper to plot substrate
             ax.set_title("Substrate Geometry (No Ray Path)")
             plt.show()
             return

         fig, ax = plt.subplots(figsize=(10, 8))

         # Plot substrate boundary
         self._plot_substrate(ax)

         # Plot the ray path
         path_array = np.array(path_history)
         ax.plot(path_array[:, 0], path_array[:, 1], color='red', linestyle='-', marker='.', markersize=5, label='Ray Path', zorder=3)
         # Mark start point
         ax.scatter([path_array[0, 0]], [path_array[0, 1]], color='lime', marker='o', s=70, zorder=5, label='Start', edgecolors='k')
         # Mark end point
         ax.scatter([path_array[-1, 0]], [path_array[-1, 1]], color='black', marker='x', s=70, zorder=5, label='End')


         # Final plot adjustments
         ax.set_xlabel("X coordinate (m)") # Assuming meters based on trace logic
         ax.set_ylabel("Y coordinate (m)")
         ax.set_title(f"{title} ({len(path_history)-1} bounces)")
         ax.set_aspect('equal', adjustable='box')
         ax.legend()
         ax.grid(True, linestyle=':', alpha=0.6)

         plt.tight_layout() # Adjust layout
         plt.show()

    def _plot_substrate(self, ax):
        """Helper function to plot the substrate boundary on given axes."""
        try:
            path = self.substrate.path
            vertices = path.vertices
            plotted_label_boundary = False
            if hasattr(path, 'entities') and path.entities:
                 for entity in path.entities:
                     if len(entity.points) > 1:
                         pts = vertices[entity.points]
                         label = 'Layer Boundary' if not plotted_label_boundary else ""
                         ax.plot(pts[:, 0], pts[:, 1], color='blue', linewidth=1.5, label=label, zorder=2)
                         plotted_label_boundary = True
            elif hasattr(path, 'segments'): # Fallback
                 segments = path.segments
                 for i0, i1 in segments:
                     p0 = vertices[i0]
                     p1 = vertices[i1]
                     label = 'Layer Boundary' if not plotted_label_boundary else ""
                     ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color='blue', linewidth=1.5, label=label, zorder=2)
                     plotted_label_boundary = True

            if not plotted_label_boundary: # Add dummy entry if path was empty/unplottable
                ax.plot([], [], color='blue', linewidth=1.5, label='Layer Boundary')
        except Exception as e:
             print(f"Error plotting substrate boundary: {e}")
             ax.plot([], [], color='blue', linewidth=1.5, label='Layer Boundary (Error)')

@njit
def _ray_trace_jit(x, y, dx, dy, wl, wls,
                   n_table, surrounding_n,
                   segs, verts,
                   alpha_table, temp,
                   max_depth, energy_threshold,
                   path_x, path_y):
    """
    Numba JIT core with Fresnel-based reflection/refraction.
    """
    path_x[0] = x
    path_y[0] = y
    throughput = 1.0
    acc_rad = 0.0
    count = 1
    
    # turns wls into numpy array for argwhere
    wls = np.asarray(wls, dtype=np.float64)
    
    idx = np.argwhere(wls == wl)
    # I want the index as an int, not a 1D array
    idx = int(idx[0][0])

    
    for _ in range(max_depth):
        if throughput < energy_threshold:
            break
        # Find closest segment
        min_t = 1e20
        hit_x = hit_y = 0.0
        seg_idx = -1
        for si in range(segs.shape[0]):
            i0, i1 = segs[si]
            x0, y0 = verts[i0]
            x1, y1 = verts[i1]
            vx = x1 - x0; vy = y1 - y0
            denom = dx * vy - dy * vx
            if abs(denom) < 1e-12: continue
            t = ((x0 - x)*vy - (y0 - y)*vx) / denom
            if t <= 1e-9: continue
            ix = x + dx * t; iy = y + dy * t
            u = ((ix - x0)*vx + (iy - y0)*vy) / (vx*vx + vy*vy)
            if u < 0.0 or u > 1.0: continue
            if t < min_t:
                min_t, hit_x, hit_y, seg_idx = t, ix, iy, si
        if seg_idx < 0:
            break
        dist = min_t
        # Absorption & emission
        alpha_cm = alpha_table[idx]
        alpha_m = alpha_cm * 1e2
        att = math.exp(-alpha_m * dist)
        emissivity = 1.0 - att
        emit = emissivity * planck_spectral_radiance(wl, temp)
        acc_rad += throughput * emit
        throughput *= att
        # Move to hit
        x, y = hit_x, hit_y
        path_x[count] = x; path_y[count] = y; count += 1
        # Surface normal
        i0, i1 = segs[seg_idx]
        vx = verts[i1,0] - verts[i0,0]
        vy = verts[i1,1] - verts[i0,1]
        nx = -vy; ny = vx
        nn = math.hypot(nx, ny)
        nx /= nn; ny /= nn
        # Ensure normal opposes ray
        if dx*nx + dy*ny > 0.0:
            nx = -nx; ny = -ny
        # Fresnel reflectance (Schlick)
        cos_i = -(dx*nx + dy*ny)
        n0 = n_table[idx]
        n1 = surrounding_n
        R0 = ((n1 - n0)/(n1 + n0))**2
        R = R0 + (1.0 - R0)*(1.0 - cos_i)**5
        # Probabilistic reflection/refraction
        if np.random.random() < R:
            # Reflect
            dot = dx*nx + dy*ny
            dx = dx - 2.0*dot*nx
            dy = dy - 2.0*dot*ny
        else:
            # Refract via Snell's law
            eta = n0 / n1
            k = 1.0 - eta*eta*(1.0 - cos_i*cos_i)
            if k < 0.0:
                dot = dx*nx + dy*ny
                dx = dx - 2.0*dot*nx
                dy = dy - 2.0*dot*ny
            else:
                sqrt_k = math.sqrt(k)
                dx = eta*dx + (eta*cos_i - sqrt_k)*nx
                dy = eta*dy + (eta*cos_i - sqrt_k)*ny
    return count, acc_rad, throughput

