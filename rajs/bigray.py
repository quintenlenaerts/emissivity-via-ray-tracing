import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from Interface import InterfaceBorder, Interface
from Material import Material
from Vec2 import Vec2
from LightRay import LightRay
from optics import planck_lambda
import copy
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

from graphx import GraphX

class RTSetup:
    """
        Ray Tracing Setup class, this will contain everything about the interface such as thicknesses, materials etc..
    """

    def __init__(self, setup_name : str, width : float = 100):
        """
            Creates a new RayTracingSetup class.

            Args:
                setup_name (str)
                width (float) : width of the setup in microns
        
        """
        self.name = setup_name

        # microns => width of all interface borders/layers will be the same; thickness can be adjusted
        self.width = width

        # is represented as : [name (str), topborder (interfaceborder), material (material)]
        self._layer_data = []

        self._thicknesses = []

        # has to be reset to None after changing the materials and or thicknesses!
        self.interface = None
    
    def SetWidth(self, new_width : float):
        self.width = new_width
    
    def AddLayer(self, layer_name: str, topborder: InterfaceBorder, material: Material):
        """
        Adds a new layer to the setup.
        :param layer_name: Identifier for the layer.
        :param topborder: InterfaceBorder at the top of this layer (None for the first layer).
        :param material: Material of this layer.
        """
        # if topborder is not None:
        #     # instantiate a new border with the same type, width, and name
        #     border_copy = InterfaceBorder(topborder._type, topborder._width, topborder._name)
        #     border_copy._mesh = topborder._mesh
        #     border_copy._mesh_rays = None
        # else:
        #     border_copy = None
        self._layer_data.append((layer_name, topborder, material))

    def SetThicknesses(self, thickness_array: list[float]):
        """
        Specify thicknesses for the stack. Can supply either:
          - A list of length (num_layers - 1) to define each gap in order
          - A list of length (num_layers - 2) to define only interior layer thicknesses (first/last assumed zero)
        """
        n = len(self._layer_data)
        m = len(thickness_array)
        if m not in {n - 2, n - 1}:
            raise ValueError(
                f"Thickness array length must be num_layers-1 ({n-1}) or num_layers-2 ({n-2}), got {m}"
            )
        self._thicknesses = thickness_array

        # resetting
        self.interface = None

    def create_interface(self, ext_height : float) -> Interface:
        """
        Builds an Interface with all borders positioned vertically by the specified thicknesses.
        """

        if self.interface != None:
            return self.interface

        if not self._layer_data:
            raise ValueError("No layers defined. Use AddLayer() first.")
        n = len(self._layer_data)
        m = len(self._thicknesses)
        if m == 0:
            raise ValueError("Thicknesses not set. Call SetThicknesses().")
        # Compute vertical offsets per topborder index
        if m == n - 1:
            # full gaps provided: cumulative sum
            heights = [0.0]
            for t in self._thicknesses:
                heights.append(heights[-1] + t)
        else:
            # only interior provided: assume first/last = 0
            heights = [0.0] + self._thicknesses + [0.0]
        if len(heights) != n:
            raise RuntimeError("Internal error: heights length mismatch.")

        # Position borders
        for idx, (_, topborder, _) in enumerate(self._layer_data):
            if topborder:
                topborder.move_right(-self.width / 2)
                topborder.move_up(heights[idx])

        # Build Interface
        iface = Interface()
        for idx, (_, topborder, material) in enumerate(self._layer_data):
            # next layer's topborder is this layer's bottomborder
            if idx < n - 1:
                _, next_border, _ = self._layer_data[idx + 1]
                bottomborder = next_border
            else:
                bottomborder = None
            iface.AddLayer(topborder, bottomborder, material)
        iface.ConnectBorders()
        iface.build_trimesh(extrusion_height=ext_height)
        self.interface = iface
        return iface

    def show(self, margin: float = 0.05, grid: bool = True):
        """
        Render the layer stack using GraphX. Automatically sets axes and draws each interface border.
        :param margin: Fractional padding around extents (default 5%)
        :param grid: Whether to enable grid (default True)
        """

        # has to be called or else
        _ = self.create_interface(1e-6)

        # Calculate total stack height as sum of all defined thicknesses
        total_height = sum(self._thicknesses) if self._thicknesses else 0.0

        # Set plot limits with 5% margin by default
        x_min = -self.width / 2 * (1 + margin)
        x_max =  self.width / 2 * (1 + margin)
        y_min = 0.0
        y_max = total_height * (1 + margin)
        GraphX.set_size(x_min, x_max, y_min, y_max)
        if grid:
            GraphX.grid_on()

        # Draw each moved border (now correctly positioned)
        for _, topborder, _ in self._layer_data:
            if topborder:
                GraphX.draw_interface_border(topborder, color='black')

        # Show
        GraphX.show()

# Params
width            = 30000.0
thickness        = 500.0
temperature      = 800.0

# Borders
InterfaceBorder.BorderType.FILE_SETTINGS("interface1_profile.csv")
int_top = InterfaceBorder(InterfaceBorder.BorderType.PERLIN, width, "TopBorder")
int_top2 = InterfaceBorder(InterfaceBorder.BorderType.LINE, width, "TopBorder2")
int_top3 = InterfaceBorder(InterfaceBorder.BorderType.ZIGZAG, width, "TopBorder3")
int_top4 = InterfaceBorder(InterfaceBorder.BorderType.SINE, width, "TopBorder4")
int_bot = InterfaceBorder(InterfaceBorder.BorderType.LINE, width, "BottomBorder")

# Materials
mat_olive = Material("Olive", "./lab-olive.csv", temperature)
mat_olive2 = Material("Olive2", "./lab-olive.csv", temperature)
mat_air = Material("Air", "", temperature, Material.MateterialTypes.SINGLE_NK)
mat_air.set_single_nk(1, 0)

# RTSetep
lab_olive = RTSetup("Lab Olive", width)
lab_olive.AddLayer("Air", None, mat_air)
lab_olive.AddLayer("Olive", int_top, mat_olive)
lab_olive.AddLayer("Olive", int_top2, mat_olive)
lab_olive.AddLayer("Olive", int_top3, mat_olive)
lab_olive.AddLayer("Olive", int_top4, mat_olive)
lab_olive.AddLayer("Air", int_bot, mat_air)

# lab_olive.SetThicknesses([500,400,300,200])
lab_olive.SetThicknesses([500,45,200, 20])



lab_olive.show()


# def sim_wl(args):
#     wavelength_meter, temp, _interface, x_min, x_max, y_min, y_max, N = args
#     s = 0.0 # the calc radiance
#     for _ in range(N):
#         # determine start pos
#         _x = np.random.uniform(x_min, x_max)
#         _y = np.random.uniform(y_min, y_max)

#         # random direction
#         _dir = np.random.uniform(0, 360)
        
#         pos_history, throughput, n_radiance = _interface.TraceOneRay(
#             Vec2(_x, _y), _dir, wavelength_meter, temp
#         )

#         s += n_radiance 

#     return s/N 


# results_radiance = np.zeros_like(wavelengths)
# black_body_radiance = [planck_lambda(wl, temperature) for wl in wavelengths]


# if __name__ == "__main__":
#     print(f"Starting simulation: {num_wavelengths} wavelengths, {N_rays_per_wl} rays per wavelength...")

#     pool_args = [
#         (wl, temperature, iface, -width/2, width/2, 0, thickness, N_rays_per_wl)
#         for wl in wavelengths_meters
#     ]

#     with ProcessPoolExecutor(max_workers=6) as executor:
#         results_iter = executor.map(sim_wl, pool_args)
#         results_radiance = list(
#             tqdm(
#                 results_iter,
#                 total=len(pool_args),
#                 desc="Simulating wavelengths",
#                 unit="λ"
#             )
#         )
#     results_radiance = np.array(results_radiance)
#     emissivity = results_radiance / black_body_radiance

#     emis_analytic = []
#     for wl in wavelengths:
#         alpha_cm = mat_olive.get_alpha(wl)    # cm⁻¹
#         alpha_m  = alpha_cm*1e2               # m⁻¹
#         L        = thickness*1e-6             # interface thickness in m (500 µm → 5e-4 m)
#         eps_analytic = 1 - np.exp(-alpha_m * L)
#         emis_analytic.append(eps_analytic)


#     print(f"Emissivity: {emissivity}")

#     # GraphX.show()

#     plt.figure(figsize=(10, 6))
#     plt.plot(wavelengths, emissivity, label="Emis")
#     plt.plot(wavelengths, emis_analytic, label="Emis Analytic")
#     plt.legend()
#     plt.xlabel("Wavelength (microns)")
#     # plt.ylabel("Average Spectral Radiance (W / m^2 / sr / m)")
#     plt.ylabel("Emissivity")
#     plt.title(f"Simulated Radiance Spectrum ({N_rays_per_wl} rays/wl)")
#     plt.grid(True, linestyle=':')
#     # plt.yscale('log')
#     plt.show()
    