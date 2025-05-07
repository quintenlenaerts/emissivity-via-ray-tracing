import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from Interface import InterfaceBorder, Interface
from Material import Material
from Vec2 import Vec2
from LightRay import LightRay
from optics import planck_lambda

from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor


width     = 30000.0
thickness = 500
# thickness_si = 700.0
# thickness_sio2 = 0.01
temperature      = 800

# thickness = thickness_sio2 + thickness_si

InterfaceBorder.BorderType.PERLIN_SETTINGS(10, 10, 1000)

int_top = InterfaceBorder(InterfaceBorder.BorderType.PERLIN, width, "TopBorder")
int_top.move_up(thickness)
int_top.move_right(-0.5 * width)

# int_mid = InterfaceBorder(InterfaceBorder.BorderType.PERLIN, width, "MiddleBorder")
# int_mid.move_up(thickness_si)
# int_mid.move_right(-0.5 * width)

int_bot = InterfaceBorder(InterfaceBorder.BorderType.PERLIN, width, "BottomBorder")
int_bot.move_right(-0.5 * width)

# Materials
mat_olive = Material("Olive", "./lab-olive.csv", temperature)
# mat_olive.save_nk_data("labolive_correct.csv")

Ms=["./BrownishYellow.csv", "./GreenYellow.csv","./VioletGray.csv"]

for m in Ms:
    _m = Material("Shit", m, temperature)
    _m.save_nk_data(m)

mat_air = Material("Air", "", temperature, Material.MateterialTypes.SINGLE_NK)
mat_air.set_single_nk(1, 0)

mat_temp = Material("Temp", "", temperature, Material.MateterialTypes.SINGLE_NK)
mat_temp.set_single_nk(1.5, 0.1)



mat_si = Material("Silicon", "./NK/correct_si.csv", temperature, Material.MateterialTypes.NK_DATA)
mat_sio2 = Material("Silicono2", "./NK/correct_sio2.csv", temperature, Material.MateterialTypes.NK_DATA)
mat_plat = Material("Plat", "./NK/correct_plat_1.csv", temperature, Material.MateterialTypes.NK_DATA)

# Build Interface stack: Air above, Material in middle, Air below
iface = Interface()
iface.AddLayer(None,    int_top, mat_air)
iface.AddLayer(int_top,    int_bot, mat_temp)
# iface.AddLayer(int_mid, int_bot, mat_si)  # set your n,k
iface.AddLayer(int_bot, None,    mat_air)
iface.ConnectBorders()
iface.build_trimesh()

num_wavelengths = 200
wavelengths      = np.linspace(0.20, 3.45, num_wavelengths)  # in microns
wavelengths_meters = wavelengths * 1e-6

N_rays_per_wl    = 5000
max_bounces      = 50

def sim_wl(args):
    wavelength_meter, temp, _interface, x_min, x_max, y_min, y_max, N = args
    s = 0.0 # the calc radiance
    for _ in range(N):
        # determine start pos
        _x = np.random.uniform(x_min, x_max)
        _y = np.random.uniform(y_min, y_max)

        # random direction
        _dir = np.random.uniform(0, 360)
        
        pos_history, throughput, n_radiance = _interface.TraceOneRay(
            Vec2(_x, _y), _dir, wavelength_meter, temp
        )

        s += n_radiance 

    return s/N 


results_radiance = np.zeros_like(wavelengths)
black_body_radiance = [planck_lambda(wl, temperature) for wl in wavelengths]


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

#     print(f"Emissivity: {emissivity}")
#     import pandas as pd
#     results_df = pd.DataFrame({
#         'Wavelength (microns)': wavelengths,
#         'Average Spectral Radiance (W/m^2/sr/m)': results_radiance,
#         'Black Body Radiance (W/m^2/sr/m)': black_body_radiance,
#         'Emissivity': emissivity
#     })
#     # results_df.to_csv("{}microns-platinum-{}wls-{}raysperwl-{}kelvin.csv".format(round(thickness),num_wavelengths, N_rays_per_wl,temperature), index=False, sep = "\t")

#     plt.figure(figsize=(10, 6))
#     plt.plot(wavelengths, emissivity, label="Emis")
#     # plt.plot(wavelengths, emis_analytic, label="Emis Analytic")
#     plt.legend()
#     plt.xlabel("Wavelength (microns)")
#     # plt.ylabel("Average Spectral Radiance (W / m^2 / sr / m)")
#     plt.ylabel("Emissivity")
#     plt.title(f"Simulated Radiance Spectrum ({N_rays_per_wl} rays/wl)")
#     plt.grid(True, linestyle=':')
#     # plt.yscale('log')
#     plt.show()


    
  