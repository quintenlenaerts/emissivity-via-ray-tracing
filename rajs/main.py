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

# from graphx import GraphX

width     = 30000.0
thickness = 50.0

# Create flat top and bottom borders
# InterfaceBorder.BorderType.SINE_SETTINGS(10, 0.5, 20000)
# InterfaceBorder.BorderType.FILE_SETTINGS("interface1_profile.csv")
int_top = InterfaceBorder(InterfaceBorder.BorderType.PERLIN, width, "TopBorder")
int_top.move_up(thickness)
int_top.move_right(-0.5 * width)

int_bot = InterfaceBorder(InterfaceBorder.BorderType.LINE, width, "BottomBorder")
int_bot.move_right(-0.5 * width)

temperature      = 800.0

# temp_border = InterfaceBorder(InterfaceBorder.BorderType.PERLIN, width, "TempBorder")
# temp_border.move_up(thickness)
# temp_border.move_right(-width/2)

# Materials
mat_olive = Material("Olive", "./lab-olive.csv", temperature)
mat_air = Material("Air", "", temperature, Material.MateterialTypes.SINGLE_NK)
mat_air.set_single_nk(1, 0)

# Build Interface stack: Air above, Material in middle, Air below
iface = Interface()
iface.AddLayer(None,    int_top, mat_air)
iface.AddLayer(int_top, int_bot, mat_olive)  # set your n,k
iface.AddLayer(int_bot, None,    mat_air)
iface.ConnectBorders()
iface.build_trimesh()

num_wavelengths = 200
wavelengths      = np.linspace(0.3, 2.4, num_wavelengths)  # in microns
wavelengths_meters = wavelengths * 1e-6

N_rays_per_wl    = 5000
max_bounces      = 50

# GraphX.set_size(-width/2*1.05, width/2*1.05, 0, thickness*1.05)
# GraphX.draw_interface_border(int_top, 'red')
# GraphX.draw_interface_border(int_bot, 'red')
# GraphX.draw_interface_border(temp_border, 'black')

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


if __name__ == "__main__":
    print(f"Starting simulation: {num_wavelengths} wavelengths, {N_rays_per_wl} rays per wavelength...")

    pool_args = [
        (wl, temperature, iface, -width/2, width/2, 0, thickness, N_rays_per_wl)
        for wl in wavelengths_meters
    ]

    with ProcessPoolExecutor(max_workers=6) as executor:
        results_iter = executor.map(sim_wl, pool_args)
        results_radiance = list(
            tqdm(
                results_iter,
                total=len(pool_args),
                desc="Simulating wavelengths",
                unit="λ"
            )
        )
    results_radiance = np.array(results_radiance)
    emissivity = results_radiance / black_body_radiance

    # emis_analytic = []
    # for wl in wavelengths:
    #     alpha_cm = mat_olive.get_alpha(wl)    # cm⁻¹
    #     alpha_m  = alpha_cm*1e2               # m⁻¹
    #     L        = thickness*1e-6             # interface thickness in m (500 µm → 5e-4 m)
    #     eps_analytic = 1 - np.exp(-alpha_m * L)
    #     emis_analytic.append(eps_analytic)


    print(f"Emissivity: {emissivity}")
    import pandas as pd
    results_df = pd.DataFrame({
        'Wavelength (microns)': wavelengths,
        'Average Spectral Radiance (W/m^2/sr/m)': results_radiance,
        'Black Body Radiance (W/m^2/sr/m)': black_body_radiance,
        'Emissivity': emissivity
    })
    results_df.to_csv("{}microns-lab-olive-{}wls-{}raysperwl.csv".format(round(thickness),num_wavelengths, N_rays_per_wl), index=False, sep = "\t")

    # GraphX.show()

    plt.figure(figsize=(10, 6))
    plt.plot(wavelengths, emissivity, label="Emis")
    # plt.plot(wavelengths, emis_analytic, label="Emis Analytic")
    plt.legend()
    plt.xlabel("Wavelength (microns)")
    # plt.ylabel("Average Spectral Radiance (W / m^2 / sr / m)")
    plt.ylabel("Emissivity")
    plt.title(f"Simulated Radiance Spectrum ({N_rays_per_wl} rays/wl)")
    plt.grid(True, linestyle=':')
    # plt.yscale('log')
    plt.show()
    