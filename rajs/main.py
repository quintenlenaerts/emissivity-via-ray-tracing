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
thickness = 500.0

# Create flat top and bottom borders
InterfaceBorder.BorderType.SINE_SETTINGS(0.5, 2, 600)
int_top = InterfaceBorder(InterfaceBorder.BorderType.SINE, width, "TopBorder")
int_top.move_up(thickness)
int_top.move_right(-0.5 * width)

int_bot = InterfaceBorder(InterfaceBorder.BorderType.LINE, width, "BottomBorder")
int_bot.move_right(-0.5 * width)

temperature      = 800.0

# Materials
mat_olive = Material("Olive", "./lab-olive.csv", temperature)

# Build Interface stack: Air above, Material in middle, Air below
iface = Interface()
iface.AddLayer(None,    int_top, Material("Air", "", temperature))
iface.AddLayer(int_top, int_bot, mat_olive)  # set your n,k
iface.AddLayer(int_bot, None,    Material("Air",   "", temperature))
iface.ConnectBorders()

num_wavelengths = 12
wavelengths      = np.linspace(0.3, 2.4, num_wavelengths)  # in microns
wavelengths_meters = wavelengths * 1e-6

N_rays_per_wl    = 200
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
        
        throughput, n_radiance = _interface.TraceOneRay(
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

    print(f"Emissivity: {emissivity}")