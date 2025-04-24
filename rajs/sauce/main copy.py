import numpy as np
import matplotlib.pyplot as plt
from perlin_noise import PerlinNoise
from tqdm import tqdm  # <-- Import tqdm

from tracers.tracer_2D import RayTracer2D, RayTracer2DOptimized
from core.layer import SingleLayerSubstrate
from core.material import Material
from core.ray import Ray

import pandas as pd

from core.core_optical_functions import planck_spectral_radiance

from concurrent.futures import ProcessPoolExecutor


def _simulate_wl(args):
    wl_um, tracer, x_min, x_max, y_min, y_max, N = args
    s = 0.0
    for _ in range(N):
        ray = Ray(x_min, x_max, y_min, y_max, wl_um)
        _, radiance, _ = tracer.trace(ray.origin, ray.direction, wl_um)
        s += radiance
    return s / N

if __name__ == "__main__":
    IaB_Olive = Material("Diamond", "gaussian_profile.csv", temperature=800) # Example material

    thickness = 500e-6

    x_min = -1.5e-2
    x_max = 1.5e-2

    y_min = 0
    y_max = 500e-6

    # Define x-values for our synthetic rough interfaces.
    xs = np.linspace(x_min, x_max, 5000)

    # Top interface (Interface 1): Base height 0 with small roughness.
    noise1d = PerlinNoise(octaves=96, seed=42)
    amplitude = 10e-6
    scale = 100
    ys_int1 = [y_max + amplitude * noise1d([scale*x]) for x in xs]
    top_points = [(x, y) for x, y in zip(xs, ys_int1)]

    substrate_geom = SingleLayerSubstrate(top_points, thickness, material = IaB_Olive)

    # 2. Create the RayTracer2D instance
    

    wl_min_microns = 0.3
    wl_max_microns = 2.4
    num_wavelengths = 50 # Number of points in the spectrum
    N_rays_per_wl = 20000  # Number of rays to average per wavelength bin

    # Wavelength array (in MICRONS for definition and plotting)
    wavelengths_microns = np.linspace(wl_min_microns, wl_max_microns, num_wavelengths)
    tracer = RayTracer2DOptimized(substrate_geom, wavelengths_microns)
    # Wavelength array (in METERS for calculations)
    wavelengths_meters = wavelengths_microns * 1e-6

    # Array to store results (average spectral radiance for each wavelength)
    results_radiance = np.zeros_like(wavelengths_microns)
    black_body_radiance = [planck_spectral_radiance(wl, 800) for wl in wavelengths_microns]
    

    print(f"Starting simulation: {num_wavelengths} wavelengths, {N_rays_per_wl} rays per wavelength...")

    # 4. Main Loop - Iterate through Wavelengths using tqdm for progress bar
    #    - Wrap `enumerate(wavelengths_meters)` with `tqdm()`
    #    - `total=num_wavelengths` tells tqdm the total number of iterations for accurate percentage
    #    - `desc="..."` adds a descriptive label to the bar
    pool_args = [
        (wl, tracer, x_min, x_max, y_min, y_max, N_rays_per_wl)
        for wl in wavelengths_microns
    ]

    with ProcessPoolExecutor(max_workers=6) as executor:
        # executor.map returns a generator of results
        results_iter = executor.map(_simulate_wl, pool_args)
        # wrap it in tqdm to get a progress bar
        results_radiance = list(
            tqdm(
                results_iter,
                total=len(pool_args),
                desc="Simulating wavelengths",
                unit="λ"
            )
        )

    # results_radiance is now a list of averaged radiances per λ
    # you can convert it to an array if you like:
    results_radiance = np.array(results_radiance)
        
       
    emissivity = results_radiance / black_body_radiance
    print(f"Emissivity: {emissivity}")
    #  I want to export the results to a file:  
    results_df = pd.DataFrame({
        'Wavelength (microns)': wavelengths_microns,
        'Average Spectral Radiance (W/m^2/sr/m)': results_radiance,
        'Black Body Radiance (W/m^2/sr/m)': black_body_radiance,
        'Emissivity': emissivity
    })
    results_df.to_csv("gaussian_profile-results.csv", index=False, sep = "\t")
    
    

    print("Simulation finished.") # tqdm automatically leaves the finished bar

    # 5. Plot the Results
    plt.figure(figsize=(10, 6))
    plt.plot(wavelengths_microns, emissivity)
    plt.xlabel("Wavelength (microns)")
    plt.ylabel("Average Spectral Radiance (W / m^2 / sr / m)")
    plt.title(f"Simulated Radiance Spectrum ({N_rays_per_wl} rays/wl)")
    plt.grid(True, linestyle=':')
    # plt.yscale('log')
    plt.show()

    # Optional: Visualize the path of the *last* ray traced
    # if 'path_history' in locals():
    #     print("\nVisualizing path of the last traced ray...")
    #     tracer.visualize_trace(path_history, title=f"Last Ray Path (Lambda={wl_um:.1f}um)")