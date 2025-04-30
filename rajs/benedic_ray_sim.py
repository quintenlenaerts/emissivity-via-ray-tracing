import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

from Interface import InterfaceBorder, Interface
from Material import Material
from Vec2 import Vec2
from LightRay import LightRay
from optics import planck_lambda

# ─────────────────────────────────────────────────────────────────────────────
# Simulation parameters
# ─────────────────────────────────────────────────────────────────────────────
width               = 30000.0       # µm
thickness_substrate = 400.0         # µm (fixed)
temperature         = 1170.0        # K
wavelength          = 0.91e-6        # m (single λ)
N_rays              = 2000       # per thickness


# array of film thicknesses to sweep (µm)
thicknesses = np.linspace(20, 30, 50)

# Materials (n,k)
mat_air = Material("Air", "", temperature, Material.MateterialTypes.SINGLE_NK)
mat_air.set_single_nk(1.0, 0.0)

mat_diamond = Material("Dia", "", temperature, Material.MateterialTypes.SINGLE_NK)
mat_diamond.set_single_nk(2.4, 0.072)

mat_substrate = Material("Si", "", temperature, Material.MateterialTypes.SINGLE_NK)
mat_substrate.set_single_nk(3.65, 0.0073)

# tiny sine perturbation so we get side‐wall collisions
InterfaceBorder.BorderType.SINE_SETTINGS(0.01, 10, 1000)

# ─────────────────────────────────────────────────────────────────────────────
# Single‐ray Monte Carlo for one λ & one interface
# ─────────────────────────────────────────────────────────────────────────────
def sim_once(iface, film_thickness):
    """Trace N_rays at fixed λ through this iface → return average emitted radiance."""
    s = 0.0
    # E_sum = 0.0
    
    for _ in range(N_rays):
        x = np.random.uniform(-width/2, width/2)
        y = np.random.uniform(0, thickness_substrate + film_thickness)
        _, _, emitted = iface.TraceOneRay(Vec2(x, y),
                                         np.random.uniform(0, 360),
                                         wavelength,
                                         temperature)
        s += emitted
    return 0, s / N_rays


def sim_phase(iface: Interface, film_thickness):

    E_sum = 0 + 0j
    accu_rad = 0

    for _ in range(N_rays):
        
        x = np.random.uniform(-width/2, width/2)
        y = np.random.uniform(0, thickness_substrate + film_thickness)

        e, rad = iface.TraceOneRayWithPhase(
            Vec2(x,y),
            np.random.uniform(0,360),
            wavelength,
            temperature
        )

        E_sum += e
        accu_rad += rad / N_rays

    # returning R_estimated
    return abs(E_sum / N_rays)**2, accu_rad

# prepare result array
emissivities = np.zeros_like(thicknesses)
emissivities_rad = np.zeros_like(thicknesses)
r_ests = np.zeros_like(thicknesses)

# ─────────────────────────────────────────────────────────────────────────────
# Sweep over thicknesses (parallelized)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    with ProcessPoolExecutor(max_workers=6) as pool:
        futures = []
        for t in thicknesses:
            # rebuild interface for this thickness
            int_top = InterfaceBorder(InterfaceBorder.BorderType.SINE, width, "Air-Dia")
            int_top.move_up(t + thickness_substrate)
            int_top.move_right(-0.5 * width)

            int_mid = InterfaceBorder(InterfaceBorder.BorderType.SINE, width, "Dia-Si")
            int_mid.move_up(thickness_substrate)
            int_mid.move_right(-0.5 * width)

            int_bot = InterfaceBorder(InterfaceBorder.BorderType.SINE, width, "Si-Air")
            int_bot.move_right(-0.5 * width)

            iface = Interface()
            iface.AddLayer(None,    int_top,      mat_air)
            iface.AddLayer(int_top, int_mid,      mat_diamond)
            iface.AddLayer(int_mid, int_bot,      mat_substrate)
            iface.AddLayer(int_bot, None,         mat_air)
            iface.ConnectBorders()
            iface.build_trimesh()

            futures.append(pool.submit(sim_phase, iface, t))

        for i, f in enumerate(tqdm(futures, desc="Thickness sweep")):
            r_est, rad = f.result()
            B = planck_lambda(wavelength*1e6, temperature)
            r_ests[i] = r_est
            emissivities[i] = rad / B
            emissivities_rad[i] = 1 - r_est

    # ─────────────────────────────────────────────────────────────────────────────
    # Save to CSV
    # ─────────────────────────────────────────────────────────────────────────────
    # df = pd.DataFrame({
    #     'Thickness (µm)': thicknesses,
    #     'Emissivity':     emissivities
    # })
    # df.to_csv("{}raysperthickness_{}kelvin_{}substratethic.csv".format(N_rays, round(temperature), round(thickness_substrate)), index=False)
    # print("Saved emissivity_vs_thickness.csv")

    # ─────────────────────────────────────────────────────────────────────────────
    # Plot emissivity vs. film thickness
    # ─────────────────────────────────────────────────────────────────────────────
    # plt.figure(figsize=(8,5))
    # plt.plot(thicknesses, emissivities, marker='o')
    # plt.xlabel("Diamond Film Thickness (µm)")
    # plt.ylabel(f"Emissivity @ {wavelength*1e6:.2f} µm")
    # plt.title("Emissivity vs. Film Thickness")
    # plt.grid(True, linestyle=':')
    # plt.show()


        # ─────────────────────────────────────────────────────────────────────────────
    # Plot emis/R vs. film thickness
    # ─────────────────────────────────────────────────────────────────────────────
    plt.figure(figsize=(8,5))

    print(emissivities)

    plt.plot(thicknesses, emissivities, marker='o', label="Emis")
    plt.plot(thicknesses, r_ests, marker='o', label="R")
    plt.plot(thicknesses, emissivities_rad, marker='o', label="1-R")
    plt.legend()
    plt.xlabel("Diamond Film Thickness (µm)")
    plt.ylabel(f"Emissivity @ {wavelength*1e6:.2f} µm")
    plt.title("Emissivity vs. Film Thickness")
    plt.grid(True, linestyle=':')
    plt.show()


    