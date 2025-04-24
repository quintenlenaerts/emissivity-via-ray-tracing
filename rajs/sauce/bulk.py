import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from Interface import InterfaceBorder, Interface
from Material import Material
from Vec2 import Vec2
from LightRay import LightRay
from optics import planck_spectral_radiance

from graphx import GraphX

# --- 1) Define single-layer geometry using your own Interface code ---
# Layer width and thickness (in same units as Vec2 coords, e.g. microns)
width     = 600.0
thickness = 500.0  # layer thickness

# Create flat top and bottom borders
InterfaceBorder.BorderType.SINE_SETTINGS(0.5, 2, 600)
int_top = InterfaceBorder(InterfaceBorder.BorderType.SINE, width, "TopBorder")
int_top.move_up(thickness)
int_top.move_right(-0.5 * width)

int_bot = InterfaceBorder(InterfaceBorder.BorderType.LINE, width, "BottomBorder")
int_bot.move_right(-0.5 * width)

# Materials
mat_olive = Material("Olive")
mat_olive.ReadFromFile("NK/correct_olive.csv")


# Build Interface stack: Air above, Material in middle, Air below
iface = Interface()
iface.AddLayer(None,    int_top, Material("Air",   1.0, 0.0))
iface.AddLayer(int_top, int_bot, mat_olive)  # set your n,k
iface.AddLayer(int_bot, None,    Material("Air",   1.0, 0.0))
iface.ConnectBorders()

# GraphX.set_size(-width/2 - 10, width/2 + 10, -10, thickness + 10)
# GraphX.draw_interface_border(int_top, color='red')
# GraphX.draw_interface_border(int_bot, color='red')
# GraphX.show()

# --- 2) Single-ray trace function (bulk emission Monte Carlo) ---

def trace_one_ray(ray, interface, max_bounces=1000):
    """
    Trace a ray through the stack, accumulating bulk emission along each segment.
    """
    rays = [ray]
    for _ in range(max_bounces):
        new_rays = []
        for r in rays:
            children = r.Send(interface) or []
            for c in children:
                if c.throughput < 1e-3:
                    continue
                new_rays.append(c)
        if not new_rays:
            break
        rays = new_rays
    # sum all segment emissions
    return sum(r.emitted_radiance for r in rays)

# --- 3) Monte Carlo bulk-emission simulator ---

# def simulate_bulk_emission(wavelengths, N_rays, interface, layer_width, layer_thickness, temperature, max_bounces=10):
#     """
#     Monte Carlo integration of exitance (M) via bulk emission in a single layer.
#     Returns spectral emissivity ε(λ).
#     """
#     emissivities = []
#     half_w = 0.5 * layer_width
#     for wl in tqdm(wavelengths, desc="λ", unit="λ"):
#         acc_weighted = 0.0
#         for _ in tqdm(range(N_rays), desc=f"rays@{wl:.2f}", unit="ray", leave=False):
#             # random origin inside layer volume
#             x0 = np.random.uniform(-half_w, half_w)
#             y0 = np.random.uniform(0.0, layer_thickness)
#             # uniform hemisphere sampling (cosine-weighted)
#             u = np.random.rand()
#             theta = np.arccos(np.sqrt(u))        # polar angle from vertical
#             phi = np.random.rand() * 2*np.pi
#             # convert to direction vector
#             dx = np.sin(theta) * np.cos(phi)
#             dy = -np.cos(theta)  # downward hemisphere => negative y
#             angle_rad = np.arctan2(dy, dx)
#             ray = LightRay(Vec2(x0, y0), np.degrees(angle_rad))
#             ray.wavelength  = wl
#             ray.temperature = temperature
#             # trace ray and get emitted spectral radiance
#             L = trace_one_ray(ray, interface, max_bounces)
#             # weight by cos(theta) for exitance (cos(theta)= -dy)
#             cos_th = -dy
#             acc_weighted += L * cos_th
#         # compute average exitance M = (2/N) * sum(L_i * cos_th)
#         M_avg = 2.0 * acc_weighted / N_rays
#         # blackbody exitance M_bb = pi * B_lambda
#         B_bb = planck_spectral_radiance(wl * 1e-6, temperature)
#         M_bb = np.pi * B_bb
#         eps = M_avg / M_bb if M_bb > 0 else 0.0
#         emissivities.append(eps)
#     return np.array(emissivities)

def simulate_bulk_emission(wavelengths, N_rays, interface, layer_width, layer_thickness, temperature, max_bounces=10):
    """
    Monte Carlo bulk-emission spectral emissivity via radiance averaging.
    """
    emissivities = []
    half_w = 0.5 * layer_width
    for wl in tqdm(wavelengths, desc="λ", unit="λ"):
        acc_L = 0.0
        for _ in tqdm(range(N_rays), desc=f"rays@{wl:.2f}", unit="ray", leave=False):
            # random origin inside layer volume
            x0 = np.random.uniform(-half_w, half_w)
            y0 = np.random.uniform(0.0, layer_thickness)
            # uniform downward direction (cosine-weighted optional)
            u = np.random.rand()
            theta = np.arccos(1 - u)    # optional: uniform over downward hemisphere
            phi = np.random.rand() * 2*np.pi
            dx = np.sin(theta) * np.cos(phi)
            dy = -abs(np.cos(theta))    # ensure downward
            angle_rad = np.arctan2(dy, dx)
            ray = LightRay(Vec2(x0, y0), np.degrees(angle_rad))
            ray.wavelength  = wl
            ray.temperature = temperature
            # trace ray to get spectral radiance
            L = trace_one_ray(ray, interface, max_bounces)
            acc_L += L
        # average radiance
        L_avg = acc_L / N_rays
        # blackbody spectral radiance
        B_bb  = planck_spectral_radiance(wl * 1e-6, temperature)
        # emissivity = L_avg / B_bb
        eps = L_avg / B_bb if B_bb > 0 else 0.0
        emissivities.append(eps)
    return np.array(emissivities)

# --- 4) Run simulation and plot results ---
wavelengths      = np.linspace(0.3, 2.4, 1)  # in microns
wavelengths = np.array([2.4])
N_rays_per_wl    = 10
temperature      = 200.0
max_bounces      = 700

eps = simulate_bulk_emission(wavelengths, N_rays_per_wl,
                              iface, width, thickness,
                              temperature, max_bounces)

# Print emissivities
print("Emissivity(λ):")
for wl,e in zip(wavelengths, eps):
    print(f"{wl:.2f} µm -> {e:.3f}")

# Plot
plt.figure(figsize=(8,5))
plt.plot(wavelengths, eps, '-o')
plt.xlabel('Wavelength (µm)')
plt.ylabel('Emissivity')
plt.title('Bulk-Emission Monte Carlo (Single Layer)')
plt.grid(True)
plt.show()
