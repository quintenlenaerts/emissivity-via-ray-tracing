import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from Material import Material
from graphx import GraphX
from Interface import InterfaceBorder, Interface
from Vec2 import Vec2
from LightRay import LightRay
from optics import planck_spectral_radiance

# --- 1) Set up drawing ---
GraphX.set_size(-5, 35, -5, 80)
GraphX.grid_on()

# Interface width (in microns)
_width = 600

# --- 2) Build sine-wave interfaces ---
InterfaceBorder.BorderType.SINE_SETTINGS(0, 8, 500)
int_top = InterfaceBorder(InterfaceBorder.BorderType.SINE, _width, "TopBorder")
int_top.move_up(50)
int_top.move_right(-0.5 * _width)

InterfaceBorder.BorderType.SINE_SETTINGS(0.8, 4, 500)
int_mid = InterfaceBorder(InterfaceBorder.BorderType.SINE, _width, "MiddleBorder")
int_mid.move_up(10)
int_mid.move_right(-0.5 * _width)

InterfaceBorder.BorderType.SINE_SETTINGS(0.3, 5, 550)
int_bot = InterfaceBorder(InterfaceBorder.BorderType.SINE, _width, "BottomBorder")
int_bot.move_right(-0.5 * _width)

# --- 3) Stack into an Interface ---
iface = Interface()
iface.AddLayer(None,    int_top, Material("Air",     1.0, 0.0))
iface.AddLayer(int_top,  int_mid, Material("Diamond", 1.6, 0.1))  # set k>0 for emission
iface.AddLayer(int_mid,  int_bot, Material("Glass",   1.5, 0.05))
iface.AddLayer(int_bot,  None,    Material("Moly",    1.0, 0.0))
iface.ConnectBorders()

# Draw the borders
GraphX.draw_interface_border(int_top, "red")
GraphX.draw_interface_border(int_mid, "green")
GraphX.draw_interface_border(int_bot, "blue")

# Choose a fixed start position (x,y) above the top interface (units: microns)
start_pos = Vec2(0.0, 75.0)
start_angle = 270.0  # straight down

# --- 4) Monkey-patch correct throughput update into LightRay ---
from LightRay import LightRay as _LR
from Vec2      import Vec2      as _Vec2

def _patched_update_throughput(self, LR_col):
    import numpy as _np
    from optics import planck_spectral_radiance as _planck
    # 1) path length in microns
    d_um = _Vec2.Distance(self._ray.source_pos, LR_col.col.point)
    # 2) wavelengths
    wl_m  = self.wavelength         # meters
    wl_um = wl_m * 1e6               # microns
    # 3) absorption coeff α [1/µm]
    alpha_um = LR_col.mat_old.get_absorption_coefficient(wl_um)
    # 4) Beer–Lambert in consistent units (µm⁻¹·µm)
    attenuation = _np.exp(-alpha_um * d_um)
    eps_seg     = 1.0 - attenuation
    # 5) emitted radiance for segment
    B           = _planck(wl_m, self.temperature)
    emit        = eps_seg * B
    # 6) accumulate and update throughput
    self.emitted_radiance += self.throughput * emit
    self.throughput        *= attenuation

# Apply monkey-patch
_LR._update_throughput = _patched_update_throughput

# --- 5) Single-ray trace with up to two bounces ---
def trace_one_ray(start_pos, start_angle, interface, wavelength, max_bounces=2):
    rays = [LightRay(start_pos, start_angle)]
    rays[0].wavelength   = wavelength
    rays[0].temperature  = 800.0  # K

    for _ in range(max_bounces):
        new_rays = []
        for ray in rays:
            children = ray.Send(interface) or []
            for child in children:
                if child.throughput < 1e-3:
                    continue
                new_rays.append(child)
        if not new_rays:
            break
        rays = new_rays

    return sum(ray.emitted_radiance for ray in rays)

# --- 6) Monte Carlo spectrum with progress bars ---
def simulate_spectrum(wavelengths, N_rays, start_pos, interface, start_angle, max_bounces=2):
    eps = []
    T_bb = 800.0
    for wl in tqdm(wavelengths, desc="Wavelengths", unit="λ"):
        acc = 0.0
        for _ in tqdm(range(N_rays), desc=f"Rays @ {wl*1e6:.1f}μm", unit="ray", leave=False):
            acc += trace_one_ray(start_pos, start_angle, interface, wl, max_bounces)
        L_avg = acc / N_rays
        B_bb  = planck_spectral_radiance(wl, T_bb)
        eps.append(L_avg / B_bb if B_bb > 0 else 0.0)
    return np.array(eps)

# --- 7) Run simulation and plot ---
# Limit to exactly two bounces for correct two-bounce emissivity
wavelengths   = np.linspace(0.3e-6, 2.4e-6, 5)
N_rays_per_wl = 500
max_bounces   = 2

emissivity = simulate_spectrum(wavelengths,
                               N_rays_per_wl,
                               start_pos,
                               iface,
                               start_angle,
                               max_bounces=max_bounces)

# Print results
print(f"Emissivity spectrum (max_bounces={max_bounces}):")
for wl, e in zip(wavelengths, emissivity):
    print(f"{wl*1e6:.2f} μm: {e:.4f}")

# Plot
plt.figure(figsize=(8,4))
plt.plot(wavelengths*1e6, emissivity, '-o')
plt.xlabel('Wavelength (μm)')
plt.ylabel('Emissivity')
plt.title(f'Simulated Spectral Emissivity (2 bounces)')
plt.grid(True)
plt.show()
