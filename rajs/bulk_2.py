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

wavelengths      = np.linspace(0.3, 2.4, 1)  # in microns
wavelengths = np.array([2.4])
N_rays_per_wl    = 10
temperature      = 700.0
max_bounces      = 10

start_pos = Vec2(0, 520)
start_dir = 300


for wl in wavelengths:
    print(iface.TraceOneRay(start_pos, start_dir, wl * 1e-6, temperature, max_bounces, debug=True))

