import numpy as np
from numba import njit
from typing import Tuple

from scipy.constants import h, c, k

Vec2 = tuple

@njit
def diamond_base_refractive_index(wavelength_microns: float) -> float:
    """
    Calculate the base refractive index of diamond using Sellmeier equation. 
    
    Based on values from: 
    G. Turri, S. Webster, Y. Chen, B. Wickham, A. Bennett, and M. Bass, "Index of refraction from the near-ultraviolet
        to the near-infrared from a single crystal microwave-assisted CVD diamond," Opt. Mater. Express  7, 855-859 (2017).

    Only applies for the 300 nm - 25 µm range. 
    
    Args:
        wavelength_microns: Wavelength in microns
        
    Returns:
        Refractive index (real part)
    """
    
    wavelength_nanometres = wavelength_microns * 1e3  # Convert to nm for Sellmeier equation
    # Sellmeier coefficients for diamond
    B1 = 4.658
    C1 = 112.5**2  # μm²
    
    n_squared = 1 + (B1 * wavelength_nanometres**2) / ( wavelength_nanometres**2 - C1)
    return np.sqrt(n_squared)


@njit
def diamond_base_refractive_index_temperature(wavelength_microns: float, temperature : float) -> float:
    """
    Calculate the base refractive index of diamond using Sellmeier equation. 
    
    Based on values from: 
    Mollart, T. P., and K. L. Lewis. "The infrared optical properties of CVD diamond at elevated temperatures."
        physica status solidi (a) 186.2 (2001): 309-318.
        
    and from:
    
    Dischler, Bernhard, and Christoph Wild, eds. Low-pressure synthetic diamond: manufacturing and applications. 
        Springer Science & Business Media, 2013.
    
    Args:
        wavelength_microns: Wavelength in microns
        temperature: Temperature in Kelvin
        
    Returns:
        Refractive index (real part)
    """
    
    wavelength_nanometres = wavelength_microns * 1e3  # Convert to nm for Sellmeier equation
    # Sellmeier coefficients for diamond
    B1 = 0.3306
    B2 = 4.3356
    
    C1 = 175**2 
    C2 = 106**2
    
    n_squared = 1 + (B1 * wavelength_nanometres**2) / ( wavelength_nanometres**2 - C1) + \
        (B2 * wavelength_nanometres**2) / ( wavelength_nanometres**2 - C2)
    

    temp_C = temperature - 273.15  # Convert to Celsius
    dn_dT = 7.8e-6 + 9.8e-8 * temp_C - 9.84e-11 * temp_C**2 + 4.74e-14 * temp_C**3
     
    return np.sqrt(n_squared) + dn_dT * temp_C


# @njit
# def nitrogen_specific_absorption(wavelength_microns: float, concentration_ppm: float) -> float:
#     """
#     Calculate absorption coefficient contribution from nitrogen dopants.
    
#     Args:
#         wavelength_microns: Wavelength in microns
#         concentration_ppm: Nitrogen concentration in parts per million
        
#     Returns:
#         Absorption coefficient contribution (cm⁻¹)
#     """
#     # Simplified model: Nitrogen has absorption peaks around 270nm and 370nm
#     # Actual data would be better but this gives reasonable behavior
#     peak1 = 0.27  # μm
#     peak2 = 0.37  # μm
#     width1 = 0.02
#     width2 = 0.03
    
#     # Lorentzian peak profiles
#     absorption1 = 50.0 * concentration_ppm * width1**2 / ((wavelength_microns - peak1)**2 + width1**2)
#     absorption2 = 30.0 * concentration_ppm * width2**2 / ((wavelength_microns - peak2)**2 + width2**2)
    
#     # Additional mid-IR absorption from nitrogen platelets
#     if 7.0 < wavelength_microns < 10.0:
#         plateau_absorption = 2.0 * concentration_ppm * np.exp(-(wavelength_microns - 8.5)**2 / 1.5**2)
#     else:
#         plateau_absorption = 0.0
        
#     return absorption1 + absorption2 + plateau_absorption

# @njit
# def boron_specific_absorption(wavelength_microns: float, concentration_ppm: float) -> float:
#     """
#     Calculate absorption coefficient contribution from boron dopants.
    
#     Args:
#         wavelength_microns: Wavelength in microns
#         concentration_ppm: Boron concentration in parts per million
        
#     Returns:
#         Absorption coefficient contribution (cm⁻¹)
#     """
#     # Boron doping creates absorption in infrared region around 2500 cm⁻¹ (4 μm)
#     peak = 4.0  # μm
#     width = 0.8
    
#     # Main absorption peak
#     absorption = 15.0 * concentration_ppm * width**2 / ((wavelength_microns - peak)**2 + width**2)
    
#     # Additional continuum absorption in p-type diamond from free carriers
#     if wavelength_microns > 2.0:
#         # Free carrier absorption ~ λ²
#         free_carrier = 0.05 * concentration_ppm * (wavelength_microns / 2.0)**2
#     else:
#         free_carrier = 0.0
        
#     return absorption + free_carrier

# @njit
# def base_absorption(wavelength_microns: float) -> float:
#     """
#     Calculate intrinsic absorption of pure diamond.
    
#     Args:
#         wavelength_microns: Wavelength in microns
        
#     Returns:
#         Base absorption coefficient (cm⁻¹)
#     """
#     # Pure diamond is transparent from ~0.22 μm to far infrared
#     # Below 0.22 μm: band gap absorption
#     # Above ~20 μm: multi-phonon absorption
    
#     if wavelength_microns < 0.22:
#         # UV absorption due to band gap
#         return 1000.0 * np.exp((0.22 - wavelength_microns) * 50)
#     elif wavelength_microns > 20.0:
#         # Far-IR multi-phonon absorption
#         return 0.1 * (wavelength_microns - 20.0)**2
#     else:
#         # Minimal absorption in transparency window
#         # Small Rayleigh scattering contribution
#         return 0.01 * (0.4 / wavelength_microns)**4

@njit
def absorption_to_k(wavelength_microns: float, absorption_coeff: float) -> float:
    """
    Convert absorption coefficient to imaginary part of refractive index.
    
    Args:
        wavelength_microns: Wavelength in microns
        absorption_coeff: Absorption coefficient in cm⁻¹
        
    Returns:
        Imaginary part of refractive index (k)
    """
    # Formula: k = α·λ / (4π)
    # α in cm⁻¹, λ in cm
    return absorption_coeff * (wavelength_microns * 1e-4) / (4 * np.pi)


@njit
def attenuation(alpha, dist):
    return np.exp(-alpha * dist)

@njit
def schlick_reflectance(n1, n2, cos_i):
    r0 = ((n1 - n2)/(n1 + n2))**2
    return r0 + (1.0 - r0)*(1.0 - cos_i)**5

@njit
def reflect2(d0x, d0y, nx, ny):
    # return reflected direction (rx, ry)
    dot = d0x*nx + d0y*ny
    return d0x - 2*dot*nx, d0y - 2*dot*ny

@njit
def refract2(d0x, d0y, nx, ny, eta):
    cos_i = -(d0x*nx + d0y*ny)
    k = 1.0 - eta*eta*(1.0 - cos_i*cos_i)
    if k < 0.0:
        return False, 0.0, 0.0
    s = eta*cos_i - np.sqrt(k)
    rx = eta*d0x + s*nx
    ry = eta*d0y + s*ny
    return True, rx, ry

@njit
def complex_refractive_index(wavelength_microns: float, absorption: float, temperature = None) -> Tuple[float, float]:
    """
    Calculate complex refractive index for doped diamond.
    
    Args:
        wavelength_microns: Wavelength in microns
        dopant_type: Type of dopant ('nitrogen', 'boron', etc.)
        dopant_concentration: Dopant concentration in ppm
        temperature: Temperature in Kelvin (optional)
        
    Returns:
        Tuple of (n, k) - real and imaginary parts of refractive index
    """
    # Real part - base diamond refractive index
    if temperature is not None:
        n = diamond_base_refractive_index_temperature(wavelength_microns, temperature)
    else:
        n = diamond_base_refractive_index(wavelength_microns)
        
    # Convert to imaginary part of refractive index
    k = absorption_to_k(wavelength_microns, absorption)
    
    return n, k

@njit
def planck_spectral_radiance(wavelength_microns: float, temperature: float) -> float:
    """
    Calculate the spectral radiance of a black body using Planck's law.
    
    Args:
        wavelength_microns: Wavelength in microns
        temperature: Temperature in Kelvin
    
    
    Returns:
        Spectral radiance in W/m²/sr/m
    """
    
        
    wavelength_metres = wavelength_microns * 1e-6  # Convert to meters
    
    B = 2.0 * h * c**2 / (wavelength_metres**5 * (np.exp(h * c / (wavelength_metres * k * temperature)) - 1.0))
    
    return B


@njit
def normalize2(v: Vec2) -> Vec2:
    norm = np.sqrt(v[0]**2 + v[1]**2)
    if norm == 0:
        return v
    return (v[0]/norm, v[1]/norm)

@njit
def dot2(v1: Vec2, v2: Vec2) -> float:
    return v1[0]*v2[0] + v1[1]*v2[1]

@njit
def scale2(v: Vec2, s: float) -> Vec2:
    return (v[0]*s, v[1]*s)

@njit
def add2(v1: Vec2, v2: Vec2) -> Vec2:
    return (v1[0]+v2[0], v1[1]+v2[1])

# # For reflection in 2D (works the same as in 3D but with 2 components)
# @njit
# def reflect2(direction: Vec2, normal: Vec2) -> Vec2:
#     # Reflection: r = d - 2*(d·n)*n
#     return (direction[0] - 2 * dot2(direction, normal) * normal[0],
#             direction[1] - 2 * dot2(direction, normal) * normal[1])