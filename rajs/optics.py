import numpy as np
from scipy.constants import h, c, k as kB
from numba import njit


@njit(cache=True)
def planck_spectral_radiance(wavelength_metres: float, temperature: float) -> float:
    """
    B_λ(T) [W / (m²·sr·m)] for a black body at T.
    wavelength in meters.
    """
    lam = wavelength_metres

    return (2*h*c**2) / (lam**5 * (np.exp(h*c/(lam*kB*temperature)) - 1.0))


@njit(cache=True)
def fresnel_coefs_full(n1, n2, cos_i):
    """
    Complex-n Fresnel for unpolarized light:
      n1, n2: complex refractive indices (n + k)
      cos_i:  cosine of incidence angle (real, ≥0)
    Returns real power reflectance R and transmittance T.
    """
    # clamp cos_i    
    if cos_i < 0.0:
        cos_i = 0.0
    elif cos_i > 1.0:
        cos_i = 1.0
    sin_i2 = 1.0 - cos_i*cos_i

    # ratio of indices
    eta = n1 / n2

    # complex transmitted angle cosine
    sin_t2 = eta*eta * sin_i2
    cos_t = np.sqrt(1.0 - sin_t2)

    # s‑polarization amplitude coefficients
    rs = (n1*cos_i - n2*cos_t) / (n1*cos_i + n2*cos_t)
    ts = (2.0*n1*cos_i) / (n1*cos_i + n2*cos_t)

    # p‑polarization amplitude coefficients
    rp = (n2*cos_i - n1*cos_t) / (n2*cos_i + n1*cos_t)
    tp = (2.0*n1*cos_i) / (n2*cos_i + n1*cos_t)

    # power reflectances
    R_s = np.abs(rs)**2
    R_p = np.abs(rp)**2
    R   = 0.5*(R_s + R_p)

    # power transmittances
    # account for change in impedance & geometry:
    #   T_s = Re(n2*cos_t)/(Re(n1)*cos_i) * |ts|^2
    # same for p
    factor = (n2.real * cos_t.real) / (n1.real * cos_i)
    T_s = factor * np.abs(ts)**2
    T_p = factor * np.abs(tp)**2
    T   = 0.5*(T_s + T_p)

    return R, T, rs, ts, rp, tp


@njit(cache=True)
def fresnel_coefs(n1, n2, cos_i):
    R, T, _, _, _, _ = fresnel_coefs_full(n1, n2, cos_i)
    return R,T

import math

@njit(cache=True)
def planck_lambda(wl, T):
    """
        wl in microns
    """
    # wl: wavelength in mirconrs, T: temperature in K
    # Physical constants
    c1 = 3.741771e-16  # W·m^2 (first radiation constant)
    c2 = 1.438776e-2   # m·K   (second radiation constant)
    lam_m = wl * 1e-6 # convert  to meters
    # Planck's law: spectral radiance per unit wavelength
    return (c1 / (lam_m**5)) / (math.exp(c2 / (lam_m * T)) - 1.0)

@njit(cache=True)
def schlick_reflectance(n1, n2, cos_i):
    """
    Calculates reflectance using Schlick's approximation for unpolarized light.

    Args:
        n1 (float): Refractive index of the incident medium.
        n2 (float): Refractive index of the transmitting medium.
        cos_i (float): Cosine of the angle of incidence (must be >= 0).

    Returns:
        float: Reflectance probability (0 to 1).
    """
    # Ensure cos_i is non-negative (angle between 0 and 90 degrees)
    cos_i = max(0.0, cos_i)
    # Calculate R0, the reflectance at normal incidence
    r0_num = (n1 - n2)
    r0_den = (n1 + n2)
    if abs(r0_den) < 1e-12: # Avoid division by zero if indices are equal
        r0 = 0.0
    else:
        r0 = (r0_num / r0_den)**2
    # Schlick's approximation formula
    return r0 + (1.0 - r0) * ((1.0 - cos_i)**5)





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




@njit(cache=True)
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

@njit(cache=True)
def complex_refractive_index(wavelength_microns: float, absorption: float, temperature = None) -> tuple[float, float]:
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