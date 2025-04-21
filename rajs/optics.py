import numpy as np
from scipy.constants import h, c, k as kB



def planck_spectral_radiance(wavelength_metres: float, temperature: float) -> float:
    """
    B_λ(T) [W / (m²·sr·m)] for a black body at T.
    wavelength in meters.
    """
    lam = wavelength_metres

    return (2*h*c**2) / (lam**5 * (np.exp(h*c/(lam*kB*temperature)) - 1.0))


import numpy as np
from scipy.constants import c, h, k as kB

def fresnel_coefs(n1, n2, cos_i):
    """
    Complex-n Fresnel for unpolarized light:
      n1, n2: complex refractive indices (n + i k)
      cos_i:  cosine of incidence angle (real, ≥0)
    Returns real power reflectance R and transmittance T.
    """
    # clamp cos_i
    cos_i = np.clip(cos_i, 0.0, 1.0)
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

    return R, T
