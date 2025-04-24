import numpy as np

class Material:
    """
    Represents an optical material, providing its complex refractive index.

    Usage for constant refractive index:
        m = Material(name, n_const, k_const)
        cn = m.get_complex_n(wl)

    Usage for wavelength-dependent data:
        m = Material(name, 0.0, 0.0)
        m.ReadFromFile(filename)
        cn = m.get_complex_n(wl)
    """

    def __init__(self, name: str, n: float= 1.0, k: float = 0.0):
        self._name = name
        self._n = float(n)
        self._k = float(k)

        # data arrays will be set when ReadFromFile is called
        self._wavelengths = None
        self._n_data = None
        self._k_data = None

    def ReadFromFile(self, filename: str):
        """
        Load wavelength-dependent refractive indices from a CSV file.
        The file must have three comma-separated columns:
            wavelength (in micrometers), n, k

        Raises:
            ValueError: if the file format is incorrect or data malformed.
        """
        try:
            data = np.loadtxt(filename, delimiter=',')
        except Exception as e:
            raise ValueError(f"Could not read file '{filename}': {e}")

        if data.ndim != 2 or data.shape[1] < 3:
            raise ValueError(f"File '{filename}' must have at least three columns: wavelength, n, k")

        # parse columns
        wl = data[:, 0]
        n_vals = data[:, 1]
        k_vals = data[:, 2]

        # sort by wavelength to ensure monotonic increase
        idx = np.argsort(wl)
        wl_sorted = wl[idx]
        n_sorted = n_vals[idx]
        k_sorted = k_vals[idx]

        # store arrays
        self._wavelengths = wl_sorted
        self._n_data = n_sorted
        self._k_data = k_sorted

    def get_complex_n(self, wavelength_meters: float) -> complex:
        """
        Returns the complex refractive index at a given wavelength.

        If wavelength-dependent data has been loaded via ReadFromFile,
        this method will interpolate (linear) within the data range.
        Otherwise, it returns the constant n + i*k provided at init.

        Args:
            wavelength (float): Wavelength in meters.

        Returns:
            complex: n(wl) + 1j*k(wl)

        Raises:
            ValueError: if wavelength outside the loaded data range.
        """
        # convert to microns
        wavelength = wavelength_meters * 1e6
        
        # if data arrays present, use interpolation
        if self._wavelengths is not None:
            wl_arr = self._wavelengths
            # check range
            if wavelength < wl_arr[0] or wavelength > wl_arr[-1]:
                raise ValueError(
                    f"Wavelength {wavelength} is outside data range "
                    f"({wl_arr[0]} to {wl_arr[-1]})"
                )
            # linear interpolation
            n_interp = np.interp(wavelength, wl_arr, self._n_data)
            k_interp = np.interp(wavelength, wl_arr, self._k_data)
            return n_interp + 1j * k_interp

        # fallback: constant values
        return self._n + 1j * self._k
    

    def get_absorption_coefficient(self, wavelength_meters: float) -> float:
        """
        Calculate the absorption coefficient α at a given wavelength.

        α(λ) = 4π * k(λ) / λ

        - wavelength: in meters
        - returns α in inverse meters (m⁻¹).
        """
        # get extinction coefficient k from complex refractive index
        cn = self.get_complex_n(wavelength_meters)
        k_val = cn.imag
        return 4 * np.pi * k_val / wavelength_meters

    def __repr__(self) -> str:
        return f"Material('{self._name}')"
    
