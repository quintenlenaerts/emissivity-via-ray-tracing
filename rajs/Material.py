import pandas as pd
import numpy as np

from optics import complex_refractive_index

class Material:
# the material class will be used to create a material object with the following properties:
#     - name: str
#     - refractive_index: [float, float]
#     - absorption: [float, float]
#  it will load the absorption data from a csv file and use core_optical_functions to calculate the refractive index and excting coefficient to be used in the simulation

    class MateterialTypes:
        ABSORPTION_DATA = 0
        SINGLE_NK = 1
        NK_DATA = 2

    def __init__(self, name: str, filename : str, temperature: float = 293.15, material_type=MateterialTypes.ABSORPTION_DATA):
        """
        Initializes the Material class with a name and a filename for absorption data.

        Args:
            name (str): The name of the material.
            filename (str): The path to the CSV file containing absorption data.
            Mateterial_type (int): Material.MaterialTypes.[ABSORPTION DATA / SINGLE NK]
                Have to call .set_single_nk() 
        """
        self.name = name
        self.filename = filename
        self.temperature = temperature
        
        self.wl = None
        self.absorption_coef = None
                
        self.refractive_index = None
        self.exciting_coef = None

        self.material_type = material_type
        if (material_type == Material.MateterialTypes.ABSORPTION_DATA):
            self.load_absorption_data()
        elif (material_type == Material.MateterialTypes.NK_DATA):
            self.load_nk_data(self.filename)
        
        # single complex index of refraction; used only when materialtype.single nk is used
        self.single_n = None
        self.single_k = None
        
        
    def load_absorption_data(self):
        """
        Loads absorption data from a CSV file and calculates the refractive index and absorption coefficient.
        """
        # Load the data from CSV file
        self.df = pd.read_csv(self.filename, sep = "\t")
        # sort the data by wavelength
        self.df = self.df.sort_values(by='Wavelength [nm]')
        
        # remove NaN values
        self.df = self.df.dropna()
        
        # the data should be smoothed a bit
        self.df['Absorption Coef. [cm-1]'] = self.df['Absorption Coef. [cm-1]'].rolling(window=5).mean()
        # self.df = self.df.dropna()
        
        
        self.wl = self.df['Wavelength [nm]'].values * 1e-3  # Convert nm to µm
        self.absorption_coef = self.df['Absorption Coef. [cm-1]'].values  # cm⁻¹
        
        self.refractive_index = np.zeros(len(self.wl))
        self.excitation_coef = np.zeros(len(self.wl))
        
        # Calculate the refractive index and excting coefficient
        for i, wl in enumerate(self.wl):
            self.refractive_index[i], self.excitation_coef[i] = complex_refractive_index(wl, self.absorption_coef[i], self.temperature)
    
    def set_single_nk(self, n_value : float, k_value : float):
        self.single_n = n_value
        self.single_k = k_value
        
    def get_complex_index(self, wavelength_microns: float):

        if self.material_type == Material.MateterialTypes.SINGLE_NK:
            if self.single_k == None or self.single_n == None:
                print("Single NK of material : " + self.name + " not yet set.")
            return self.single_n, self.single_k

        # for the given wavelength, if it doesn't exist in the array, interpolate the value:
        # check if the wavelength is within the range of the data
        if wavelength_microns < self.wl[0] or wavelength_microns > self.wl[-1]:
            raise ValueError(f"Wavelength {wavelength_microns} microns is out of range ({self.wl[0]} - {self.wl[-1]} microns)")
        
        # find the nearest wavelength in the array
        index = np.searchsorted(self.wl, wavelength_microns)
        
        # if the wavelength is exactly in the array, return the value
        if self.wl[index] == wavelength_microns:
            return self.refractive_index[index], self.excitation_coef[index]
        
        # if the wavelength is not in the array, interpolate the value
        else:
            if index == 0:
                return self.refractive_index[0], self.excitation_coef[0]
            elif index == len(self.wl):
                return self.refractive_index[-1], self.excitation_coef[-1]
            else:
                # linear interpolation
                wl1, wl2 = self.wl[index-1], self.wl[index]
                n1, n2 = self.refractive_index[index-1], self.refractive_index[index]
                k1, k2 = self.excitation_coef[index-1], self.excitation_coef[index]
                
                n = n1 + (n2 - n1) * (wavelength_microns - wl1) / (wl2 - wl1)
                k = k1 + (k2 - k1) * (wavelength_microns - wl1) / (wl2 - wl1)
                
                return n, k

    def load_nk_data(self, filename: str):
        """
        Loads wavelength-dependent complex refractive index (n and k) data from a CSV file
        CSV format (no header): wavelength [µm], refractive index n, extinction coefficient k.
        Computes absorption coefficient (cm⁻¹) from k.
        """
        # Load the data from CSV file (comma-separated, no header)
        df = pd.read_csv(filename, header=None, names=['wl', 'n', 'k'])
        # sort the data by wavelength
        df = df.sort_values(by='wl')
        # remove NaN values
        df = df.dropna()

        # set wavelength and complex index arrays
        self.wl = df['wl'].values  # µm
        self.refractive_index = df['n'].values
        self.excitation_coef = df['k'].values
        # compute absorption coefficient alpha [cm⁻¹]: alpha = 4 * pi * k / lambda_cm
        # lambda_cm = wl [µm] * 1e-4 cm/µm
        # thus alpha = 4*pi*k / (wl*1e-4) = 4*pi*k*1e4 / wl
        self.absorption_coef = 4 * np.pi * self.excitation_coef * 1e4 / self.wl

    def get_alpha(self, wavelength_microns):
        """
            Returns alpha in cm^-1
        """
        
        if self.material_type == Material.MateterialTypes.SINGLE_NK:
            return 1e4 *  4 * np.pi * self.single_k / wavelength_microns

        # for the given wavelength, if it doesn't exist in the array, interpolate the value:
        # check if the wavelength is within the range of the data
        
        if wavelength_microns < self.wl[0] or wavelength_microns > self.wl[-1]:
            raise ValueError(f"Wavelength {wavelength_microns} microns is out of range ({self.wl[0]} - {self.wl[-1]} microns)")
        # find the nearest wavelength in the array 
        index = np.searchsorted(self.wl, wavelength_microns)
        
        # if the wavelength is exactly in the array, return the value
        if self.wl[index] == wavelength_microns:
            return self.absorption_coef[index]
        
        # if the wavelength is not in the array, interpolate the value
        
        else:
            if index == 0:
                return self.absorption_coef[0]
            elif index == len(self.wl):
                return self.absorption_coef[-1]
            else:
                # linear interpolation
                wl1, wl2 = self.wl[index-1], self.wl[index]
                alpha1, alpha2 = self.absorption_coef[index-1], self.absorption_coef[index]
                
                alpha = alpha1 + (alpha2 - alpha1) * (wavelength_microns - wl1) / (wl2 - wl1)
                
                return alpha
            
            
    def make_tables(self, wls):
        """
        Create tables for refractive index and absorption coefficient.

        Args:
            wl_min (float): Minimum wavelength in microns.
            wl_max (float): Maximum wavelength in microns.
        """
        # Create a new DataFrame with the specified wavelength range
        self.n_table = np.zeros(len(wls))
        self.k_table = np.zeros(len(wls))
        self.a_table = np.zeros(len(wls))
        
        for i, wl in enumerate(wls):
            n, k = self.get_complex_index(wl)
            alpha = self.get_alpha(wl)
            
            self.n_table[i] = n
            self.k_table[i] = k
            self.a_table[i] = alpha
            
        print(self.n_table, self.a_table)
            

            
