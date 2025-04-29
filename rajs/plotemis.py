import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os

def load_emissivity_data(csv_path):
    """
    Reads a tab-separated CSV and returns wavelength and emissivity arrays.
    """
    df = pd.read_csv(csv_path, sep='\t')
    wavelengths = df['Wavelength (microns)'].to_numpy()
    emissivity  = df['Emissivity'].to_numpy()
    return wavelengths, emissivity

def main():
    parser = argparse.ArgumentParser(description='Plot emissivity from one or more CSV files.')
    parser.add_argument('files', nargs='+', help='Paths to CSV files containing emissivity data.')
    args = parser.parse_args()

    plt.figure()
    for path in args.files:
        wls, eps = load_emissivity_data(path)
        label = os.path.basename(path)
        plt.plot(wls, eps, label=label)
    plt.xlabel('Wavelength (microns)')
    plt.ylabel('Emissivity')
    plt.title('Emissivity vs. Wavelength')
    plt.grid(True)
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()
