import pandas as pd
import os
from astropy.coordinates import Distance
import astropy.units as u
import numpy as np

def get_DC2_data(file_name = "dp2_star_gmax_27_skim.parquet", folder = "data/background"):
    """
    Load the DC2 data from a parquet file and return it as a pandas DataFrame.
    
    Parameters:
    file_name (str): The name of the parquet file to load. Default is "DC2_object_catalog_v1.1.4.parquet".
    folder (str): The folder where the parquet file is located. Default is "data".
    
    Returns:
    pd.DataFrame: A DataFrame containing the DC2 data.
    """
    # Get the absolute path to the project root
    # __file__ is src/streamgoggles/utils.py, so go up two levels to reach project root
    package_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(package_dir))
    folder_abs = os.path.join(project_root, folder)
    path = os.path.join(folder_abs, file_name)
    try:
        data = pd.read_parquet(path)
        return data
    except Exception as e:
        print(f"Error loading data from {file_name}: {e}")
        return None


def convert_DM_to_kpc(distance_modulus):
    return Distance(distmod=distance_modulus).to(u.kpc)

def convert_kpc_to_DM(distance_kpc):
    return Distance(distance_kpc, unit=u.kpc).distmod

def convert_FeH_to_z(feh):
    # Function coming from github.com/DarkEnergySurvey/ugali/blob/e4aa0a26e9d245d489ccb6b4cdb149386fdcb45b/ugali/isochrone/parsec.py#L190
    # Taken from Table 3 and Section 3 of Bressan et al. 2012
    # Confirmed in Section 2.1 of Marigo et al. 2017
    Y_p     = 0.2485           # Primordial He abundance
    c       = 1.78             # He enrichment ratio

    Z_solar = 0.01524          # Solar metal abundance
    Y_solar = 0.2485           # Solar He abundance
    X_solar = 1 - Y_solar - Z_solar

    return (1 - Y_p)/( (1 + c) + X_solar/Z_solar * 10**(-feh))