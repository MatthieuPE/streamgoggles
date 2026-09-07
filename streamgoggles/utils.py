import numpy
import pandas as pd
import os

def get_DC2_data(file_name = "dp2_star_gmax_27_skim.parquet", folder = "data"):
    """
    Load the DC2 data from a parquet file and return it as a pandas DataFrame.
    
    Parameters:
    file_name (str): The name of the parquet file to load. Default is "DC2_object_catalog_v1.1.4.parquet".
    folder (str): The folder where the parquet file is located. Default is "data".
    
    Returns:
    pd.DataFrame: A DataFrame containing the DC2 data.
    """
    path = os.path.join(folder, file_name)
    try:
        data = pd.read_parquet(path)
        return data
    except Exception as e:
        print(f"Error loading data from {file_name}: {e}")
        return None


