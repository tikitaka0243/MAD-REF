import numpy as np
import pandas as pd
from tqdm import tqdm
import netCDF4 as nc
import os

from OceanMask import is_ocean_3d_vector
from DataNormalization import coordinates_normalization


# -------------- Generate domian points ---------------

def generate_domain_points(save_path, r_min, r_max, theta_min, theta_max, phi_min, phi_max, r_num, theta_num, phi_num):

    print('Generating domain points.')

    r_range = np.linspace(r_min, r_max, r_num, endpoint=True)
    theta_range = np.linspace(theta_min, theta_max, theta_num, endpoint=True)
    phi_range = np.linspace(phi_min, phi_max, phi_num, endpoint=True)


    # Generate the four-dimensional meshgrid
    r, theta, phi = np.meshgrid(r_range, theta_range, phi_range)


    # Flatten the values of the four variables into one-dimensional arrays and convert to a Pandas DataFrame
    data = np.column_stack((np.round(r.ravel(), 4), theta.ravel(), phi.ravel()))
    
    # print('mesh_pre:\n', data.describe())
    # print(data)
    
    # Filter out rows with land coordinates

    is_ocean = is_ocean_3d_vector(data)

    # print(len(is_ocean), np.sum(is_ocean))

    data = data[is_ocean]

     
    data = coordinates_normalization(data, r_min, r_max, theta_min, theta_max, phi_min, phi_max)

    # Export the DataFrame to a CSV file
    # print('mesh_pre:\n', data.describe())
    # print(data)
    save_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', save_path))
    np.save(save_path, data)