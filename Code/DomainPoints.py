import numpy as np
import pandas as pd
from tqdm import tqdm
import netCDF4 as nc
import os

from OceanMask import is_ocean_3d_fast
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
    data = pd.DataFrame({
        'r': np.round(r.ravel(), 4),
        'theta': theta.ravel(),
        'phi': phi.ravel()
    })

    file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Data', 'GLO-MFC_001_024_mask_bathy.nc'))
    dataset = nc.Dataset(file_path)
    # all_vars = dataset.variables.keys()

    latitude = dataset.variables['latitude'][:]
    longitude = dataset.variables['longitude'][:]
    deptho = dataset.variables['deptho'][:]
    
    # print('mesh_pre:\n', data.describe())
    # print(data)
    
    is_ocean = [True] * len(data)
    # Filter out rows with land coordinates

    for j in tqdm(range(len(data))):
        r, theta, phi = data.iloc[j]
        if not is_ocean_3d_fast(-r, theta, phi, latitude, longitude, deptho):
            is_ocean[j] = False
        else:
            continue
    data = data[is_ocean]
    data = data.to_numpy()
    
    data = coordinates_normalization(data, r_min, r_max, theta_min, theta_max, phi_min, phi_max)

    # Export the DataFrame to a CSV file
    # print('mesh_pre:\n', data.describe())
    # print(data)
    save_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', save_path, 'domain_points.npy'))
    np.save(save_path, data)