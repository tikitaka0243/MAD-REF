import numpy as np
import pandas as pd
from global_land_mask import globe
import matplotlib.pyplot as plt
from tqdm import tqdm
import netCDF4 as nc

import sys
sys.path.append()
from OceanMask import is_ocean_3d_fast


# -------------- Generate domian points ---------------

def generate_domain_points(r_min, r_max, theta_min, theta_max, phi_min, phi_max, r_num, theta_num, phi_num, save_path):
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

    dataset = nc.Dataset('GLO-MFC_001_024_mask_bathy.nc')
    all_vars = dataset.variables.keys()

    latitude = dataset.variables['latitude'][:]
    longitude = dataset.variables['longitude'][:]
    deptho = dataset.variables['deptho'][:]
    
    print('mesh_pre:\n', data.describe())
    print(data)
    
    is_ocean = [True] * len(data)
    # Filter out rows with land coordinates
    for j in tqdm(range(len(data)), mininterval=8):
        r, theta, phi = data.iloc[j]
        if not is_ocean_3d_fast(-r, theta, phi, latitude, longitude, deptho):
            is_ocean[j] = False
        else:
            continue
    data = data[is_ocean]


    # Export the DataFrame to a CSV file
    print('mesh_pre:\n', data.describe())
    print(data)
    np.save(save_path, data.to_numpy())