import numpy as np
import pandas as pd
from global_land_mask import globe
import matplotlib.pyplot as plt
from tqdm import tqdm
import netCDF4 as nc

import sys
sys.path.append('/public/home/tianting/PINN_equator/')
from ocean_mask import is_ocean_3d_fast, get_ocean_depth_fast
from Functions import move_phi


print('\n############### LoadData6_2 (Generate mesh_pre_fast) #################')


data_path = ['Equator_all/mesh_pre.csv', 'WholeWorld_sample/mesh_pre.csv']
save_path = ['mesh_pre_temp.csv', 'mesh_pre_WW_temp.csv']
for model in range(1, 2):
    mesh_pre = pd.read_csv(data_path[model])
    if model == 0:
        mesh_pre.loc[mesh_pre['phi'] <= -20 / 180 * np.pi, 'phi'] += 200 / 180 * np.pi
        mesh_pre.loc[mesh_pre['phi'] > -20 / 180 * np.pi, 'phi'] -= 160 / 180 * np.pi

    dataset = nc.Dataset('/public/home/tianting/PINN_equator/GLO-MFC_001_024_mask_bathy.nc')
    all_vars = dataset.variables.keys()
    latitude = dataset.variables['latitude'][:]
    longitude = dataset.variables['longitude'][:]
    deptho = dataset.variables['deptho'][:]


    mesh_pre['deep'] = False
    for i in tqdm(range(len(mesh_pre)), mininterval=10):
        x = get_ocean_depth_fast(mesh_pre.iloc[i, 1], mesh_pre.iloc[i, 2], latitude, longitude, deptho)
        if x is not None:
            if x >= 2000:
                mesh_pre.loc[i, 'deep'] = True
    
    mesh_pre.to_csv(save_path[model], index=False)




# mesh_pre = pd.read_csv('mesh_pre_WW_temp.csv')
# mesh_pre['drop'] = False
# print(mesh_pre['deep'].value_counts())
# print(mesh_pre, '\n', mesh_pre.describe())


# for i in range(len(mesh_pre))[:100]:
#     i_theta = mesh_pre.loc[i, 'theta']
#     i_phi = mesh_pre.loc[i, 'phi']
#     print(i_theta, i_phi)