import numpy as np
import pandas as pd
from global_land_mask import globe
import matplotlib.pyplot as plt
from tqdm import tqdm
import netCDF4 as nc

import sys
sys.path.append()
from ocean_mask import is_ocean_3d_fast, get_ocean_depth
from Functions import move_phi


print('\n############### LoadData6 (Generate mesh_pre) #################')


# -------------- Generate pridiction coordinates ---------------

# r_range = np.linspace(0, 1, 81, endpoint=True)
# theta_range = np.linspace(-20 / 180 * np.pi, 20 / 180 * np.pi, 161, endpoint=True)
# phi_range = np.linspace(-50 / 180 * np.pi, 50 / 180 * np.pi, 401, endpoint=True)

# # r_range = np.linspace(0, 1, 51, endpoint=True)
# # theta_range = np.linspace(-80 / 180 * np.pi, 90 / 180 * np.pi, 341, endpoint=True)
# # phi_range = np.linspace(-180 / 180 * np.pi, 180 / 180 * np.pi, 721, endpoint=True)


# mesh_pre = pd.DataFrame(columns=['r', 'theta', 'phi'])

# for theta in tqdm(theta_range):
#     for phi in phi_range:
#         if get_ocean_depth(theta, phi) is not None:
#             new_point = pd.DataFrame({'r': [1], 'theta': [theta], 'phi': [phi]}, index=[0])
#             mesh_pre = pd.concat([mesh_pre, new_point])
            
#             if get_ocean_depth(theta, phi) < 2000:
#                 new_point = pd.DataFrame({'r': [-get_ocean_depth(theta, phi) / 2000 + 1], 'theta': [theta], 'phi': [phi]}, index=[0])
#                 mesh_pre = pd.concat([mesh_pre, new_point])
#             else:
#                 new_point = pd.DataFrame({'r': [0], 'theta': [theta], 'phi': [phi]}, index=[0])
#                 mesh_pre = pd.concat([mesh_pre, new_point])
                

# for theta in [theta_range[0], theta_range[-1]]:
#     for r in tqdm(r_range):
#         for phi in phi_range:
#             if is_ocean_3d((1 - r) * 2000, theta, phi):
#                 new_point = pd.DataFrame({'r': [r], 'theta': [theta], 'phi': [phi]}, index=[0])
#                 mesh_pre = pd.concat([mesh_pre, new_point])
                
# for phi in [phi_range[0], phi_range[-1]]:
#     for r in tqdm(r_range):
#         for theta in theta_range:
#             if is_ocean_3d((1 - r) * 2000, theta, phi):
#                 new_point = pd.DataFrame({'r': [r], 'theta': [theta], 'phi': [phi]}, index=[0])
#                 mesh_pre = pd.concat([mesh_pre, new_point])
                

# # Export the DataFrame to a CSV file
# print('mesh_pre:\n', mesh_pre.describe())
# print(mesh_pre)
# mesh_pre.to_csv('mesh_pre_WW.csv', index=False)


# Old

# Define the range and step size for each variable
for i in tqdm(range(1)):
    if i == 0:
        
        r_range = np.linspace(0, 1, 81, endpoint=True)
        theta_range = np.linspace(-20 / 180 * np.pi, 20 / 180 * np.pi, 161, endpoint=True)
        phi_range = np.concatenate((np.linspace(150 / 180 * np.pi, 180 / 180 * np.pi, 120, endpoint=False), np.linspace(-180 / 180 * np.pi, -110 / 180 * np.pi, 281, endpoint=True)))
    else:
        r_range = np.linspace(0, 1, 41, endpoint=True)
        theta_range = np.linspace(-80 / 180 * np.pi, 90 / 180 * np.pi, 361, endpoint=True)
        phi_range = np.linspace(-180 / 180 * np.pi, 180 / 180 * np.pi, 721, endpoint=True)


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
        if not is_ocean_3d_fast((1 - r) * 2000, theta, phi, latitude, longitude, deptho):
            is_ocean[j] = False
        else:
            continue
    data = data[is_ocean]
    
    if i == 0:
        data['phi'] = move_phi(data['phi'])


    # Export the DataFrame to a CSV file
    print('mesh_pre:\n', data.describe())
    print(data)
    if i == 0:
        data.to_csv('Equator_all/mesh_pre.csv', index=False)
        data.to_csv('Equator_sample/mesh_pre.csv', index=False)
    else: 
        data.to_csv('WholeWorld_sample/mesh_pre.csv', index=False)


# # -------------- Generate pridiction coordinates II ---------------

# resolution = [50, 360, 720]

# T = [15 / 730, 196 / 730, 380 / 730, 561 / 730, 804 / 730]

# R = 1
# THETA = np.linspace(-90 / 180 * np.pi, 90 / 180 * np.pi, resolution[1], endpoint=True).tolist()
# PHI = np.linspace(-180 / 180 * np.pi, 180 / 180 * np.pi, resolution[2], endpoint=True).tolist()
# r, theta, phi, t = np.meshgrid(R, THETA, PHI, T)
# mesh_pre_1 = pd.DataFrame({
#     'r': r.ravel(),
#     'theta': theta.ravel(),
#     'phi': phi.ravel(),
#     't': t.ravel(),
# })

# R = np.linspace(0, 1, resolution[0], endpoint=False).tolist()
# THETA = -90 / 180 * np.pi
# PHI = np.linspace(-180 / 180 * np.pi, 180 / 180 * np.pi, resolution[2], endpoint=True).tolist()
# r, theta, phi, t = np.meshgrid(R, THETA, PHI, T)
# mesh_pre_2 = pd.DataFrame({
#     'r': r.ravel(),
#     'theta': theta.ravel(),
#     'phi': phi.ravel(),
#     't': t.ravel(),
# })

# R = np.linspace(0, 1, resolution[0], endpoint=False).tolist()
# THETA = np.linspace(90 / 180 * np.pi, -90 / 180 * np.pi, resolution[1], endpoint=False).tolist()
# PHI = -180 / 180 * np.pi
# r, theta, phi, t = np.meshgrid(R, THETA, PHI, T)
# mesh_pre_3 = pd.DataFrame({
#     'r': r.ravel(),
#     'theta': theta.ravel(),
#     'phi': phi.ravel(),
#     't': t.ravel(),
# })

# mesh_pre = pd.concat([mesh_pre_1, mesh_pre_2, mesh_pre_3])

# # Filter out rows with land coordinates
# is_land = np.array([globe.is_land(lat / np.pi * 180, lon / np.pi * 180) for lat, lon in zip(mesh_pre['theta'], mesh_pre['phi'])])
# mesh_pre = mesh_pre[~is_land]


# THETA = np.linspace(-90 / 180 * np.pi, 90 / 180 * np.pi, resolution[1], endpoint=True).tolist()
# PHI = np.linspace(-180 / 180 * np.pi, 180 / 180 * np.pi, resolution[2], endpoint=True).tolist()

# boundary_points = pd.DataFrame()
# for i in range(1, len(THETA) - 1):
#     for j in range(1, len(PHI) - 1):
#         lat = THETA[i] / np.pi * 180
#         lat_p = THETA[i + 1] / np.pi * 180
#         lat_m = THETA[i - 1] / np.pi * 180
#         lon = PHI[j] / np.pi * 180
#         lon_p = PHI[j + 1] / np.pi * 180
#         lon_m = PHI[j - 1] / np.pi * 180
        
#         bound__ = globe.is_land(lat, lon)
#         bound_p_p = globe.is_ocean(lat_p, lon_p)
#         bound_p_m = globe.is_ocean(lat_p, lon_m)
#         bound_m_p = globe.is_ocean(lat_m, lon_p)
#         bound_m_m = globe.is_ocean(lat_m, lon_m)
#         bound_p = globe.is_ocean(lat_p, lon)
#         bound__p = globe.is_ocean(lat, lon_p)
#         bound_m = globe.is_ocean(lat_m, lon)
#         bound__m = globe.is_ocean(lat, lon_m)
        
#         if bound__ and (bound_p_p or bound_p_m or bound_m_p or bound_m_m or bound_p or bound__p or bound_m or bound__m):
#             boundary_points = pd.concat([boundary_points, pd.DataFrame([[THETA[i], PHI[j]]])])

# boundary_points.columns = ['theta', 'phi']
# # print(boundary_points, '\n', boundary_points.describe())
            
# R = pd.DataFrame(np.linspace(0, 1, resolution[0], endpoint=True))
# T = pd.DataFrame(T)
# boundary_points = R.merge(boundary_points, how='cross')
# boundary_points = boundary_points.merge(T, how='cross')
# boundary_points.columns = ['r', 'theta', 'phi', 't']
# mesh_pre = pd.concat([mesh_pre, boundary_points])

# print(mesh_pre, '\n', mesh_pre.describe())

# mesh_pre.to_csv('WholeWorld_sample/mesh_pre.csv', index=False)
    









# fig, ax = plt.subplots(figsize=(10, 4), dpi=300)      
# im = ax.scatter(boundary_points['theta'], boundary_points['phi'], s=2)

# cbar = fig.colorbar(im)
# plt.savefig('test.png', bbox_inches='tight')
# plt.close()
        
        
        