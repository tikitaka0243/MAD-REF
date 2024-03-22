import os
import netCDF4 as nc
import numpy as np
import pandas as pd


def is_ocean_3d_vector(data):

    file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Data', 'GLO-MFC_001_024_mask_bathy.nc'))
    dataset = nc.Dataset(file_path)
    # all_vars = dataset.variables.keys()

    latitude = dataset.variables['latitude'][:]
    longitude = dataset.variables['longitude'][:]
    deptho = dataset.variables['deptho'][:]
    
    lat = data[:, 1]
    lon = data[:, 2]
    loc_lat = find_closest_vector(latitude, lat)
    loc_lon = find_closest_vector(longitude, lon)

    depth = -data[:, 0]

    masked_land = deptho.mask[loc_lat, loc_lon]

    is_ocean = deptho[loc_lat, loc_lon] >= depth
    is_ocean[masked_land] = False
    is_ocean[lat < -80] = False

    return is_ocean


def is_ocean_3d_fast(depth, lat ,lon, latitude, longitude, deptho):

    if lat <= -80:
        return False
    
    loc_lat = find_closest(latitude, lat)
    loc_lon = find_closest(longitude, lon)

    if deptho.mask[loc_lat][loc_lon] == True:
        return False
    elif deptho[loc_lat][loc_lon] < depth:
        return False
    else:
        return True
    

def find_closest_vector(list, value):
    min_value = min(list)
    loc = np.copy(value)
    value_180 = loc == 180
    loc[value_180] = 0
    loc[~value_180] = np.round((loc[~value_180] - min_value) / (1 / 12))

    return loc.astype(np.int32)


def find_closest(list, value):
    min_value = min(list)
    if value == 180:
        return 0
    return round((value - min_value) / (1 / 12))


if __name__ == '__main__':
    print(is_ocean_3d_fast(131, 24.097, -27.157))