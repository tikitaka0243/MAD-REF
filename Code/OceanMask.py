import os
import netCDF4 as nc
import numpy as np

    
    
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


def find_closest(list, value):
    min_value = min(list)
    if value == 180:
        return 0
    return round((value - min_value) / (1 / 12))


if __name__ == '__main__':
    print(is_ocean_3d_fast(131, 24.097, -27.157))