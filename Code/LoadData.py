import glob
import pandas as pd
import numpy as np
from tqdm import *
import os
import datetime
import netCDF4 as nc


pd.set_option('display.width', 1000)
pd.options.display.max_columns = 40


def data_filter(data, r_min, r_max, theta_min, theta_max, phi_min, phi_max, t_min, t_max):

    t_min = pd.to_datetime(t_min).value / 10 ** 9
    t_max = pd.to_datetime(t_max).value / 10 ** 9

    data = data[(data[:, 0] >= r_min) & (data[:, 0] <= r_max) &
                (data[:, 1] >= theta_min) & (data[:, 1] <= theta_max) &
                (data[:, 2] >= phi_min) & (data[:, 2] <= phi_max) &
                (data[:, 3] >= t_min) & (data[:, 3] <= t_max)]

    # data[:, 0] = (data[:, 0] - r_min) / (r_max - r_min)
    # data[:, 1] = (data[:, 1] - theta_min) / (theta_max - theta_min)
    # data[:, 2] = (data[:, 2] - phi_min) / (phi_max - phi_min)
    # data[:, 3] = (data[:, 3] - t_min) / (t_max - t_min)

    return data


def argo_concat(data_path, save_path):
    filenames = glob.glob(os.path.join(data_path + "/*.csv"))

    data = pd.DataFrame()
    print('Loading', len(filenames), 'Argo data sets:')
    for filename in tqdm(filenames):
        raw_data = pd.read_csv(filename)
        data = pd.concat([data, raw_data])

    print('Saving the raw Argo data.')

    data.to_csv(save_path, index=False)
    # print(data.head(20))


def argo_filter(data_path, save_path, r_min, r_max, theta_min, theta_max, phi_min, phi_max, t_min, t_max):
    print('Filtering the Argo data.')

    raw_argo_data = pd.read_csv(data_path)
    argo_all = raw_argo_data[['PRES_ADJUSTED (decibar)', 'LATITUDE (degree_north)', 'LONGITUDE (degree_east)', 'DATE (YYYY-MM-DDTHH:MI:SSZ)',  'TEMP_ADJUSTED (degree_Celsius)', 'PSAL_ADJUSTED (psu)']].dropna()
    argo_all.columns = ['r', 'theta', 'phi', 't', 'temp', 'sal']


    argo_all['t'] = pd.to_datetime(argo_all['t']).dt.tz_convert(None)
    argo_all['t'] = argo_all['t'].values.astype(np.int64) / 10 ** 9 # Seconds from Jan 1st 1970

    argo_all['r'] = - argo_all['r']

    argo_all = argo_all.to_numpy()

    argo_all = data_filter(argo_all, r_min, r_max, theta_min, theta_max, phi_min, phi_max, t_min, t_max)
    # print(pd.DataFrame(argo_all), '\n', pd.DataFrame(argo_all).describe())

    np.save(save_path, argo_all)


def nc_convert_and_filter(nc_path, npy_save_path, r_min=None, r_max=None, theta_min=None, theta_max=None, phi_min=None, phi_max=None, t_min=None, t_max=None):

    start_date = datetime.datetime(1950, 1, 1, 0, 0)
    end_date = datetime.datetime(1970, 1, 1, 0, 0)
    time_difference = end_date - start_date
    seconds_1950_to_1970 = time_difference.total_seconds()


    dataset = nc.Dataset(nc_path)
    all_vars = list(dataset.variables.keys())

    # vo: northward velocity (v_theta)
    # uo: eastward velocity (v_phi)
    # wo: vertical velocity

    depth = dataset.variables['depth'][:]
    latitude = dataset.variables['latitude'][:]
    longitude = dataset.variables['longitude'][:]
    time = dataset.variables['time'][:]

    if len(all_vars) == 6:
        var_1 = dataset.variables[all_vars[4]][:]
        var_2 = dataset.variables[all_vars[5]][:]
        indices = np.where(~var_1.mask)
        var_1 = var_1[indices]
        var_2 = var_2[indices]
        result = np.column_stack((depth[indices[1]], latitude[indices[2]], longitude[indices[3]], time[indices[0]], var_1, var_2))
    elif len(all_vars) == 5:
        var = dataset.variables[all_vars[4]][:]
        indices = np.where(~var.mask)
        var = var[indices]
        result = np.column_stack((depth[indices[1]], latitude[indices[2]], longitude[indices[3]], time[indices[0]], var))

    result[:, 0] *= -1
    result[:, 3] = result[:, 3] * 3600 - seconds_1950_to_1970 # Seconds from 1st Jan 1970

    if any([r_min, r_max, theta_min, theta_max, phi_min, phi_max, t_min, t_max]):
        result = data_filter(result, r_min, r_max, theta_min, theta_max, phi_min, phi_max, t_min, t_max)
    # print(pd.DataFrame(result), '\n', pd.DataFrame(result).describe())

    result = result.data
    result = np.asarray(result)

    np.save(npy_save_path, result)


def nc_folder_convert_and_filter(folder_path, r_min=None, r_max=None, theta_min=None, theta_max=None, phi_min=None, phi_max=None, t_min=None, t_max=None):
    nc_paths = []
    npy_save_paths = []
    for file in os.listdir(folder_path):
        if file.endswith(".nc"):
            nc_paths.append(os.path.join(folder_path, file))
            npy_save_paths.append(os.path.join(folder_path, file[:-3] + '.npy'))

    for i in tqdm(range(len(nc_paths))):
        nc_convert_and_filter(nc_path=nc_paths[i], 
                            npy_save_path=npy_save_paths[i], 
                            r_min=r_min, r_max=r_max, 
                            theta_min=theta_min, theta_max=theta_max, 
                            phi_min=phi_min, phi_max=phi_max, 
                            t_min=t_min, t_max=t_max)

 
def argo_split(data_path, save_path, trian_vali_test=[8, 1, 1]):

    print('Spliting the Argo data.')

    df = np.load(data_path)

    sample = np.random.choice(df.shape[0], size=df.shape[0], replace=False)
    part1 = df[sample[:round(df.shape[0] / sum(trian_vali_test) * trian_vali_test[0])]]
    part2 = df[sample[round(df.shape[0] / sum(trian_vali_test) * trian_vali_test[0]):round(df.shape[0] / sum(trian_vali_test) * (trian_vali_test[0] + trian_vali_test[1]))]]
    part3 = df[sample[round(df.shape[0] / sum(trian_vali_test) * (trian_vali_test[0] + trian_vali_test[1])):]]

    np.save(os.path.join(save_path, "argo_train.npy"), part1)
    np.save(os.path.join(save_path, "argo_vali.npy"), part2)
    np.save(os.path.join(save_path, "argo_test.npy"), part3)


def currents_merge_and_split(data_path, save_path, trian_vali_test=[8, 1, 1], ratio=1):

    print('Merging and spliting the currents data.')

    for i in tqdm(['cur', 'wcur']):
        npy_folder = f'{data_path}/{i}'

        # Lists to store all parts for each file
        all_part1 = []
        all_part2 = []
        all_part3 = []

        # Iterate over all CSV files in folder
        for file_name in tqdm(os.listdir(npy_folder)):
            if file_name.endswith(".npy"):
                # Read CSV file
                file_path = os.path.join(npy_folder, file_name)
                df = np.load(file_path)
                
                # Split into three parts
                n = round(df.shape[0] * ratio)
                b_1 = round(n / sum(trian_vali_test) * trian_vali_test[0])
                b_2 = round(n / sum(trian_vali_test) * (trian_vali_test[0] + trian_vali_test[1]))

                sample = np.random.choice(df.shape[0], size=df.shape[0], replace=False)
                part1 = df[sample[:b_1]]
                part2 = df[sample[b_1:b_2]]
                part3 = df[sample[b_2:n]]
                
                # Append the three parts to the respective lists
                all_part1.append(part1)
                all_part2.append(part2)
                all_part3.append(part3)

        # Concatenate all parts for each file
        merged_part1 = np.concatenate(all_part1, axis=0)
        merged_part2 = np.concatenate(all_part2, axis=0)
        merged_part3 = np.concatenate(all_part3, axis=0)

        # Export as NPY files
        np.save(os.path.join(save_path, f"{i}_train.npy"), merged_part1)
        np.save(os.path.join(save_path, f"{i}_vali.npy"), merged_part2)
        np.save(os.path.join(save_path, f"{i}_test.npy"), merged_part3)


def load_data(argo_data_path, argo_save_path, currents_data_path, currents_save_path, r_min, r_max, theta_min, theta_max, phi_min, phi_max, t_min, t_max, trian_vali_test=[8, 1, 1], ratio=1):

    print('#' * 15, 'Data loading', '#' * 15)

    # ---------------------- Argo data -----------------------

    # Merge data from different Argo floats
    argo_concat(data_path=argo_data_path, 
                save_path=f'{argo_save_path}/raw_argo_data.csv')

    # Select relevant variables, filter the data and standardize them
    argo_filter(data_path=f'{argo_save_path}/raw_argo_data.csv', 
                save_path=f'{argo_save_path}/argo_data_filtered.npy', 
                r_min=r_min, r_max=r_max, 
                theta_min=theta_min, theta_max=theta_max, 
                phi_min=phi_min, phi_max=phi_max, 
                t_min=t_min, t_max=t_max)

    # Split data into training, validation and test sets.
    argo_split(data_path=f'{argo_save_path}/argo_data_filtered.npy', 
            save_path=argo_save_path, 
            trian_vali_test=trian_vali_test)


    # ---------------------- Currents data -----------------------

    # Convert NC files, filter the data and standardize them
    print('Coverting the Copernicus NC files and filtering the currents data.')

    for folder in ['cur', 'wcur']:
        folder_path = os.path.join(currents_data_path, folder)
        nc_folder_convert_and_filter(folder_path, r_min=r_min, r_max=r_max, 
                                    theta_min=theta_min, theta_max=theta_max, 
                                    phi_min=phi_min, phi_max=phi_max, 
                                    t_min=t_min, t_max=t_max)
        

    # Merge data fron different files and split them into training, validation and test sets.
    currents_merge_and_split(data_path=currents_data_path,
                            save_path=currents_save_path, 
                            trian_vali_test=trian_vali_test, 
                            ratio=ratio) 
        