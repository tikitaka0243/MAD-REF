import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

import sys
sys.path.append('/public/home/tianting/PINN_equator/')
import Functions


print('\n################ LoadData7 (Normalization) ################')


pd.set_option('display.width', 1000)
pd.options.display.max_columns = 40



# ----------- Normalization --------------

data_path = '/public/home/tianting/PINN_equator/Data/Equator_all/'
WW = False

# Argo
argo_train = pd.read_csv(data_path + 'Argo/argo_train.csv')

argo_train_scale = argo_train.iloc[:, :6]
argo_train_scale['r'] = (argo_train_scale['r'] + 2000) / 2000

if(not WW):
    print(1)
    argo_train_scale['phi'] = Functions.move_phi(argo_train_scale['phi'])

argo_train_scale['temp'] = (argo_train['temp'] - argo_train['temp'].mean()) / argo_train['temp'].std()
argo_train_scale['sal'] = (argo_train['sal'] - argo_train['sal'].mean()) / argo_train['sal'].std()




argo_train_scale = argo_train_scale.round(6)
argo_train_scale.to_csv(data_path + 'Argo/argo_train_scale.csv', index=False)

for df_name in ['vali', 'test', 'test_pre']:
    df_argo = pd.read_csv(data_path + 'Argo/argo_' + df_name + '.csv')
    df_argo_scale = df_argo.iloc[:, :6]
    df_argo_scale['r'] = (df_argo_scale['r'] + 2000) / 2000
    
    if(not WW):
        df_argo_scale['phi'] = Functions.move_phi(df_argo_scale['phi'])

    df_argo_scale = df_argo_scale.round(6)
    df_argo_scale.to_csv(data_path + 'Argo/argo_' + df_name + '_scale.csv', index=False)


# Currents
currents_cur_train = pd.read_csv(data_path + 'Currents_m/cur_train.csv')
currents_wcur_train = pd.read_csv(data_path + 'Currents_m/wcur_train.csv')

currents_cur_train_scale = currents_cur_train[['depth', 'latitude', 'longitude', 'time', 'vo', 'uo']]
currents_wcur_train_scale = currents_wcur_train[['depth', 'latitude', 'longitude', 'time', 'wo']]
# uo: eastward velocity
# vo: northward velocity
# wo: vertical velocity
currents_cur_train_scale.columns = ['r', 'theta', 'phi', 't', 'v_theta', 'v_phi']
currents_wcur_train_scale.columns = ['r', 'theta', 'phi', 't', 'w']

currents_cur_train_scale['r'] = (-currents_cur_train_scale['r'] + 2000) / 2000
currents_wcur_train_scale['r'] = (-currents_wcur_train_scale['r'] + 2000) / 2000
currents_cur_train_scale.iloc[:, 1] = currents_cur_train_scale.iloc[:, 1] / 180 * np.pi
currents_wcur_train_scale.iloc[:, 1] = currents_wcur_train_scale.iloc[:, 1] / 180 * np.pi
currents_cur_train_scale['phi'] = currents_cur_train_scale['phi'] / 180 * np.pi
currents_wcur_train_scale['phi'] = currents_wcur_train_scale['phi'] / 180 * np.pi

if(not WW):
    currents_cur_train_scale['phi'] = Functions.move_phi(currents_cur_train_scale['phi'])
    currents_wcur_train_scale['phi'] = Functions.move_phi(currents_wcur_train_scale['phi'])


currents_cur_train_scale['v_theta'] = (currents_cur_train_scale['v_theta'] - currents_cur_train_scale['v_theta'].mean()) / currents_cur_train_scale['v_theta'].std()
currents_cur_train_scale['v_phi'] = (currents_cur_train_scale['v_phi'] - currents_cur_train_scale['v_phi'].mean()) / currents_cur_train_scale['v_phi'].std()
currents_wcur_train_scale['w'] = (currents_wcur_train_scale['w'] - currents_wcur_train_scale['w'].mean()) / currents_wcur_train_scale['w'].std()

currents_cur_train_scale = currents_cur_train_scale.round(6)
currents_wcur_train_scale = currents_wcur_train_scale.round(6)
currents_cur_train_scale.to_csv(data_path + 'Currents_m/cur_train_scale.csv', index=False)
currents_wcur_train_scale.to_csv(data_path + 'Currents_m/wcur_train_scale.csv', index=False)

for df_name in tqdm(['cur_test', 'cur_vali', 'cur_test_pre', 'wcur_test', 'wcur_vali', 'wcur_test_pre']):
    df_currents = pd.read_csv(data_path + 'Currents_m/' + df_name + '.csv')
    if df_currents.shape[1] == 6:
        df_currents_scale = df_currents[['depth', 'latitude', 'longitude', 'time', 'vo', 'uo']]
        df_currents_scale.columns = ['r', 'theta', 'phi', 't', 'v_theta', 'v_phi']
    else:
        df_currents_scale = df_currents[['depth', 'latitude', 'longitude', 'time', 'wo']]
        df_currents_scale.columns = ['r', 'theta', 'phi', 't', 'w']
    
    df_currents_scale['r'] = (-df_currents_scale['r'] + 2000) / 2000
    df_currents_scale.iloc[:, 1] = df_currents_scale.iloc[:, 1] / 180 * np.pi
    df_currents_scale['phi'] = df_currents_scale['phi'] / 180 * np.pi
    
    if(not WW):
        df_currents_scale['phi'] = Functions.move_phi(df_currents_scale['phi'])
    
    df_currents_scale = df_currents_scale.round(6)
    df_currents_scale.to_csv(data_path + 'Currents_m/' + df_name + '_scale.csv', index=False)


# Mean and Standard Deviation
train_mean_std = [
    [argo_train['temp'].mean(), argo_train['sal'].mean(), currents_wcur_train['wo'].mean(), currents_cur_train['vo'].mean(), currents_cur_train['uo'].mean()], 
    [argo_train['temp'].std(), argo_train['sal'].std(), currents_wcur_train['wo'].std(), currents_cur_train['vo'].std(), currents_cur_train['uo'].std()]
]
columns = ['temp', 'sal', 'w', 'v_theta', 'v_phi']
index = ['mean', 'std']

train_mean_std = pd.DataFrame(data=train_mean_std, columns=columns, index=index)
train_mean_std.to_csv(data_path + 'train_mean_std.csv')




# ----------------- Old ------------------

# pd.set_option('display.width', 1000)
# pd.options.display.max_columns = 40


# timestamp = [1514764800, 1546300800, 1577836800, 1609459200, 1640995200, 1672531200]
# t_range = timestamp[5] - timestamp[0]
# r_earth = 6371000


# # Train data
# data_train = pd.read_csv('argo_train.csv')
# domain_points_3d = pd.read_csv('domain_points_3d.csv')
# boundary_points_3d = pd.read_csv('boundary_points_3d.csv')
# coordinate = pd.concat([data_train.iloc[:, :3], domain_points_3d, boundary_points_3d], axis=0)
# t_temp_sal = data_train.iloc[:, 3:6]

# coordinate_scale = (coordinate - coordinate.mean()) / coordinate.std()
# t_temp_sal_scale = (t_temp_sal - t_temp_sal.mean()) / t_temp_sal.std()

# data_train_scale = pd.concat([coordinate_scale.iloc[:data_train.shape[0], :], t_temp_sal_scale], axis=1)
# domain_points_scale = coordinate_scale.iloc[data_train.shape[0] : (data_train.shape[0] + domain_points_3d.shape[0]), :]
# boundary_points_scale = coordinate_scale.iloc[(data_train.shape[0] + domain_points_3d.shape[0]):, :]

# print('data_train_scale:\n', data_train_scale)
# print(data_train_scale.describe())
# print('domain_points_scale:\n', domain_points_scale)
# print(domain_points_scale.describe())
# print('boundary_points_scale:\n', boundary_points_scale)
# print(boundary_points_scale.describe())
# data_train_scale.to_csv('argo_train_scale.csv', index=False)
# domain_points_scale.to_csv('domain_points_scale.csv', index=False)
# boundary_points_scale.to_csv('boundary_points_scale.csv', index=False)


# # Test and validation data
# data_vali = pd.read_csv('argo_vali.csv')
# data_test = pd.read_csv('argo_test.csv')

# coordinate_vali = data_vali.iloc[:, :3]
# t_temp_sal_vali = data_vali.iloc[:, 3:6]
# coordinate_test = data_test.iloc[:, :3]
# t_temp_sal_test = data_test.iloc[:, 3:6]

# coordinate_vali_scale = (coordinate_vali - coordinate.mean()) / coordinate.std()
# t_temp_sal_vali_scale = (t_temp_sal_vali - t_temp_sal.mean()) / t_temp_sal.std()
# coordinate_test_scale = (coordinate_test - coordinate.mean()) / coordinate.std()
# t_temp_sal_test_scale = (t_temp_sal_test - t_temp_sal.mean()) / t_temp_sal.std()

# data_vali_scale = pd.concat([coordinate_vali_scale, t_temp_sal_vali_scale], axis=1)
# data_test_scale = pd.concat([coordinate_test_scale, t_temp_sal_test_scale], axis=1)

# print('data_vali_scale:\n', data_vali_scale)
# print(data_vali_scale.describe())
# print('data_test_scale:\n', data_test_scale)
# print(data_test_scale.describe())
# data_vali_scale.to_csv('argo_vali_scale.csv', index=False)
# data_vali_scale.to_csv('argo_test_scale.csv', index=False)

# # Pridiction Data
# data_pre = pd.read_csv('mesh_pre.csv')
# coordinate_pre = data_pre.iloc[:, :3]
# t_temp_sal_pre = data_pre.iloc[:, 3:6]
# coordinate_pre_scale = (coordinate_pre - coordinate.mean()) / coordinate.std()
# t_temp_sal_pre_scale = (t_temp_sal_pre - t_temp_sal['t'].mean()) / t_temp_sal['t'].std()
# data_pre_scale = pd.concat([coordinate_pre_scale, t_temp_sal_pre_scale], axis=1)
# print('mesh_pre_scale:\n', data_pre_scale)
# print(data_pre_scale.describe())
# data_pre_scale.to_csv('mesh_pre_scale.csv', index=False)


# # Save the mean and standard deviation
# coordinate_mean = coordinate.mean().to_frame().transpose()
# coordinate_std = coordinate.std().to_frame().transpose()
# t_temp_sal_mean = t_temp_sal.mean().to_frame().transpose()
# t_temp_sal_std = t_temp_sal.std().to_frame().transpose()

# train_mean = pd.concat([coordinate_mean, t_temp_sal_mean], axis=1)
# train_std = pd.concat([coordinate_std, t_temp_sal_std], axis=1)

# train_mean_std = pd.concat([train_mean, train_std], axis=0)
# train_mean_std.index = ['mean', 'std']
# train_mean_std.columns = ['r', 'theta', 'phi', 't', 'temp', 'sal']

# print('train_mean_std:\n', train_mean_std)
# train_mean_std.to_csv('train_mean_std.csv', index=False)
