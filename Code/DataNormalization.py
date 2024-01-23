import numpy as np
import pandas as pd
from tqdm import tqdm
import os


pd.set_option('display.width', 1000)
pd.options.display.max_columns = 40



# ----------- Normalization --------------

def coordinates_normalization(data, r_min, r_max, theta_min, theta_max, phi_min, phi_max, t_min=None, t_max=None):
    data_scale = np.copy(data)
    data_scale[:, 0] = (data[:, 0] - r_min) / (r_max - r_min)
    data_scale[:, 1] = (data[:, 1] - theta_min) / (theta_max - theta_min)
    data_scale[:, 2] = (data[:, 2] - phi_min) / (phi_max - phi_min)

    if (t_min is not None) and (t_max is not None):
        t_min = pd.to_datetime(t_min).value / 10 ** 9
        t_max = pd.to_datetime(t_max).value / 10 ** 9
        data_scale[:, 3] = (data[:, 3] - t_min) / (t_max - t_min)

    return data_scale


def data_normalization(data_path, r_min, r_max, theta_min, theta_max, phi_min, phi_max, t_min, t_max):

    print('Normalizing data.')

    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', data_path))

    # ------------- Argo ---------------

    # Argo
    argo_train = np.load(os.path.join(data_path, 'Argo/argo_train.npy'))

    # Coordinates
    argo_train_scale = coordinates_normalization(argo_train, r_min, r_max, theta_min, theta_max, phi_min, phi_max, t_min, t_max)

    # temp and sal
    for j in [4, 5]:
        argo_train_scale[:, j] = (argo_train[:, j] - np.mean(argo_train[:, j])) / np.std(argo_train[:, j])

    np.save(os.path.join(data_path, 'Argo/argo_train_scale.npy'), argo_train_scale)

    for df_name in ['vali', 'test']:
        df_argo = np.load(os.path.join(data_path, 'Argo/argo_' + df_name + '.npy'))

        df_argo_scale = coordinates_normalization(df_argo, r_min, r_max, theta_min, theta_max, phi_min, phi_max, t_min, t_max)

        np.save(os.path.join(data_path, 'Argo/argo_' + df_name + '_scale.npy'), df_argo_scale)


    # ---------------- Currents ---------------

    cur_train = np.load(os.path.join(data_path, 'Currents/cur_train.npy'))
    wcur_train = np.load(os.path.join(data_path, 'Currents/wcur_train.npy'))

    # Coordinates
    cur_train_scale = coordinates_normalization(cur_train, r_min, r_max, theta_min, theta_max, phi_min, phi_max, t_min, t_max)
    wcur_train_scale = coordinates_normalization(wcur_train, r_min, r_max, theta_min, theta_max, phi_min, phi_max, t_min, t_max)

    # uo: eastward velocity
    # vo: northward velocity
    # wo: vertical velocity

    # v_theta, v_phi
    for j in [4, 5]:
        cur_train_scale[:, j] = (cur_train[:, j] - np.mean(cur_train[:, j])) / np.std(cur_train[:, j])

    # w
    wcur_train_scale[:, 4] = (wcur_train[:, 4] - np.mean(wcur_train[:, 4])) / np.std(wcur_train[:, 4])

    np.save(os.path.join(data_path, 'Currents/cur_train_scale.npy'), cur_train_scale)
    np.save(os.path.join(data_path, 'Currents/wcur_train_scale.npy'), wcur_train_scale)


    for df_name in ['cur_test', 'cur_vali', 'wcur_test', 'wcur_vali']:
        df_currents = np.load(os.path.join(data_path, 'Currents/' + df_name + '.npy'))

        df_currents_scale = coordinates_normalization(df_currents, r_min, r_max, theta_min, theta_max, phi_min, phi_max, t_min, t_max)
        
        np.save(os.path.join(data_path, 'Currents/' + df_name + '_scale.npy'), df_currents_scale)


    # Mean and Standard Deviation
    train_mean_std = [
        [np.mean(argo_train[:, 4]), np.mean(argo_train[:, 5]), np.mean(wcur_train[:, 4]), np.mean(cur_train[:, 4]), np.mean(cur_train[:, 5])], 
        [np.std(argo_train[:, 4]), np.std(argo_train[:, 5]), np.std(wcur_train[:, 4]), np.std(cur_train[:, 4]), np.std(cur_train[:, 5])]
    ]
    columns = ['temp', 'sal', 'w', 'v_theta', 'v_phi']
    index = ['mean', 'std']

    train_mean_std = pd.DataFrame(data=train_mean_std, columns=columns, index=index)
    train_mean_std.to_csv(os.path.join(data_path, 'train_mean_std.csv'))