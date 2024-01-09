import numpy as np
import pandas as pd
from tqdm import tqdm

import sys
sys.path.append()


pd.set_option('display.width', 1000)
pd.options.display.max_columns = 40



# ----------- Normalization --------------


def data_normalization(data_path):

    # ------------- Argo ---------------

    # Argo
    argo_train = np.load(data_path + 'Argo/argo_train.npy')

    # r
    argo_train_scale = np.copy(argo_train)
    argo_train_scale[:, 0] = (argo_train[:, 0] + 2000) / 2000

    # temp and sal
    for j in [4, 5]:
        argo_train_scale[:, j] = (argo_train[:, j] - np.mean(argo_train[:, j])) / np.std(argo_train[:, j])

    np.save(data_path + 'Argo/argo_train_scale.npy', argo_train_scale)

    for df_name in ['vali', 'test']:
        df_argo = np.load(data_path + 'Argo/argo_' + df_name + '.npy')
        df_argo_scale = np.copy(df_argo)
        df_argo_scale[:, 0] = (df_argo_scale[:, 0] + 2000) / 2000

        np.save(data_path + 'Argo/argo_' + df_name + '_scale.npy', df_argo_scale)


    # ---------------- Currents ---------------

    cur_train = np.load(data_path + 'Currents/cur_train.npy')
    wcur_train = np.load(data_path + 'Currents/wcur_train.npy')

    cur_train_scale = np.copy(cur_train)
    wcur_train_scale = np.copy(wcur_train)
    # uo: eastward velocity
    # vo: northward velocity
    # wo: vertical velocity

    cur_train_scale[:, 0] = (-cur_train[:, 0] + 2000) / 2000
    wcur_train_scale[:, 0] = (-wcur_train[:, 0] + 2000) / 2000

    for j in [1, 2]:
        cur_train_scale[:, j] = cur_train[:, j] / 180 * np.pi
        wcur_train_scale[:, j] = wcur_train[:, j] / 180 * np.pi

    for j in [4, 5]:
        cur_train_scale[:, j] = (cur_train[:, j] - np.mean(cur_train[:, j])) / np.std(cur_train[:, j])
    wcur_train_scale[:, 4] = (wcur_train[:, 4] - np.mean(wcur_train[:, 4])) / np.std(wcur_train[:, 4])

    np.save(data_path + 'Currents/cur_train_scale.npy', cur_train_scale)
    np.save(data_path + 'Currents/wcur_train_scale.npy', wcur_train_scale)


    for df_name in tqdm(['cur_test', 'cur_vali', 'wcur_test', 'wcur_vali']):
        df_currents = np.load(data_path + 'Currents/' + df_name + '.npy')

        df_currents_scale = np.copy(df_currents)
        
        df_currents_scale[:, 0] = (-df_currents[:, 0] + 2000) / 2000
        for j in [1, 2]:
            df_currents_scale[:, j] = df_currents[:, j] / 180 * np.pi
        
        np.save(data_path + 'Currents/' + df_name + '_scale.npy', df_currents_scale)


    # Mean and Standard Deviation
    train_mean_std = [
        [np.mean(argo_train[:, 4]), np.mean(argo_train[:, 5]), np.mean(wcur_train[:, 4]), np.mean(cur_train[:, 4]), np.mean(cur_train[:, 5])], 
        [np.std(argo_train[:, 4]), np.std(argo_train[:, 5]), np.std(wcur_train[:, 4]), np.std(cur_train[:, 4]), np.std(cur_train[:, 5])]
    ]
    columns = ['temp', 'sal', 'w', 'v_theta', 'v_phi']
    index = ['mean', 'std']

    train_mean_std = pd.DataFrame(data=train_mean_std, columns=columns, index=index)
    train_mean_std.to_csv(data_path + 'train_mean_std.csv')