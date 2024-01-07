import pandas as pd
import numpy as np


print('################# filter the data #################')

save_path = 'Equator_all/Argo/'

for set in ['argo', 'argo_pre']:
    if set == 'argo':
        df_year4 = pd.read_csv('RawData/argo_year_4.csv')
        df_year5 = pd.read_csv('RawData/argo_year_5.csv')
        df = pd.concat([df_year4, df_year5])
    else:
        df = pd.read_csv('RawData/Argo_pre/argo_all.csv')
    
    t_0 = 1609459200
    t_range = 1672531200 - t_0
    df['t'] = (df['t'] - t_0) / t_range
    df = df[(df['r'] >= -2000) & (df['r'] <= 0)]
    df = df[(df['theta'] <= 20 / 180 * np.pi) & (df['theta'] >= -20 / 180 * np.pi)]
    df = df[(df['phi'] >= 150 / 180 * np.pi) | (df['phi'] <= -110 / 180 * np.pi)]
    pd.set_option('display.width', 1000)
    pd.options.display.max_columns = 40
    print(df.describe())
    df.to_csv(save_path + f'{set}_all.csv', index=False)
