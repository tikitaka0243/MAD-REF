import glob
import pandas as pd
from tqdm import *
import os


# path: argo data path
def data_concat(path):
    filenames = glob.glob(path + "/*.csv")

    data = pd.DataFrame()
    print('Loading', len(filenames), 'Argo data sets:')
    for filename in tqdm(filenames):
        raw_data = pd.read_csv(filename)
        data = pd.concat([data, raw_data])
    print('Saving the raw data(it takes time)...')

    current_dir = os.getcwd()
    save_path = os.path.join(current_dir, '..', 'Argo/raw_argo_data.csv')
    data.to_csv(save_path, index=False)
    print(data.head(20))
