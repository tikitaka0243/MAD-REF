import pandas as pd


print('\n################# LoadData9 (Data Check) ####################')


pd.set_option('display.width', 1000)
pd.options.display.max_columns = 40

DATA_PATH = ['WholeWorld_sample/Argo/', 'WholeWorld_sample/Currents_m/currents_', 'Equator_all/Argo/', 'Equator_all/Currents_m/']
DF_NAME = [['argo_all', 'argo_train', 'argo_vali', 'argo_test', 'argo_test_pre'], ['cur_train', 'cur_vali', 'cur_test', 'cur_test_pre', 'wcur_train', 'wcur_vali', 'wcur_test', 'wcur_test_pre'], ['argo_test_pre', 'argo_test', 'argo_train', 'argo_vali'], ['cur_train', 'cur_vali', 'cur_test', 'cur_test_pre', 'wcur_train', 'wcur_vali', 'wcur_test', 'wcur_test_pre']]
scale = False

for i in [3, 4]:
    data_path = DATA_PATH[i]
    
    for df_name in DF_NAME[i]:
        if scale:
            df = pd.read_csv(data_path + df_name + '_scale.csv')
        else:
            df = pd.read_csv(data_path + df_name + '.csv')
        print('------------------- ' + df_name[df_name.rfind("/") + 1:] + ' -------------------\n', df)
        print(df.describe(), '\n')









