import pandas as pd


# data_path = 'Equator_all/Argo/'
data_path = 'WholeWorld_all/Argo/'

df = pd.read_csv('WholeWorld_sample/Argo/argo_all.csv')
n = len(df)
bar_1 = round(n / 10 * 8)
bar_2 = round(n / 10 * 9)
df = df.sample(n = n, replace=False)
df_train = df[:bar_1]
df_vali = df[bar_1:bar_2]
df_test = df[bar_2:]
df_train.to_csv(data_path + 'argo_train.csv', index=False)
df_vali.to_csv(data_path + 'argo_vali.csv', index=False)
df_test.to_csv(data_path + 'argo_test.csv', index=False)
