import pandas as pd
import numpy as np
import shutil, os


pd.set_option('display.width', 1000)
pd.options.display.max_columns = 40


print('#################### LoadData11 - Comparison Dataset ####################')

def generate_comparison_dataset(i_):
    if not os.path.exists('ComparisonDataset/' + str(i_)):
        os.makedirs('ComparisonDataset/' + str(i_))
        
    for df_name in ['argo', 'cur', 'wcur']:
        for df_type in ['train', 'vali', 'test', 'test_pre']: 
            if df_name != 'argo':
                df = pd.read_csv('Currents_reanalysis/currents_' + df_name + '_' + df_type + '_scale.csv')
            else:
                df = pd.read_csv('argo_' + df_type + '_scale.csv')


            r = 0.8
            u_theta = df['theta'].max() * r
            l_theta = df['theta'].min() * r
            u_phi = df['phi'].max() * r
            l_phi = df['phi'].min() * r

            r_ = 0.2
            u_theta_ = df['theta'].max() * r_
            l_theta_ = df['theta'].min() * r_
            u_phi_ = df['phi'].max() * r_
            l_phi_ = df['phi'].min() * r_

            if df_type == 'test':
                df_c = df.loc[(df['theta'] <= u_theta) &
                            (df['theta'] >= l_theta) & 
                            (df['phi'] <= u_phi) & 
                            (df['phi'] >= l_phi)]
                df_test = df_c.loc[(df_c['theta'] > u_theta_) |
                                (df_c['theta'] < l_theta_) | 
                                (df_c['phi'] > u_phi_) | 
                                (df_c['phi'] < l_phi_)]
                df_test_outside = df.loc[(df['theta'] > u_theta) |
                                        (df['theta'] < l_theta) | 
                                        (df['phi'] > u_phi) |
                                        (df['phi'] < l_phi)]
                df_test_inside = df.loc[(df['theta'] <= u_theta_) &
                                        (df['theta'] >= l_theta_) & 
                                        (df['phi'] <= u_phi_) & 
                                        (df['phi'] >= l_phi_)]
                
                df_test.to_csv('ComparisonDataset/' + str(i_) + '/' + df_name + '_test.csv', index=False)
                df_test_outside.to_csv('ComparisonDataset/' + str(i_) + '/' + df_name + '_test_outside.csv', index=False)
                df_test_inside.to_csv('ComparisonDataset/' + str(i_) + '/' + df_name + '_test_inside.csv', index=False)
                
            else:
                df_c = df.loc[(df['theta'] <= u_theta) &
                            (df['theta'] >= l_theta) & 
                            (df['phi'] <= u_phi) & 
                            (df['phi'] >= l_phi)]
                df_c = df_c.loc[(df_c['theta'] > u_theta_) |
                                (df_c['theta'] < l_theta_) | 
                                (df_c['phi'] > u_phi_) | 
                                (df_c['phi'] < l_phi_)]

                if df_type == 'test_pre':
                    df_c.to_csv('ComparisonDataset/' + str(i_) + '/' + df_name + '_test_pre.csv', index=False)
                else:
                    for i in [100, 200, 500, 1000, 2000, 5000, 10000, 20000]:
                        i_ori = i
                        if df_type == 'vali':
                            i =  round(i / 8)
                        df_c_sample = df_c.sample(i, replace=False)
                        df_c_sample.to_csv('ComparisonDataset/' + str(i_) + '/' + df_name + '_' + df_type + '_' + str(i_ori) + '.csv', index=False)
        

for i in range(5):
    generate_comparison_dataset(i)