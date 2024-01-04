import pandas as pd
import numpy as np
import concurrent.futures


pd.set_option('display.width', 1000)
pd.options.display.max_columns = 40


print('################## LoadData10 (Sampling) ###################')

file_name = 'argo_train_scale'
save_path = 'Sample_spatio/'
# file_name = 'Currents_reanalysis/currents_cur_vali_scale'
df = pd.read_csv(file_name + '.csv')
# data = data.iloc[:, :6]


for i in [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]:

    # Define number of bins and samples per bin
    n_bins = 10
    n_bins_theta = 4
    n_samples = round(len(df) * i / (n_bins ** 2 * n_bins_theta))

    # Calculate bin sizes
    max_r = 1
    min_r = 0
    max_theta = 20 / 180 * np.pi
    min_theta = -20 / 180 * np.pi
    max_phi = 50 / 180 * np.pi
    min_phi = -50 / 180 * np.pi
    r_bin_size = (max_r - min_r) / n_bins
    theta_bin_size = (max_theta - min_theta) / n_bins_theta
    phi_bin_size = (max_phi - min_phi) / n_bins

    # Add bin columns to data frame
    df['r_bin'] = ((df['r'] - min_r) / r_bin_size).astype(int)
    df['theta_bin'] = ((df['theta'] - min_theta) / theta_bin_size).astype(int)
    df['phi_bin'] = ((df['phi'] - min_phi) / phi_bin_size).astype(int)

    def process_bin(r_bin, theta_bin, phi_bin):
        # Get data for current bin
        df_bin = df[(df['r_bin'] == r_bin) & (df['theta_bin'] == theta_bin) & (df['phi_bin'] == phi_bin)]

        # Sample from current bin
        if len(df_bin) >= n_samples:
            df_sample_bin = df_bin.sample(n=n_samples, random_state=42)
        else:
            df_sample_bin = df_bin

        # Return sampled data
        return df_sample_bin

    # Initialize data frames for sampling
    df_sample = pd.DataFrame()

    # Create a thread pool with 8 workers
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        # Loop through bins and submit jobs to the thread pool
        futures = [executor.submit(process_bin, r_bin, theta_bin, phi_bin) for r_bin in range(n_bins) for theta_bin in range(n_bins_theta) for phi_bin in range(n_bins)]

        # Get results from completed jobs and add them to respective data frames
        for future in concurrent.futures.as_completed(futures):
            df_sample_bin = future.result()
            df_sample = pd.concat([df_sample, df_sample_bin])

    df_sample = df_sample.drop(['r_bin', 'theta_bin', 'phi_bin'], axis=1)
    df_sample.to_csv(save_path + file_name + '_' + str(i) + '.csv', index=False)

# for i in ['1e3', '2e3', '5e3', '1e4', '2e4', '5e4', '1e5', '2e5', '5e5', '1e6']:
#     data_sample = df.sample(n=int(float(i) * (3 / 20)), replace=False, random_state=1921)
#     data_sample.to_csv(file_name + '_' + i + '.csv')
#     print('data_sample (i=' + i + '):\n', data_sample, '\n', data_sample.describe())