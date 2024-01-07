import pandas as pd
import numpy as np
from tqdm import tqdm
import concurrent.futures


print('##################### Data Sampling #########################')


# Load data from CSV files
data_path = 'WholeWorld_sample/Argo/'
df = pd.read_csv(data_path + 'argo_all.csv')

# Define number of bins and samples per bin
n_bins_r = 10
n_bins_theta = 10
n_bins_phi = 20
n_bins_t = 12
n_train_samples = 25000
n_val_samples = 3100
n_test_samples = 3100

# Calculate bin sizes
max_r = 0
min_r = -2000
max_theta = 90 / 180 * np.pi
min_theta = -90 / 180 * np.pi
max_phi = 180 / 180 * np.pi
min_phi = -180 / 180 * np.pi
max_t = 1
min_t = 0
r_bin_size = (max_r - min_r) / n_bins_r
theta_bin_size = (max_theta - min_theta) / n_bins_theta
phi_bin_size = (max_phi - min_phi) / n_bins_phi
t_bin_size = (max_t - min_t) / n_bins_t

# Add bin columns to data frame
df['r_bin'] = ((df['r'] - min_r) / r_bin_size).astype(int)
df['theta_bin'] = ((df['theta'] - min_theta) / theta_bin_size).astype(int)
df['phi_bin'] = 0
df.loc[df['phi'] > 0, 'phi_bin'] = ((np.pi - df.loc[df['phi'] > 0, 'phi']) / (10 / 180 * np.pi)).astype(int)
df.loc[df['phi'] < 0, 'phi_bin'] = ((df.loc[df['phi'] < 0, 'phi'] + np.pi) / (10 / 180 * np.pi)).astype(int) + 2
df['t_bin'] = ((df['t'] - min_t) / t_bin_size).astype(int)
# print(df['phi_bin'].value_counts())

def process_bin(r_bin, theta_bin, phi_bin, t_bin):
    # Get data for current bin
    df_bin = df[(df['r_bin'] == r_bin) & (df['theta_bin'] == theta_bin) & (df['phi_bin'] == phi_bin) & (df['t_bin'] == t_bin)]

    # Sample from current bin for testing, validation, and training
    if len(df_bin) >= (n_train_samples + n_val_samples + n_test_samples):
        df_test_bin = df_bin.sample(n=n_test_samples, random_state=42)
        df_bin = df_bin.drop(df_test_bin.index)
        df_val_bin = df_bin.sample(n=n_val_samples, random_state=42)
        df_bin = df_bin.drop(df_val_bin.index)
        df_train_bin = df_bin.sample(n=n_train_samples, random_state=42)
    elif len(df_bin) >= (n_val_samples + n_test_samples):
        df_test_bin = df_bin.sample(n=n_test_samples, random_state=42)
        df_bin = df_bin.drop(df_test_bin.index)
        df_val_bin = df_bin.sample(n=len(df_bin)-n_test_samples, random_state=42)
        df_bin = df_bin.drop(df_val_bin.index)
        df_train_bin = df_bin
    elif len(df_bin) >= (n_test_samples):
        df_test_bin = df_bin.sample(n=n_test_samples, random_state=42)
        df_val_bin = df_bin.drop(df_test_bin.index)
        df_train_bin = pd.DataFrame()
    else:
        df_test_bin = df_bin
        df_val_bin = pd.DataFrame()
        df_train_bin = pd.DataFrame()

    # Return sampled data
    return df_train_bin, df_val_bin, df_test_bin

# Initialize data frames for training, validation, and testing
df_train = pd.DataFrame()
df_val = pd.DataFrame()
df_test = pd.DataFrame()

# Create a thread pool with 32 workers
with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
    # Loop through bins and submit jobs to the thread pool
    futures = [executor.submit(process_bin, r_bin, theta_bin, phi_bin, t_bin) for r_bin in tqdm(range(n_bins_r)) for theta_bin in tqdm(range(n_bins_theta)) for phi_bin in range(n_bins_phi) for t_bin in range(n_bins_t)]

    # Get results from completed jobs and add them to respective data frames
    for future in concurrent.futures.as_completed(futures):
        df_train_bin, df_val_bin, df_test_bin = future.result()
        df_train = pd.concat([df_train, df_train_bin])
        df_val = pd.concat([df_val, df_val_bin])
        df_test = pd.concat([df_test, df_test_bin])

# Export data frames to CSV files
df_train = df_train.drop(['r_bin', 'theta_bin', 'phi_bin', 't_bin'], axis=1)
df_val = df_val.drop(['r_bin', 'theta_bin', 'phi_bin', 't_bin'], axis=1)
df_test = df_test.drop(['r_bin', 'theta_bin', 'phi_bin', 't_bin'], axis=1)
df_train.to_csv(data_path + 'argo_train.csv', index=False)
df_val.to_csv(data_path + 'argo_vali.csv', index=False)
df_test.to_csv(data_path + 'argo_test.csv', index=False)




# ----------------------------- Test pre ------------------------

# # Load data from CSV files
# df_pre = pd.read_csv('WholeWorld/argo_pre_all.csv')

# # Define number of bins and samples per bin
# n_bins_r = 10
# n_bins_theta = 10
# n_bins_phi = 20
# n_bins_t = 3
# n_pre_samples = 31

# # Calculate bin sizes
# max_r = 0
# min_r = -2000
# max_theta = 90 / 180 * np.pi
# min_theta = -90 / 180 * np.pi
# max_phi = 180 / 180 * np.pi
# min_phi = -180 / 180 * np.pi
# max_t = 1.125
# min_t = 1
# r_bin_size = (max_r - min_r) / n_bins_r
# theta_bin_size = (max_theta - min_theta) / n_bins_theta
# phi_bin_size = (max_phi - min_phi) / n_bins_phi
# t_bin_size = (max_t - min_t) / n_bins_t

# # Add bin columns to data frame
# df_pre['r_bin'] = ((df_pre['r'] - min_r) / r_bin_size).astype(int)
# df_pre['theta_bin'] = ((df_pre['theta'] - min_theta) / theta_bin_size).astype(int)
# df_pre['phi_bin'] = 0
# df_pre.loc[df_pre['phi'] > 0, 'phi_bin'] = ((np.pi - df_pre.loc[df_pre['phi'] > 0, 'phi']) / (10 / 180 * np.pi)).astype(int)
# df_pre.loc[df_pre['phi'] < 0, 'phi_bin'] = ((df_pre.loc[df_pre['phi'] < 0, 'phi'] + np.pi) / (10 / 180 * np.pi)).astype(int) + 2
# df_pre['t_bin'] = ((df_pre['t'] - min_t) / t_bin_size).astype(int)
# # print(df['phi_bin'].value_counts())

# def process_bin(r_bin, theta_bin, phi_bin, t_bin):
#     # Get data for current bin
#     df_bin = df_pre[(df_pre['r_bin'] == r_bin) & (df_pre['theta_bin'] == theta_bin) & (df_pre['phi_bin'] == phi_bin) & (df_pre['t_bin'] == t_bin)]

#     # Sample from current bin for testing, validation, and training
#     if len(df_bin) >= (n_pre_samples):
#         df_pre_bin = df_bin.sample(n=n_pre_samples, random_state=42)
#     else:
#         df_pre_bin = df_bin

#     # Return sampled data
#     return df_pre_bin

# # Initialize data frames for training, validation, and testing
# df_pre_sample = pd.DataFrame()

# # Create a thread pool with 32 workers
# with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
#     # Loop through bins and submit jobs to the thread pool
#     futures = [executor.submit(process_bin, r_bin, theta_bin, phi_bin, t_bin) for r_bin in tqdm(range(n_bins_r)) for theta_bin in tqdm(range(n_bins_theta)) for phi_bin in range(n_bins_phi) for t_bin in range(n_bins_t)]

#     # Get results from completed jobs and add them to respective data frames
#     for future in concurrent.futures.as_completed(futures):
#         df_pre_bin = future.result()
#         df_pre_sample = pd.concat([df_pre_sample, df_pre_bin])

# # Export data frames to CSV files
# df_pre_sample = df_pre_sample.drop(['r_bin', 'theta_bin', 'phi_bin', 't_bin'], axis=1)
# df_pre_sample.to_csv('WholeWorld/argo_test_pre.csv', index=False)