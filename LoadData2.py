from LoadData import argo_concat, argo_filter, currents_convert_and_filter, currents_merge_and_split, argo_split


# ---------------------- Argo data -----------------------

# # Merge data from different Argo floats
# argo_concat(data_path='Data/Argo/RawFiles', 
#             save_path='Data/Argo/raw_argo_data.csv')

# # Select relevant variables, filter the data and standardize them
# argo_filter(data_path='Data/Argo/raw_argo_data.csv', 
#             save_path='Data/Argo/argo_data_filtered.npy', 
#             min_r=-2000, max_r=0, 
#             min_theta=-10, max_theta=10, 
#             min_phi=-140, max_phi=-120, 
#             min_t='2021-01-01T00:00:00Z', max_t='2021-07-01T00:00:00Z')

# # Split data into training, validation and test sets.
# argo_split(data_path='Data/Argo/argo_data_filtered.npy', 
#            save_path='Data/Argo', 
#            ratio=[8, 1, 1])


# ---------------------- Currents data -----------------------

# # Convert NC files, filter the data and standardize them
# currents_convert_and_filter(data_path='Data/Currents/RawFiles', 
#                             save_path='Data/Currents/RawFiles', 
#                             min_r=-2000, max_r=0, 
#                             min_theta=-10, max_theta=10, 
#                             min_phi=-140, max_phi=-120, 
#                             min_t='2021-01-01T00:00:00Z', max_t='2021-07-01T00:00:00Z')

# Merge data fron different files and split them into training, validation and test sets.
currents_merge_and_split(data_path='Data/Currents/RawFiles',
                         save_path='Data/Currents', 
                         trian_vali_test=[8, 1, 1], 
                         ratio=0.18) 