from Code.LoadData import argo_concat, argo_filter, argo_split, nc_folder_convert_and_filter, currents_merge_and_split
import os
import argparse


def load_data(argo_data_path, argo_save_path, currents_data_path, currents_save_path, r_min, r_max, theta_min, theta_max, phi_min, phi_max, t_min, t_max, train_vali_test=[8, 1, 1], ratio=1):

    print('#' * 15, 'Data loading', '#' * 15)

    # ---------------------- Argo data -----------------------

    # Merge data from different Argo floats
    argo_concat(data_path=argo_data_path, 
                save_path=f'{argo_save_path}/raw_argo_data.csv')

    # Select relevant variables, filter the data and standardize them
    argo_filter(data_path=f'{argo_save_path}/raw_argo_data.csv', 
                save_path=f'{argo_save_path}/argo_data_filtered.npy', 
                r_min=r_min, r_max=r_max, 
                theta_min=theta_min, theta_max=theta_max, 
                phi_min=phi_min, phi_max=phi_max, 
                t_min=t_min, t_max=t_max)

    # Split data into training, validation and test sets.
    argo_split(data_path=f'{argo_save_path}/argo_data_filtered.npy', 
            save_path=argo_save_path, 
            train_vali_test=train_vali_test)


    # ---------------------- Currents data -----------------------

    # Convert NC files, filter the data and standardize them
    print('Coverting the Copernicus NC files and filtering the currents data.')

    for folder in ['cur', 'wcur']:
        folder_path = os.path.join(currents_data_path, folder)
        nc_folder_convert_and_filter(folder_path, r_min=r_min, r_max=r_max, 
                                    theta_min=theta_min, theta_max=theta_max, 
                                    phi_min=phi_min, phi_max=phi_max, 
                                    t_min=t_min, t_max=t_max)
        

    # Merge data fron different files and split them into training, validation and test sets.
    currents_merge_and_split(data_path=currents_data_path,
                            save_path=currents_save_path, 
                            train_vali_test=train_vali_test, 
                            ratio=ratio) 


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="LOAD_DATA Configuration")

    parser.add_argument("--argo_data_path", type=str, default='Data/Argo/RawFiles', help="Path to Argo raw data files")
    parser.add_argument("--argo_save_path", type=str, default='Data/Argo', help="Path to save processed Argo data")
    parser.add_argument("--currents_data_path", type=str, default='Data/Currents/RawFiles', help="Path to currents raw data files")
    parser.add_argument("--currents_save_path", type=str, default='Data/Currents', help="Path to save processed currents data")
    parser.add_argument("--r_min", type=int, default=-2000, help="Minimum r value")
    parser.add_argument("--r_max", type=int, default=0, help="Maximum r value")
    parser.add_argument("--theta_min", type=int, default=-10, help="Minimum theta value")
    parser.add_argument("--theta_max", type=int, default=10, help="Maximum theta value")
    parser.add_argument("--phi_min", type=int, default=-140, help="Minimum phi value")
    parser.add_argument("--phi_max", type=int, default=-120, help="Maximum phi value")
    parser.add_argument("--t_min", type=str, default='2021-01-01T00:00:00Z', help="Start time")
    parser.add_argument("--t_max", type=str, default='2021-07-01T00:00:00Z', help="End time")
    parser.add_argument("--train_vali_test", type=lambda s: [int(item) for item in s.split(':')], default=[8, 1, 1], help="Split ratio for training, validation, and testing as colon-separated integers")
    parser.add_argument("--ratio", type=float, default=0.18, help="Ratio for subsampling")

    args = parser.parse_args()
    load_data(**vars(args))