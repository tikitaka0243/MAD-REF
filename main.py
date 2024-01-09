from Code.LoadData import load_data


load_data(argo_data_path='Data/Argo/RawFiles', 
          argo_save_path='Data/Argo', 
          currents_data_path='Data/Currents/RawFiles', 
          currents_save_path='Data/Currents', 
          min_r=-2000, max_r=0, 
          min_theta=-10, max_theta=10, 
          min_phi=-140, max_phi=-120, 
          min_t='2021-01-01T00:00:00Z', 
          max_t='2021-07-01T00:00:00Z', 
          trian_vali_test=[8, 1, 1], ratio=0.18)