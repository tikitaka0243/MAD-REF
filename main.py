from Code.LoadData import load_data
from Code.PENN import penn


load_data(argo_data_path='Data/Argo/RawFiles', 
          argo_save_path='Data/Argo', 
          currents_data_path='Data/Currents/RawFiles', 
          currents_save_path='Data/Currents', 
          min_r=-2000, max_r=0, 
          min_theta=-10, max_theta=10, 
          min_phi=-140, max_phi=-120, 
          min_t='2021-01-01T00:00:00Z', 
          max_t='2021-07-01T00:00:00Z', 
          trian_vali_test=[8, 1, 1], ratio=1)

penn(data_path, min_r, max_r, min_theta, max_theta, min_phi, max_phi, r_num, theta_num, phi_num, batch_size, init_beta_tau, init_beta_sigma, num_domain, num_boundary, input_output_size, n_layers, activation, initializer, model_save_path_1, model_save_path_2, variable_save_path, save_period, resample_period_1, resample_period_2, num_iter_1, num_iter_2, optimizer, learning_rate_1, learning_rate_2, loss_weights_1, loss_weights_2)