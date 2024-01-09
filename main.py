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

penn(data_path='Data', 
     min_r=-2000, max_r=0, 
     min_theta=-10, max_theta=10, 
     min_phi=-140, max_phi=-120, 
     min_t='2021-01-01T00:00:00Z', 
     max_t='2021-07-01T00:00:00Z', 
     r_num=81, theta_num=200, phi_num=200, 
     batch_size=512, init_beta_tau=0.0, init_beta_sigma=0.0, num_domain=512, num_boundary=128, 
     input_output_size=[4] + [6], n_layers=[3] * 2 + [4] * 4, 
     activation='tanh', initializer='Glorot uniform', 
     model_save_path_1='Data/Model1', model_save_path_2='Data/Model2', 
     variable_save_path='Data/variable.dat', 
     save_period=100, resample_period_1=10, resample_period_2=10, 
     num_iter_1=4000000, num_iter_2=4000000, 
     optimizer='adam', learning_rate_1=1e-4, learning_rate_2=1e-5, 
     loss_weights_1=[1, 1, 100, 10, 10], 
     loss_weights_2=[1, 1, 1, 1, 1, 1, 1, 1, 100, 10, 10])