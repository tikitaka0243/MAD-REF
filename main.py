from Code.LoadData import load_data
from Code.MDRF import mdrf_net
from Code.Simulation import gen_simulate_data, simulation_train, simulation_gen_new_x, simulation_inference, taylor_green_vortex
import numpy as np
from sklearn.metrics import mean_squared_error


# load_data(argo_data_path='Data/Argo/RawFiles', 
#           argo_save_path='Data/Argo', 
#           currents_data_path='Data/Currents/RawFiles', 
#           currents_save_path='Data/Currents', 
#           r_min=-2000, r_max=0, 
#           theta_min=-10, theta_max=10, 
#           phi_min=-140, phi_max=-120, 
#           t_min='2021-01-01T00:00:00Z', 
#           t_max='2021-07-01T00:00:00Z', 
#           train_vali_test=[8, 1, 1], ratio=0.18)

# mdrf_net(data_path='Data', 
#      r_min=-2000, r_max=0, 
     # theta_min=-10, theta_max=10, 
     # phi_min=-140, phi_max=-120, 
     # t_min='2021-01-01T00:00:00Z', 
     # t_max='2021-07-01T00:00:00Z', 
     # r_num=30, theta_num=30, phi_num=30, 
     # batch_size=512, init_beta_tau=0.0, init_beta_sigma=0.0, num_domain=512, num_boundary=128, 
     # input_output_size=[4] + [6], n_layers=[3] * 2 + [4] * 4, 
     # activation='tanh', initializer='Glorot uniform', 
     # model_save_path_1='Code/Model1', model_save_path_2='Code/Model2', 
     # variable_save_path='Data/variable.dat', 
     # save_period=100, resample_period_1=10, resample_period_2=10, 
     # num_iter_1=4000000, num_iter_2=4000000, 
     # optimizer='adam', learning_rate_1=1e-4, learning_rate_2=1e-5, 
     # loss_weights_1=[1, 1, 100, 10, 10], 
     # loss_weights_2=[1, 1, 1, 1, 1, 1, 1, 1, 100, 10, 10])


if __name__ == '__main__':
     pass
     
     
