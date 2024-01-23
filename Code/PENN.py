import os
import sys

file_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(file_path)


from DataNormalization import data_normalization
from DomainPoints import generate_domain_points
from ModelTraining import penn_training


def penn(data_path, r_min, r_max, theta_min, theta_max, phi_min, phi_max, t_min, t_max, r_num, theta_num, phi_num, batch_size, init_beta_tau, init_beta_sigma, num_domain, num_boundary, input_output_size, n_layers, activation, initializer, model_save_path_1, model_save_path_2, variable_save_path, save_period, resample_period_1, resample_period_2, num_iter_1, num_iter_2, optimizer, learning_rate_1, learning_rate_2, loss_weights_1, loss_weights_2):

    # # Generate sampling points
    # generate_domain_points(data_path, r_min, r_max, theta_min, theta_max, phi_min, phi_max, r_num, theta_num, phi_num)

    # # Data normalization
    # data_normalization(data_path, r_min, r_max, theta_min, theta_max, phi_min, phi_max, t_min, t_max)

    # PENN training
    penn_training(data_path, data_path, batch_size, init_beta_tau, init_beta_sigma, num_domain, num_boundary, input_output_size, n_layers, activation, initializer, model_save_path_1, model_save_path_2, variable_save_path, save_period, resample_period_1, resample_period_2, num_iter_1, num_iter_2, optimizer, learning_rate_1, learning_rate_2, loss_weights_1, loss_weights_2)


if __name__ == '__main__':
    ...