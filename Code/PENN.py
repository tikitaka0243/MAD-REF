from DataNormalization import data_normalization
from DomainPoints import generate_domain_points
from ModelTraining import penn_training


def penn(data_path, min_r, max_r, min_theta, max_theta, min_phi, max_phi, r_num, theta_num, phi_num, batch_size, init_beta_tau, init_beta_sigma, num_domain, num_boundary, input_output_size, n_layers, activation, initializer, model_save_path_1, model_save_path_2, variable_save_path, save_period, resample_period_1, resample_period_2, num_iter_1, num_iter_2, optimizer, learning_rate_1, learning_rate_2, loss_weights_1, loss_weights_2):

    # Generate sampling points
    generate_domain_points(save_path=data_path, 
                           min_r=min_r, max_r=max_r, 
                           min_theta=min_theta, max_theta=max_theta, 
                           min_phi=min_phi, max_phi=max_phi)

    # Data normalization
    data_normalization(data_path)

    # PENN training
    penn_training(data_path, domain_points_path=data_path, batch_size, init_beta_tau, init_beta_sigma, num_domain, num_boundary, input_output_size, n_layers, activation, initializer, model_save_path_1, model_save_path_2, variable_save_path, save_period, resample_period_1, resample_period_2, num_iter_1, num_iter_2, optimizer, learning_rate_1, learning_rate_2, loss_weights_1, loss_weights_2)
