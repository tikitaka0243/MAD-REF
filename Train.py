import os
import argparse
from Code.DataNormalization import data_normalization
from Code.DomainPoints import generate_domain_points
from Code.ModelTraining import mdrf_net_training


def mdrf_net(data_path, r_min, r_max, theta_min, theta_max, phi_min, phi_max, t_min, t_max, r_num, theta_num, phi_num, batch_size, init_beta_tau, init_beta_sigma, num_domain, num_boundary, input_output_size, n_layers, activation, initializer, model_save_path_1, model_save_path_2, variable_save_path, save_period, resample_period_1, resample_period_2, num_iter_1, num_iter_2, optimizer, learning_rate_1, learning_rate_2, loss_weights_1, loss_weights_2):

    # # Generate sampling points
    generate_domain_points(os.path.join(data_path, 'domain_points.npy'), r_min, r_max, theta_min, theta_max, phi_min, phi_max, r_num, theta_num, phi_num)

    # Data normalization
    data_normalization(data_path, r_min, r_max, theta_min, theta_max, phi_min, phi_max, t_min, t_max)

    # MDRF-Net training
    mdrf_net_training(data_path, data_path, batch_size, init_beta_tau, init_beta_sigma, num_domain, num_boundary, input_output_size, n_layers, activation, initializer, model_save_path_1, model_save_path_2, variable_save_path, save_period, resample_period_1, resample_period_2, num_iter_1, num_iter_2, optimizer, learning_rate_1, learning_rate_2, loss_weights_1, loss_weights_2)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="MDRF_NET Configuration")

    parser.add_argument("--data_path", type=str, default='Data', help="Path to the data")
    parser.add_argument("--r_min", type=int, default=-2000, help="Minimum r value")
    parser.add_argument("--r_max", type=int, default=0, help="Maximum r value")
    parser.add_argument("--theta_min", type=int, default=-10, help="Minimum theta value")
    parser.add_argument("--theta_max", type=int, default=10, help="Maximum theta value")
    parser.add_argument("--phi_min", type=int, default=-140, help="Minimum phi value")
    parser.add_argument("--phi_max", type=int, default=-120, help="Maximum phi value")
    parser.add_argument("--t_min", type=str, default='2021-01-01T00:00:00Z', help="Start time")
    parser.add_argument("--t_max", type=str, default='2021-07-01T00:00:00Z', help="End time")
    parser.add_argument("--r_num", type=int, default=30, help="Number of r points")
    parser.add_argument("--theta_num", type=int, default=30, help="Number of theta points")
    parser.add_argument("--phi_num", type=int, default=30, help="Number of phi points")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size")
    parser.add_argument("--init_beta_tau", type=float, default=0.0, help="Initial beta tau value")
    parser.add_argument("--init_beta_sigma", type=float, default=0.0, help="Initial beta sigma value")
    parser.add_argument("--num_domain", type=int, default=512, help="Number of domain points")
    parser.add_argument("--num_boundary", type=int, default=128, help="Number of boundary points")
    parser.add_argument("--input_output_size", type=lambda s: [int(item) for item in s.split(',')], default=[4] + [6], help="Input and output sizes as comma-separated integers")
    parser.add_argument("--n_layers", type=lambda s: [int(item) for item in s.split(',')], default=[3]*2 + [4]*4, help="Number of layers for each part as comma-separated integers")
    parser.add_argument("--activation", type=str, default='tanh', help="Activation function")
    parser.add_argument("--initializer", type=str, default='Glorot uniform', help="Weight initializer")
    parser.add_argument("--model_save_path_1", type=str, default='Code/Model1', help="Save path for step 1")
    parser.add_argument("--model_save_path_2", type=str, default='Code/Model2', help="Save path for step 2")
    parser.add_argument("--variable_save_path", type=str, default='Data/variable.dat', help="Variable save path")
    parser.add_argument("--save_period", type=int, default=100, help="Save period")
    parser.add_argument("--resample_period_1", type=int, default=10, help="Resample period for step 1")
    parser.add_argument("--resample_period_2", type=int, default=10, help="Resample period for step 2")
    parser.add_argument("--num_iter_1", type=int, default=4000000, help="Number of iterations for step 1")
    parser.add_argument("--num_iter_2", type=int, default=4000000, help="Number of iterations for step 2")
    parser.add_argument("--optimizer", type=str, default='adam', help="Optimizer algorithm")
    parser.add_argument("--learning_rate_1", type=float, default=1e-4, help="Learning rate for step 1")
    parser.add_argument("--learning_rate_2", type=float, default=1e-5, help="Learning rate for step 2")
    parser.add_argument("--loss_weights_1", type=lambda s: [float(item) for item in s.split(',')], default=[1, 1, 100, 10, 10], help="Loss weights for step 1 as comma-separated floats")
    parser.add_argument("--loss_weights_2", type=lambda s: [float(item) for item in s.split(',')], default=[1, 1, 1, 1, 1, 1, 1, 1, 100, 10, 10], help="Loss weights for step 2 as comma-separated floats")

    args = parser.parse_args()

    mdrf_net(**vars(args))