from Code.Simulation import gen_simulate_data, simulation_train, simulation_gen_new_x, simulation_inference, taylor_green_vortex, simulation_plot, unpack_x
import numpy as np
from sklearn.metrics import mean_squared_error


# gen_simulate_data(num=10000, noise_std=0, simulation_data_path = 'Data/Simulation/Train')

# simulation_train(simulation_data_path='Data/Simulation/Train', output_dir='Output/Model/Simulation')

# simulation_gen_new_x(density=101, density_t=101, save_path='Data/simulation_new_x.npy')

# simulation_inference(checkpoint='Output/Model/Simulation/Step2/model-30000.pt', new_x_path='Data/Simulation/simulation_new_x.npy', save_path='Output/Prediction/Simulation_2steps/predcition.npy')

# new_x = np.load('Data/Simulation/simulation_new_x.npy')
# x, z, t = unpack_x(new_x)
# taylor_green_vortex(x, z, t, eta = 0.01, zeta = 0.01, eta_tau = 0.01, zeta_tau = 0.01, save=True, save_path='Data/Simulation/simulation_solution.npy')

# new_x_plot = np.load('Data/Simulation/simulation_new_x_plot.npy')
# x, z, t = unpack_x(new_x_plot)
# taylor_green_vortex(x, z, t, eta = 0.01, zeta = 0.01, eta_tau = 0.01, zeta_tau = 0.01, save=True, save_path='Data/Simulation/simulation_solution_plot.npy')

# solution = np.load('Data/Simulation/simulation_solution.npy')
# prediction = np.load('Output/Prediction/Simulation_2steps/predcition.npy')
# print('Simulation RMSE:', np.sqrt(mean_squared_error(solution, prediction)))

# simulation_gen_new_x(density=1001, density_t=5, save_path='Data/Simulation/simulation_new_x_plot.npy')

# simulation_inference(
#     checkpoint='Output/Model/Simulation/Step2/model-30000.pt', 
#     new_x_path='Data/Simulation/simulation_new_x_plot.npy', 
#     save_path='Output/Prediction/Simulation_2steps/predcition_plot.npy')

# simulation_plot(
#     X_path='Data/Simulation/simulation_new_x_plot.npy', 
#     Y_path='Output/Prediction/Simulation/predcition_plot.npy', 
#     save_path='Output/Plot/Simulation/Prediction')

simulation_plot(
    X_path='Data/Simulation/simulation_new_x_plot.npy', 
    Y_path='Data/Simulation/simulation_solution_plot.npy', 
    save_path='Output/Plot/Simulation/Solution')