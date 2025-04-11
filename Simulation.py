import numpy as np
import pandas as pd
from Code.Model import SimulationModel
from Code.Plot import SimulationPlot
from Code.utils import unpack_x_sim
from Code.DataProcess import SimulationData


######################### Generation #########################


ETA = 0.01
ZETA = 0.01
ZETA_TAU = 0.02

# -------------------- Data generation --------------------

# Generatie simulation data
# data_loader = SimulationData(eta = ETA, zeta = ZETA, zeta_tau = ZETA_TAU)
# data_loader.gen_simulate_data(num=1000, noise_std=0, simulation_data_path = 'Data/Simulation/Train')
# data_loader.gen_simulate_data(num=250, noise_std=0, simulation_data_path = 'Data/Simulation/Validation')

# Plot data
# plotter = SimulationPlot()
# plotter.simulation_data_plot(simulation_data_path = 'Data/Simulation/Train', save_path='Output/Plot/Simulation/Data/Train')
# plotter.simulation_data_plot(simulation_data_path = 'Data/Simulation/Validation', save_path='Output/Plot/Simulation/Data/Validation')

# -------------------- Coordinates generation --------------------

# Generate new coordinates for test
# simulation_gen_new_x(density=101, density_t=101, save_path='Data/simulation_new_x.npy')

# Generate new coordinates for plot
# simulation_gen_new_x(density=1001, density_t=5, save_path='Data/Simulation/simulation_new_x_plot.npy')

# -------------------- Solutions generation --------------------

# Generate solution for test
# new_x = np.load('Data/Simulation/Coordinates/simulation_new_x.npy')
# x, z, t = unpack_x_sim(new_x)
# taylor_green_vortex(x, z, t, eta = ETA, zeta = ZETA, zeta_tau = ZETA_TAU, save=True, save_path='Data/Simulation/Solutions/simulation_solution.npy')

# Generate solution for plot
# new_x_plot = np.load('Data/Simulation/Coordinates/simulation_new_x_plot.npy')
# x, z, t = unpack_x_sim(new_x_plot)
# taylor_green_vortex(x, z, t, eta = ETA, zeta = ZETA, zeta_tau = ZETA_TAU, save=True, save_path='Data/Simulation/Solutions/simulation_solution_plot.npy')

# Plot solution
# simulation_plot(
#     X_path='Data/Simulation/Coordinates/simulation_new_x_plot.npy', 
#     Y_path='Data/Simulation/Solutions/simulation_solution_plot.npy', 
#     save_path='Output/Plot/Simulation/Solution')


######################### MAD-REF net (Dynamic Loss Weight) #########################


ITERS = [5000, 100000]
TASK_NAME = f'2Steps_2Para_HardConAll_MetaMax'

# -------------------- Training --------------------

mdrf = SimulationModel(task_name=TASK_NAME, 
                       dynamic_loss_weight=False, 
                       Q=False, 
                       hard_constraint=True, 
                       meta_learning=True, 
                       output_dir=f'Output/Model/Simulation/{TASK_NAME}')
# mdrf.simulation_train(simulation_data_path='Data/Simulation', 
#                       iters=ITERS, 
#                       display=100, 
#                       save_period=1000, 
#                       lrs=[1e-4, 1e-5], 
#                       loss_weight=[1] * 4 + [1] * 3, 
#                       frozen=False)

# Train 10 times
# for i in tqdm(range(10)):
#     mdrf.simulation_train(simulation_data_path='Data/Simulation/Train', output_dir=f'Output/Model/Simulation/{TASK_NAME}', iters=ITERS, display=100, lrs=LRS, variable_file_name=f'variable_{i}.dat')

# -------------------- Prediction --------------------

# Prediction for plot
# mdrf.simulation_inference_steps(
#     model_num_1 = ITERS[0],
#     model_num_2 = ITERS[1], 
#     checkpoint=f'Output/Model/Simulation/{TASK_NAME}', 
#     new_x_path='Data/Simulation/Coordinates/simulation_new_x_plot.npy', 
#     save_path=f'Output/Prediction/Simulation/{TASK_NAME}/prediction_plot',
#     para=2)

# mdrf.rmse_trend()


# -------------------- Plot --------------------

mdrf_plot = SimulationPlot()

# Plot prediction
# mdrf_plot.simulation_plot_2steps(
#     X_path='Data/Simulation/Coordinates/simulation_new_x_plot.npy', 
#     Y_path=f'Output/Prediction/Simulation/{TASK_NAME}/prediction_plot', 
#     save_path=f'Output/Plot/Simulation/Prediction/{TASK_NAME}',
#     plot_step_1=True)

# simulation_plot_res(
#     X_path='Data/Simulation/Coordinates/simulation_new_x_plot.npy', 
#     pre_path=f'Output/Prediction/Simulation/{TASK_NAME}/prediction_plot_step2.npy', 
#     sol_path='Data/Simulation/Solutions/simulation_solution_plot.npy', 
#     save_path=f'Output/Plot/Simulation/Prediction/{TASK_NAME}Res')

# simulation_plot_parameters(model_dir=f'Output/Model/Simulation/{TASK_NAME}', save_dir='Output/Plot/Simulation/parameters.png')

# simulation_plot_rmse_2(
#     sol_path='Data/Simulation/Solutions/simulation_solution_plot.npy', 
#     mdrf_path=f'Output/Prediction/Simulation/{TASK_NAME}/prediction_plot_step2.npy',
#     nmdrf_path='Output/Prediction/Simulation/NoPhysics/prediction_plot.npy',
#     gpr_path='Output/Prediction/Simulation/GPR/prediction_plot.npy',
#     kriging_path='Output/Prediction/Simulation/Kriging/prediction_plot.npy',
#     save_path=f'Output/Plot/Simulation/simualtion_rmse_{TASK_NAME}.png')

# mdrf_plot.simlulation_plot_lambdas(file_path=f'Output/Model/Simulation/{TASK_NAME}/Step2/lambdas.csv', output_image_path=f'Output/Plot/Simulation/{TASK_NAME}_simualtion_lambdas.png')

mdrf_plot.plot_rmse_trends(tasks=['2Steps_2Para', 
                                  '2Steps_2Para_Meta', 
                                  '2Steps_2Para_HardConPeriodic', 
                                  '2Steps_2Para_HardConPeriodic_MetaMin'], 
                           task_names=['None', 
                                       'Meta Learning', 
                                       'Hard Constraint', 
                                       'Hard Constraint + Meta Learning'], 
                           output_path='Output/Plot/Simulation/rmse_trends_average.png',
                           average_tasks=True, 
)

######################### No Physics #########################

# -------------------- Training --------------------

# simulation_train_no_physics(simulation_data_path='Data/Simulation/Train', output_dir='Output/Model/Simulation/NoPhysics', iters=15000, display=100, lrs=1e-4)

# -------------------- Prediction --------------------

# Prediction for plot
# simulation_inference(
#     checkpoint='Output/Model/Simulation/NoPhysics/model-10000.pt', 
#     new_x_path='Data/Simulation/Coordinates/simulation_new_x_plot.npy', 
#     save_path='Output/Prediction/Simulation/NoPhysics/prediction_plot.npy')

# -------------------- Plot --------------------

# Plot prediction
# simulation_plot(
#     X_path='Data/Simulation/Coordinates/simulation_new_x_plot.npy', 
#     Y_path='Output/Prediction/Simulation/NoPhysics/prediction_plot.npy', 
#     save_path='Output/Plot/Simulation/Prediction/NoPhysics')

# simulation_plot_res(
#     X_path='Data/Simulation/Coordinates/simulation_new_x_plot.npy', 
#     pre_path='Output/Prediction/Simulation/NoPhysics/prediction_plot.npy', 
#     sol_path='Data/Simulation/Solutions/simulation_solution_plot.npy', 
#     save_path='Output/Plot/Simulation/Prediction/NoPhysicsRes')


######################### Kriging #########################

# -------------------- Prediction --------------------

# simulation_other_methods_inference(
#     simulation_data_path='Data/Simulation/Train', 
#     new_x_path='Data/Simulation/Coordinates/simulation_new_x_plot.npy', 
#     save_path='Output/Prediction/Simulation/Kriging/prediction_plot.npy',
#     model='kriging')

# -------------------- Plot --------------------

# Plot kriging results
# simulation_plot(X_path='Data/Simulation/Coordinates/simulation_new_x_plot.npy', Y_path='Output/Prediction/Simulation/Kriging/prediction_plot.npy', save_path='Output/Plot/Simulation/Prediction/Kriging')


######################### GPR #########################

# -------------------- Prediction --------------------

# Gaussian preocess regression
# simulation_other_methods_inference(
#     simulation_data_path='Data/Simulation/Train', 
#     new_x_path='Data/Simulation/Coordinates/simulation_new_x_plot.npy', 
#     save_path='Output/Prediction/Simulation/GPR/prediction_plot.npy',
#     model='gpr')

# -------------------- Plot --------------------

# Plot GPR
# simulation_plot(X_path='Data/Simulation/Coordinates/simulation_new_x_plot.npy', Y_path='Output/Prediction/Simulation/GPR/prediction_plot.npy', save_path='Output/Plot/Simulation/Prediction/GPR')


######################### Modelling #########################



# -------------------- Prediction --------------------

# Prediction for test
# simulation_inference(checkpoint='Output/Model/Simulation/Step2/model-30000.pt', new_x_path='Data/Simulation/simulation_new_x.npy', save_path='Output/Prediction/Simulation_2steps/prediction.npy')





# -------------------- Testing --------------------

# solution = np.load('Data/Simulation/simulation_solution.npy')
# prediction = np.load('Output/Prediction/Simulation_2steps/prediction.npy')
# print('Simulation RMSE:', np.sqrt(mean_squared_error(solution, prediction)))


######################### Plot #########################




