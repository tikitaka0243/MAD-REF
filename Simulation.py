from Code.Simulation import gen_simulate_data, simulation_train, simulation_gen_new_x, simulation_inference, taylor_green_vortex, simulation_plot, unpack_x, simulation_data_plot, simulation_plot_2steps, simulation_inference_steps, simulation_other_methods_inference, simulation_train_no_physics, simulation_plot_res, simulation_plot_parameters, simulation_plot_rmse_2
import numpy as np
from tqdm import tqdm


def simulation():

    ######################### Generation #########################

    ETA = 0.01
    ZETA = 0.01
    ZETA_TAU = 0.02

    # -------------------- Data generation --------------------

    # Generatie simulation data
    gen_simulate_data(num=1000, noise_std=0, simulation_data_path = 'Data/Simulation/Train', eta = ETA, zeta = ZETA, zeta_tau = ZETA_TAU)

    # Plot data
    simulation_data_plot(simulation_data_path = 'Data/Simulation/Train', save_path='Output/Plot/Simulation/Data')

    # -------------------- Coordinates generation --------------------

    # Generate new coordinates for test
    simulation_gen_new_x(density=101, density_t=101, save_path='Data/simulation_new_x.npy')

    # Generate new coordinates for plot
    simulation_gen_new_x(density=1001, density_t=5, save_path='Data/Simulation/simulation_new_x_plot.npy')

    # -------------------- Solutions generation --------------------

    # Generate solution for test
    new_x = np.load('Data/Simulation/simulation_new_x.npy')
    x, z, t = unpack_x(new_x)
    taylor_green_vortex(x, z, t, eta = ETA, zeta = ZETA, zeta_tau = ZETA_TAU, save=True, save_path='Data/Simulation/simulation_solution.npy')

    # Generate solution for plot
    new_x_plot = np.load('Data/Simulation/Coordinates/simulation_new_x_plot.npy')
    x, z, t = unpack_x(new_x_plot)
    taylor_green_vortex(x, z, t, eta = ETA, zeta = ZETA, zeta_tau = ZETA_TAU, save=True, save_path='Data/Simulation/Solutions/simulation_solution_plot.npy')

    # Plot solution
    simulation_plot(
        X_path='Data/Simulation/Coordinates/simulation_new_x_plot.npy', 
        Y_path='Data/Simulation/Solutions/simulation_solution_plot.npy', 
        save_path='Output/Plot/Simulation/Solution')


    ######################### MDRF-Net #########################

    ITERS = [5000, 100000]
    LRS = [1e-4, 1e-5]
    TASK_NAME = '2Steps2Para'
    Q = False

    # -------------------- Training --------------------

    simulation_train(simulation_data_path='Data/Simulation/Train', output_dir=f'Output/Model/Simulation/{TASK_NAME}', iters=ITERS, display=100, lrs=LRS, Q=Q)

    # Train 10 times
    for i in tqdm(range(10)):
        simulation_train(simulation_data_path='Data/Simulation/Train', output_dir=f'Output/Model/Simulation/{TASK_NAME}', iters=ITERS, display=100, lrs=LRS, variable_file_name=f'variable_{i}.dat')

    # -------------------- Prediction --------------------

    # Prediction for plot
    simulation_inference_steps(
        ITERS, 
        checkpoint=f'Output/Model/Simulation/{TASK_NAME}', 
        new_x_path='Data/Simulation/Coordinates/simulation_new_x_plot.npy', 
        save_path=f'Output/Prediction/Simulation/{TASK_NAME}/prediction_plot',
        para=2,
        Q=Q)

    # -------------------- Plot --------------------

    # Plot prediction
    simulation_plot_2steps(
        X_path='Data/Simulation/Coordinates/simulation_new_x_plot.npy', 
        Y_path=f'Output/Prediction/Simulation/{TASK_NAME}/prediction_plot', 
        save_path=f'Output/Plot/Simulation/Prediction/{TASK_NAME}')

    simulation_plot_res(
        X_path='Data/Simulation/Coordinates/simulation_new_x_plot.npy', 
        pre_path=f'Output/Prediction/Simulation/{TASK_NAME}/prediction_plot_step2.npy', 
        sol_path='Data/Simulation/Solutions/simulation_solution_plot.npy', 
        save_path=f'Output/Plot/Simulation/Prediction/{TASK_NAME}Res')

    simulation_plot_parameters(model_dir=f'Output/Model/Simulation/{TASK_NAME}', save_dir='Output/Plot/Simulation/parameters.png')

    simulation_plot_rmse_2(
        sol_path='Data/Simulation/Solutions/simulation_solution_plot.npy', 
        mdrf_path='Output/Prediction/Simulation/2Steps2Para/prediction_plot_step2.npy',
        nmdrf_path='Output/Prediction/Simulation/NoPhysics/prediction_plot.npy',
        gpr_path='Output/Prediction/Simulation/GPR/prediction_plot.npy',
        kriging_path='Output/Prediction/Simulation/Kriging/prediction_plot.npy',
        save_path='Output/Plot/Simulation/simualtion_rmse.png')

    ######################### No Physics #########################

    # -------------------- Training --------------------

    simulation_train_no_physics(simulation_data_path='Data/Simulation/Train', output_dir='Output/Model/Simulation/NoPhysics', iters=15000, display=100, lrs=1e-4)

    # -------------------- Prediction --------------------

    # Prediction for plot
    simulation_inference(
        checkpoint='Output/Model/Simulation/NoPhysics/model-10000.pt', 
        new_x_path='Data/Simulation/Coordinates/simulation_new_x_plot.npy', 
        save_path='Output/Prediction/Simulation/NoPhysics/prediction_plot.npy')

    # -------------------- Plot --------------------

    # Plot prediction
    simulation_plot(
        X_path='Data/Simulation/Coordinates/simulation_new_x_plot.npy', 
        Y_path='Output/Prediction/Simulation/NoPhysics/prediction_plot.npy', 
        save_path='Output/Plot/Simulation/Prediction/NoPhysics')

    simulation_plot_res(
        X_path='Data/Simulation/Coordinates/simulation_new_x_plot.npy', 
        pre_path='Output/Prediction/Simulation/NoPhysics/prediction_plot.npy', 
        sol_path='Data/Simulation/Solutions/simulation_solution_plot.npy', 
        save_path='Output/Plot/Simulation/Prediction/NoPhysicsRes')


    ######################### Kriging #########################

    # -------------------- Prediction --------------------

    simulation_other_methods_inference(
        simulation_data_path='Data/Simulation/Train', 
        new_x_path='Data/Simulation/Coordinates/simulation_new_x_plot.npy', 
        save_path='Output/Prediction/Simulation/Kriging/prediction_plot.npy',
        model='kriging')

    # -------------------- Plot --------------------

    # Plot kriging results
    simulation_plot(X_path='Data/Simulation/Coordinates/simulation_new_x_plot.npy', Y_path='Output/Prediction/Simulation/Kriging/prediction_plot.npy', save_path='Output/Plot/Simulation/Prediction/Kriging')


    ######################### GPR #########################

    # -------------------- Prediction --------------------

    # Gaussian preocess regression
    simulation_other_methods_inference(
        simulation_data_path='Data/Simulation/Train', 
        new_x_path='Data/Simulation/Coordinates/simulation_new_x_plot.npy', 
        save_path='Output/Prediction/Simulation/GPR/prediction_plot.npy',
        model='gpr')

    # -------------------- Plot --------------------

    # Plot GPR
    simulation_plot(X_path='Data/Simulation/Coordinates/simulation_new_x_plot.npy', Y_path='Output/Prediction/Simulation/GPR/prediction_plot.npy', save_path='Output/Plot/Simulation/Prediction/GPR')


if __name__ == '__main__':
    simulation()
