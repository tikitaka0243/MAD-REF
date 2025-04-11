from datetime import datetime, timezone
import json
import math
import os
from typing import Literal, Optional
from matplotlib import pyplot as plt, ticker
from matplotlib.colors import ListedColormap, LogNorm
import matplotlib.colors as colors
import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import pandas as pd
from sklearn.metrics import root_mean_squared_error
from tqdm import tqdm
from scipy.sparse import coo_matrix
from PIL import Image, ImageDraw, ImageFont 
import re

from Code.DataProcess import RealData, SimulationData
from Code.utils import DATA_PATH, EXTEND, EXTEND_GLOBAL, EXTEND_SEAICE, LINESTYLES, METHOD_COLORS, METHOD_MARKERS, PURPLE_COLORS, SLICE_COOR_NAMES, SLICE_FIGSIZE, SLICE_LAT_UNIT, SLICE_LAT_UNIT_GLOBAL, SLICE_LONG_UNIT, SLICE_LONG_UNIT_GLOBAL, TASK_COLORS, TIME_LIST, VAR_COLORS, VAR_NAMES, VAR_UNITS, VARS, VMAX, VMAX_GLOBAL, VMIN, VMIN_GLOBAL, VMIN_VMAX_SEAICE, WARM_COLORS, convert_to_timestamp, data_split, find_files_with_prefix, gen_folder, gen_folder_2steps, ndarray_check, normalize_t, read_dat_to_numpy, unpack_x, unpack_x_sim, unpack_y_sim


def simulation_gen_plot_folders(path, unique_t):
    for t_ in unique_t:
        path_ = os.path.join(path, f't={t_:.2f}')
        if not os.path.exists(path_):
            os.mkdir(path_)



def find_closest_ts(t_point, t_vector, n=100):
    dis = np.abs(t_vector - t_point)
    indices_of_smallest = np.argpartition(dis, n)[:n]
    return indices_of_smallest


def calc_y(x, constant=0.7, sign=1):
    return sign * abs(constant ** 3 - abs(x) ** 3) ** (1/3)


def data_domain_transform(x):
    return x / 2 + 0.5



class SimulationPlot():

    def simulation_data_plot(self, simulation_data_path = 'Data/Simulation/Train', save_path='Output/Plot/Simulation/Data'):
        print('Plotting simualtion data...')
        
        gen_folder(save_path)

        unique_t = np.linspace(0, 1, 5, endpoint=True)
        simulation_gen_plot_folders(save_path, unique_t)
        
        tau, w, v, p, x_z_t = SimulationData.simulation_read_data(simulation_data_path)
        Y = np.column_stack((tau, w, v, p))
        x, z, t = unpack_x_sim(x_z_t)
        
        for t_i, t_ in tqdm(enumerate(unique_t), total=len(unique_t)):
            plot_indices = find_closest_ts(t_, t.flatten(), n=200)
            x_plot = x[plot_indices]
            z_plot = z[plot_indices]
            
            for var in range(4):
                Y_t = Y[plot_indices]
                self.simulation_plot_one(var, t_, t_i, x_plot, z_plot, Y_t, save_path, s=10, marker='.')
                
        print('Done.')

    def simulation_plot_one(self, var, t_, t_i, x, z, Y_t, save_path, s=0.5, marker='s', vmins=[-1.1, -1, -1, -0.4, -1], vmaxs=[1.1, 1, 1, 0.4, 1], color_num=[13, 15, 17, 19, 21], cmaps=[plt.cm.plasma, plt.cm.RdYlGn, plt.cm.RdGy, plt.cm.inferno_r, plt.cm.cividis]):
        CMAPS = []
        TITLES = ['Temperature', 'Vertical Velocity', 'Horizontal Velocity', 'Pressure', 'Q']
        SAVE_NAMES = ['tau', 'w', 'v', 'p', 'q']

        for cmap in cmaps:
            colors = cmap(np.linspace(0, 1, color_num[t_i], endpoint=True))
            CMAPS.append(ListedColormap(colors))

        plt.figure(figsize=(4, 3))
        sc = plt.scatter(x, z, c=Y_t[:, var], cmap=CMAPS[var], alpha=1, s=s, marker=marker, vmin=vmins[var], vmax=vmaxs[var])
        plt.colorbar(sc)
        if var < 3:
            self.simulation_plot_data_domain()
        plt.title(TITLES[var])
        plt.xlabel('x')
        plt.ylabel('z')
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.savefig(
            os.path.join(save_path, f't={t_:.2f}', f'{SAVE_NAMES[var]}.png'), 
            dpi=300, 
            bbox_inches='tight')
        plt.close()
        
        
    def simulation_plot_data_domain(self, l=1):
        x_values_pos = np.linspace(0, 0.7, 1000, endpoint=True)
        x_values_neg = -x_values_pos
        y_values_pos = calc_y(x_values_pos, sign=1)
        y_values_neg = calc_y(x_values_neg, sign=-1)
        
        x_values_pos = data_domain_transform(x_values_pos)
        x_values_neg = data_domain_transform(x_values_neg)
        y_values_pos = data_domain_transform(y_values_pos)
        y_values_neg = data_domain_transform(y_values_neg)

        plt.plot(x_values_pos, y_values_pos, color='cyan', linewidth=l)
        plt.plot(x_values_pos, y_values_neg, color='cyan', linewidth=l)
        plt.plot(x_values_neg, y_values_pos, color='cyan', linewidth=l)
        plt.plot(x_values_neg, y_values_neg, color='cyan', linewidth=l)
    
    
    def simulation_plot_add_rmse(self, rmse):
        annotation_text = f"RMSE={rmse:.3f}"
        annotation_position = (0.5, 0.1)
        plt.text(annotation_position[0], annotation_position[1], annotation_text, fontsize=12, color='black')

        
    def simulation_plot(self, X_path, Y_path, save_path='Output/Plot/Simulation/Prediction', vmins=[-1.1, -1, -1, -0.4, -3], vmaxs=[1.1, 1, 1, 0.4, 3], color_num=[13, 15, 17, 19, 21], cmaps=[plt.cm.plasma, plt.cm.RdYlGn, plt.cm.RdGy, plt.cm.inferno_r, plt.cm.bone]):
        print('Simulation plot...')
        
        gen_folder(save_path)
        
        X = np.load(X_path)
        Y = np.load(Y_path)

        t = X[:, 2]
        unique_t = np.unique(t)
        
        simulation_gen_plot_folders(save_path, unique_t)
        
        for t_i, t_ in tqdm(enumerate(unique_t), total=len(unique_t)):
            x, z, _ = unpack_x_sim(X[X[:, 2] == t_])
            Y_t = Y[X[:, 2] == t_]

            for var in range(Y.shape[1]):
                self.simulation_plot_one(var, t_, t_i, x, z, Y_t, save_path, vmins=vmins, vmaxs=vmaxs, color_num=color_num, cmaps=cmaps)
                
        print('Done.')
        
        
    def simulation_plot_res(self, X_path, pre_path, sol_path, save_path):
        pre = np.load(pre_path)
        sol = np.load(sol_path)[:, :4]
        res = np.abs(pre - sol)
        res_path = f'{pre_path[:-4]}_res.npy'
        np.save(res_path, res)
        
        self.simulation_plot(X_path, res_path, save_path, vmins = [0, 0, 0, 0], vmaxs = [0.5, 0.5, 0.5, 0.25], color_num = [20, 22, 24, 26, 28], cmaps=[plt.cm.gray_r] * 4)
        

    def simulation_plot_2steps(self, X_path='Data/Simulation/Coordinates/simulation_new_x_plot.npy', Y_path='Output/Prediction/Simulation/2Steps/prediction_plot', save_path='Output/Plot/Simulation/Prediction/2Steps', plot_step_1=False):
        print('Plotting model results of two steps...')

        gen_folder_2steps(save_path)

        if plot_step_1:
            # Plot prediction (Step1)
            self.simulation_plot(
                X_path=X_path, 
                Y_path=f'{Y_path}_step1.npy', 
                save_path=os.path.join(save_path, 'Step1'))
            
            print(f'Images of step 1 results saved to {os.path.join(save_path, "Step1")}')

        # Plot prediction (Step2)
        self.simulation_plot(
            X_path=X_path, 
            Y_path=f'{Y_path}_step2.npy', 
            save_path=os.path.join(save_path, 'Step2'))
        
        print(f'Images of step 2 results saved to {os.path.join(save_path, "Step2")}')

    @staticmethod
    def simlulation_plot_lambdas(output_image_path, lambdas=None, file_path=None):

        # Read the CSV file
        if file_path is not None:
            df = pd.read_csv(file_path)
        else:
            df = pd.DataFrame(lambdas, columns=['lambda_pde'])
        
        # Setting the x-axis as training iteration number (assuming each row represents 100 iterations)
        df['iteration'] = range(100, len(df) * 100 + 100, 100)

        df = df[df['iteration'] <= 100000]
        
        # Plotting the data
        plt.figure(figsize=(10, 6), dpi=300)
        plt.plot(df['iteration'], df['lambda_pde'], label='Lambda PDE')
        # plt.plot(df['iteration'], df['lambda_icbc'], label='Lambda ICBC')
        
        # Adding labels and title
        plt.xlabel('Training Iteration')
        plt.ylabel('Lambda Value')
        plt.title('Lambda Values over Training Iterations')
        plt.legend()

        # plt.xscale('log')
        # plt.yscale('log')

        plt.ylim((0, 5))
        
        # Show the plot
        plt.grid(True)

        gen_folder(os.path.dirname(output_image_path))
        plt.savefig(output_image_path, format='jpg', bbox_inches='tight')

        plt.close()
        
    def simulation_plot_parameters(self, model_dir='Output/Model/Simulation/2Steps2Para', save_dir='Output/Plot/Simulation/variables.png'):
        var_folder_dir = os.path.join(model_dir, 'Step2')
        var_file_dirs = find_files_with_prefix(var_folder_dir, 'variable_')
        
        var_files = []
        for var_file_dir in var_file_dirs:
            var_file = read_dat_to_numpy(var_file_dir)
            var_files.append(var_file)
            
        var_files = np.dstack(var_files)
        var_files = var_files[1:, :, :]
        var_mean = np.mean(var_files, axis=2)
        var_lower = np.percentile(var_files, q=2.5, axis=2)
        var_upper = np.percentile(var_files, q=97.5, axis=2)

            
        plt.figure(figsize=(5, 6), dpi=300)
        
        plt.plot(var_mean[:, 0], var_mean[:, 1], color='crimson', alpha=1, linewidth=1, label='zeta')
        plt.plot(var_mean[:, 0], var_mean[:, 2], color='royalblue', alpha=1, linewidth=1, label='zeta_tau')
            
        plt.fill_between(var_lower[:, 0], var_lower[:, 1], var_upper[:, 1], color='lightpink', alpha=0.5, edgecolor='none')
        plt.fill_between(var_lower[:, 0], var_lower[:, 2], var_upper[:, 2], color='lightsteelblue', alpha=0.5, edgecolor='none')
        
        plt.axhline(y=0.01, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        plt.axhline(y=0.02, color='gray', linestyle='--', linewidth=1, alpha=0.5)

        plt.title('Unknown Parameters')
        plt.xlabel('Iteration')
        plt.ylabel('Parameter Value')
        plt.xscale('log')
        plt.yscale('log')
        plt.legend()
        plt.savefig(save_dir, bbox_inches='tight')
        plt.close()

    def plot_rmse_trends(self, tasks, task_names, output_path='Output/Plot/Simulation/rmse_trends.png', average_tasks=False, folder_path = 'Output/Prediction/Simulation/RMSETrends'): 

        # Create a single figure
        plt.figure(figsize=(10, 6))

        # For keeping track of whether a task's legend has been added
        legend_added = {task: False for task in tasks}

        # Loop over tasks and plot their corresponding data
        for i, task in enumerate(tasks):
            # Load the data for the task
            df = pd.read_csv(os.path.join(folder_path, f'rmse_results_{task}.csv'))

            # Get the color for the current task
            color = PURPLE_COLORS[i % len(PURPLE_COLORS)]  # Ensure we cycle through the colors if there are more tasks

            if average_tasks:
                # Compute the mean RMSE across all variables
                mean_rmse = df[VARS].mean(axis=1)

                # Plot the mean RMSE for the task
                plt.plot(df.index * 1000, mean_rmse, label=f'{task_names[i]}', color=color, linestyle='-', linewidth=2)
            else:
                # Plot each variable for the current task
                for j, column in enumerate(VARS):
                    # Add the task's name to the legend only for the first variable
                    if not legend_added[task]:
                        label = f'{task_names[i]}'
                        legend_added[task] = True
                    else:
                        label = None  # No label for subsequent variables from the same task

                    # Plot the data
                    plt.plot(df.index * 1000, df[column], label=label, linestyle=LINESTYLES[j], color=color)

        # Add the legend (only one per task)
        plt.legend()

        # Add title and axis labels
        plt.title('RMSE Trends')
        plt.xlabel('Iterations')
        plt.ylabel('RMSE')

        # Log scale for the y-axis
        plt.yscale('log')

        # Save the plot to a file
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()


    def simulation_plot_rmse(
        self,
        X_path='Data/Simulation/Coordinates/simulation_new_x_plot.npy', 
        sol_path='Data/Simulation/Solutions/simulation_solution_plot.npy', 
        mdrf_path='Output/Prediction/Simulation/2Steps2Para/prediction_plot_step2.npy',
        nmdrf_path='Output/Prediction/Simulation/NoPhysics/prediction_plot.npy',
        gpr_path='Output/Prediction/Simulation/GPR/prediction_plot.npy',
        kriging_path='Output/Prediction/Simulation/Kriging/prediction_plot.npy',
        save_path='Output/Plot/Simulation/simualtion_rmse.png',
        data_domain_bool=False):
        
        X = np.load(X_path)
        sol = np.load(sol_path)
        mdrf = np.load(mdrf_path)
        nmdrf = np.load(nmdrf_path)
        gpr = np.load(gpr_path)
        kriging = np.load(kriging_path)
        
        sol = sol[:, :4]
        x, z, t = X[:, 0], X[:, 1], X[:, 2]
        if data_domain_bool:
            data_domain = simulation_data_domain_check(x, z)
            sol = sol[data_domain]
            t = t[data_domain]
        t_series = pd.Series(t, name='Group') 
        
        rmses = []
        for i, pre in enumerate([mdrf, nmdrf, gpr, kriging]):
            if data_domain_bool:
                pre = pre[data_domain]
            se = np.square(sol - pre)
            se = pd.DataFrame(se)
            se['Group'] = t_series
            rmse = se.groupby('Group').mean().apply(np.sqrt)
            rmses.append(rmse)
            
        MODEL_NAMES = ['MAD-REF Net', 'N-MAD-REF Net', 'GPR', 'R-Kriging']
        COLORS = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']

        
        plt.figure(figsize=(5, 6), dpi=300)
        
        for i in range(len(rmses)):
            rmse = rmses[i]
            if i == 0:
                j_num = rmse.shape[1]
            else:
                j_num = rmse.shape[1] - 1
            for j in range(j_num):
                plt.plot(rmse.index, rmse.iloc[:, j], color=COLORS[i], alpha=1, linewidth=1.5, label=MODEL_NAMES[i], linestyle=LINESTYLES[j])
                plt.scatter(rmse.index, rmse.iloc[:, j], color=COLORS[i], alpha=1, s=15)
        
        if data_domain_bool:
            plt.title('Data Domain')
        else:
            plt.title('Whole Domain')
        plt.xlabel('t')
        plt.ylabel('RMSE')
        plt.yscale('log')
        plt.ylim(5e-4, 4e-1)
        # plt.legend()
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        
    def simulation_plot_rmse_2(
        self,
        X_path='Data/Simulation/Coordinates/simulation_new_x_plot.npy',
        sol_path='Data/Simulation/Solutions/simulation_solution_plot.npy', 
        mdrf_path='Output/Prediction/Simulation/2Steps2Para/prediction_plot_step2.npy',
        nmdrf_path='Output/Prediction/Simulation/NoPhysics/prediction_plot.npy',
        gpr_path='Output/Prediction/Simulation/GPR/prediction_plot.npy',
        kriging_path='Output/Prediction/Simulation/Kriging/prediction_plot.npy',
        save_path='Output/Plot/Simulation/simualtion_rmse.png'):
        
        self.simulation_plot_rmse(X_path, sol_path, mdrf_path, nmdrf_path, gpr_path, kriging_path, save_path, data_domain_bool=False)

        self.simulation_plot_rmse(X_path, sol_path, mdrf_path, nmdrf_path, gpr_path, kriging_path, save_path=save_path[:-4] + '_data_domain.png', data_domain_bool=True)

    def plot_loss_curve(loss_list, steps, save_path, pde_num=4, title='Loss Curve', ylabel='Loss'):

        # Initialize the plot
        plt.figure(figsize=(10, 6))

        loss_num = len(loss_list[0])
        # Plot each loss curve separately
        for i in range(loss_num):
            loss_values = [loss[i] for loss in loss_list]

            if loss_num > pde_num:
                if i < pde_num:
                    plt.plot(steps, loss_values, label=f'pde_{i + 1}')
                else:
                    plt.plot(steps, loss_values, label=f'data_{i + 1 - pde_num}', linestyle='--')
            else:
                plt.plot(steps, loss_values, label=f'data_{i + 1}', linestyle='--')

        # Add title and labels
        plt.title(title, fontsize=16)
        plt.xlabel('Steps', fontsize=14)
        plt.ylabel(ylabel, fontsize=14)

        plt.yscale('log')

        # Add grid and legend
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=12)

        plt.savefig(save_path, bbox_inches='tight')
        plt.close()


class RealDataPlot():

    def __init__(self, data_loader=None):

        self.data_loader = data_loader
        self.area = data_loader.area if data_loader is not None else None
        self.area_config = {
            'Local': {
                'plot_function': self.plot_local_,
                'vmins': VMIN,
                'vmaxs': VMAX,
                'extends': EXTEND
            },
            'Global': {
                'plot_function': self.plot_global_,
                'vmins': VMIN_GLOBAL,
                'vmaxs': VMAX_GLOBAL,
                'extends': EXTEND_GLOBAL
            }
        }

    def plot(self, 
             task_name,  
             fit_path='Prediction/RealData/prediction_plot.npy', 
             t_list=TIME_LIST,
             new_x_path='Data/RealData/Local/new_x_plot.npy',
             area='Local',
             sublearner=None,
             var_range=range(7),
             save_path=None):

        config = self.area_config.get(area, {})
        plot_function = config.get('plot_function')
        vmins = config.get('vmins')
        vmaxs = config.get('vmaxs')
        extends = config.get('extends')
        
        new_x = np.load(new_x_path)
        new_x = self.data_loader.normalization_coordinates(new_x)

        fit = np.load(fit_path)

        if sublearner is not None:
            df = np.column_stack((new_x, fit))
            df = self.data_loader.data_filter_theta(df, sublearner, normalize=True)

            if sublearner == 2:
                df = self.data_loader.coordinates_rotation(df)
                
            new_x = df[:, :4]
            fit = df[:, 4:]

        t_list_ = convert_to_timestamp(t_list)
        t_ = normalize_t(t_list_, self.data_loader.t_min)

        for i, t in enumerate(tqdm(t_, desc="Time Loop Progress")):

            for j in tqdm(var_range, desc=f"Variable Loop Progress ({t_list[i]})", leave=False):

                fit_plot = fit[new_x[:, 3] == t, j]
                new_x_plot = new_x[new_x[:, 3] == t]

                if save_path is not None:
                    save_path_ = os.path.join(f'Output/Plot/RealData/{task_name}', save_path, f'{t_list[i]}/{VAR_NAMES[j]}.png')
                else:
                    save_path_ = os.path.join(f'Output/Plot/RealData/{task_name}', f'{t_list[i]}/{VAR_NAMES[j]}.png')

                plot_function(
                    new_x_plot[:, 0], 
                    new_x_plot[:, 1], 
                    new_x_plot[:, 2], 
                    fit_plot, 
                    VAR_COLORS[j], 
                    vmin=vmins[j], 
                    vmax=vmaxs[j], 
                    label=VAR_UNITS[j], 
                    extend=extends[j], 
                    save_path=save_path_,
                    var_name=VAR_NAMES[j]
                )


    
    def plot_data(self, datapath='Data/RealData/Local', save_path='Output/Plot/RealData/Local/Data', area='Local', rotate=False):

        config = self.area_config.get(area, {})
        plot_function = config.get('plot_function')
        vmins = config.get('vmins')
        vmaxs = config.get('vmaxs')
        extends = config.get('extends')

        for data_type in tqdm(['vali', 'train', 'test'], desc="Processing Data Types"):

            i = 0

            for data_name in tqdm(DATA_PATH, desc=f"Processing {data_type}", leave=False):
            
                df = np.load(os.path.join(datapath, f'{data_name}_{data_type}.npy'))

                

                if area != 'local' or data_name != 'Argo/argo':
                    timestamp_1 = datetime(2021, 1, 1, 0, 0).timestamp()
                    timestamp_2 = datetime(2021, 2, 1, 0, 0).timestamp()
                    df = df[(df[:, 3] > timestamp_1) & (df[:, 3] < timestamp_2)]

                # print(f'{data_name}_{data_type}.npy')
                # ndarray_check(df)

                df[:, :4] = self.data_loader.normalization_coordinates(df)

                if area == 'Global' and rotate:
                    df = self.data_loader.coordinates_rotation(df)

                r_, theta_, phi_, t_ = unpack_x(df)


                if df.shape[1] == 6:
                    for j in range(2):
                        plot_function(r_, theta_, phi_, df[:, 4 + j], VAR_COLORS[i], vmins[i], vmaxs[i], VAR_UNITS[i], extends[i], save_path=os.path.join(save_path, f'{VAR_NAMES[i]}_{data_type}.png'), var_name=VAR_NAMES[i])
                        i += 1

                else:
                    plot_function(r_, theta_, phi_, df[:, 4], VAR_COLORS[i], vmins[i], vmaxs[i], VAR_UNITS[i], extends[i], save_path=os.path.join(save_path, f'{VAR_NAMES[i]}_{data_type}.png'), var_name=VAR_NAMES[i])
                    i += 1
        

    def plot_local_(self, r, theta, phi, var, cmap, vmin, vmax, label, extend, save_path, var_name):

        fig = plt.figure(figsize=(30, 12), dpi=300)
        ax = fig.add_subplot(111, projection='3d', proj_type='ortho')
        ax.set_box_aspect(aspect = (4, 10, 1))
        ax.view_init(elev=50)
        # ax.view_init(elev=50, azim=-20)

        if var_name in ['v_theta', 'v_phi']:
            ax.xaxis.pane.set_facecolor((0.8, 0.8, 0.8))
            ax.yaxis.pane.set_facecolor((0.8, 0.8, 0.8))
            ax.zaxis.pane.set_facecolor((0.8, 0.8, 0.8))
         
        img = ax.scatter(theta, phi, r, c=var, cmap=cmap, vmin=vmin, vmax=vmax, s=5, alpha=1, linewidths=0, rasterized=True,    antialiased=False)
        cbar = fig.colorbar(img, shrink=0.5, pad=0.06, format='%.2e', extend=extend)
        
        cbar.ax.tick_params(labelsize=25)
        cbar.set_label(label, fontsize=25, labelpad=20)

        self.scientific_notation(cbar)

        plt.xlim(theta.min(), theta.max())
        plt.ylim(phi.min(), phi.max())
        ax.set_zlim(r.min(), r.max())
        ax.invert_xaxis()
        ax.tick_params(axis='both', which='major', labelsize=25, pad=8)
        ax.tick_params(axis='x', pad=15, direction='out', labelrotation=53)
        ax.tick_params(axis='z', pad=25)
        ax.tick_params(axis='y', pad=28, direction='out', labelrotation=-27)
        ax.set_zlabel('Depth (m)', fontsize=25, labelpad=45)

        self.plot_local_set_ticks(ax)
        
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)

        gen_folder(os.path.dirname(save_path))
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()


    def plot_local_set_ticks(self, ax):
        ax.set_zticks([-0.4, -0.2, 0])
        ax.set_zticklabels(['-2000', '-1000', '0'])

        # Set x-axis ticks and labels
        xticks_degrees = [110, 100, 90, 80, 70]  # Degrees for x-axis ticks
        xticks_radians = np.deg2rad(xticks_degrees)  # Convert degrees to radians
        xticklabels = ['20°S', '10°S', '0°', '10°N', '20°N']  # Labels for x-axis ticks

        ax.set_xticks(xticks_radians)  # Set x-axis ticks in radians
        ax.set_xticklabels(xticklabels)  # Set x-axis tick labels
        ax.invert_xaxis()

        # Set y-axis ticks and labels
        yticks_degrees = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]  # Degrees for y-axis ticks
        yticks_radians = np.deg2rad(yticks_degrees)  # Convert degrees to radians
        yticklabels = ['150°E', '160°E', '170°E', '180°', '170°W', '160°W', '150°W', '140°W', '130°W', '120°W', '110°W']  # Labels for y-axis ticks

        ax.set_yticks(yticks_radians)  # Set y-axis ticks in radians
        ax.set_yticklabels(yticklabels)  # Set y-axis tick labels

    
    def plot_global_(self, r, theta, phi, var, cmap, vmin, vmax, label, extend, save_path, var_name):
        fig = plt.figure(figsize=(30, 12), dpi=300)
        ax = fig.add_subplot(111, projection='3d', proj_type='ortho')
        ax.set_box_aspect(aspect = (4, 6, 0.6))
        ax.view_init(50, -20)
        # ax.view_init(elev=50, azim=-20)

        if var_name in ['v_theta', 'v_phi']:
            # ax.set_facecolor((0.8, 0.8, 0.8))
            ax.xaxis.pane.set_facecolor((0.8, 0.8, 0.8))
            ax.yaxis.pane.set_facecolor((0.8, 0.8, 0.8))
            ax.zaxis.pane.set_facecolor((0.8, 0.8, 0.8))
         
        img = ax.scatter(theta, phi, r, c=var, cmap=cmap, vmin=vmin, vmax=vmax, s=2, alpha=1, linewidths=0, rasterized=True,    antialiased=False)
        cbar = fig.colorbar(img, shrink=0.5, pad=0.06, format='%.2e', extend=extend)
        
        cbar.ax.tick_params(labelsize=25)
        cbar.set_label(label, fontsize=25, labelpad=12)

        self.scientific_notation(cbar)

        plt.xlim(theta.min(), theta.max())
        plt.ylim(phi.min(), phi.max())
        ax.set_zlim(r.min(), r.max())
        ax.invert_xaxis()
        ax.tick_params(axis='both', which='major', labelsize=25, pad=8)
        ax.tick_params(axis='x', pad=15)
        ax.tick_params(axis='z', pad=15)
        ax.tick_params(axis='y', pad=22)
        ax.set_zlabel('Depth (m)', fontsize=25, labelpad=31)

        self.plot_global_set_ticks(ax)
        
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)

        gen_folder(os.path.dirname(save_path))
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()

    def scientific_notation(self, cbar):
        formatter = ticker.ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((-1, 1))
        cbar.ax.yaxis.set_major_formatter(formatter)
        cbar.ax.yaxis.get_offset_text().set_fontsize(20)

    def plot_global_set_ticks(self, ax):
        ax.set_zticks([-0.4, -0.2, 0])
        ax.set_zticklabels(['-2000', '-1000', '0'])
        # Set x-axis ticks and labels
        xticks_degrees = [0, 45, 90, 135, 180][::-1]  # Degrees for x-axis ticks
        xticks_radians = np.deg2rad(xticks_degrees)  # Convert degrees to radians
        xticklabels = ['90°S', '45°S', '0°', '45°N', '90°N']  # Labels for x-axis ticks

        ax.set_xticks(xticks_radians)  # Set x-axis ticks in radians
        ax.set_xticklabels(xticklabels)  # Set x-axis tick labels
        ax.invert_xaxis()

        # Set y-axis ticks and labels
        yticks_degrees = [0, 45, 90, 135, 180, 225, 270, 315, 360]  # Degrees for y-axis ticks
        yticks_radians = np.deg2rad(yticks_degrees)  # Convert degrees to radians
        yticklabels = ['180°', '135°W', '90°W', '45°W', '0°', '45°E', '90°E', '135°E', '180°']  # Labels for y-axis ticks

        ax.set_yticks(yticks_radians)  # Set y-axis ticks in radians
        ax.set_yticklabels(yticklabels)  # Set y-axis tick labels


    def plot_data_time_distribution(self, data_path='Data/RealData/Local', save_path='Output/Plot/RealData/Local//Data/TimeDistribution'):

        for path in DATA_PATH:
            for data_type in ['train', 'vali', 'test']:
                df = np.load(os.path.join(data_path, f'{path}_{data_type}.npy'))
                t = df[:, 3]

                self.plot_histogram(t, save_path=os.path.join(save_path, f'{os.path.basename(path)}_{data_type}.jpg'))

    def plot_histogram(self, data, bins=50, title="Data distribution over time", xlabel="Timestamp", ylabel="Counts", save_path=None):
        """
        Draws a histogram for the given data distribution.
        
        Parameters:
        data : list or array-like
            The data to plot the histogram for.
        bins : int, optional, default: 10
            Number of bins in the histogram.
        title : str, optional, default: "Histogram"
            Title of the histogram.
        xlabel : str, optional, default: "Values"
            Label for the x-axis.
        ylabel : str, optional, default: "Frequency"
            Label for the y-axis.
        color : str, optional, default: "blue"
            Color of the histogram.
        alpha : float, optional, default: 0.7
            Transparency level, ranging from 0 (fully transparent) to 1 (opaque).
        """
        plt.figure(figsize=(8, 6))
        plt.hist(data, bins=bins, edgecolor='black')
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(axis='y', alpha=0.75)

        gen_folder(os.path.dirname(save_path))
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()

    def _plot_rmse(self, iterations, rmse_values, title, save_path):
        """
        Plot RMSE values and save the plot to the specified path.

        Args:
            iterations (list): List of iteration numbers.
            rmse_values (list): List of RMSE values corresponding to the iterations.
            title (str): Title of the plot.
            save_path (str): Path to save the plot.
        """

        plt.figure()
        plt.plot(iterations, rmse_values, linestyle='-', linewidth=2)
        plt.title(title)
        plt.xlabel('Iteration')
        plt.ylabel('RMSE')
        plt.grid(True)

        gen_folder(os.path.dirname(save_path))
        plt.savefig(save_path, bbox_inches='tight') # Save the plot
        plt.close()  # Close the plot to free up memory

    def read_csv_and_plot_rmse(self, task_name_1, task_name_2, label_1, label_2, var, output_file):
        file1 = f'Output/Model/RealData/{task_name_1}/Step1/validation_rmse/validation_results.csv'
        file2 = f'Output/Model/RealData/{task_name_2}/Step1/validation_rmse/validation_results.csv'

        # Read CSV files
        df1 = pd.read_csv(file1)
        df2 = pd.read_csv(file2)
        
        # Extract 'Iteration' and 'RMSE_v_theta' columns
        iterations1 = df1['Iteration']
        rmse_method1 = df1[f'RMSE_{var}']
        
        iterations2 = df2['Iteration']
        rmse_method2 = df2[f'RMSE_{var}']
        
        # Find the maximum common iteration value
        max_common_iteration = min(iterations1.max(), iterations2.max())
        
        # Truncate data based on the maximum common iteration value
        df1_truncated = df1[df1['Iteration'] <= max_common_iteration]
        df2_truncated = df2[df2['Iteration'] <= max_common_iteration]
        
        iterations1 = df1_truncated['Iteration']
        rmse_method1 = df1_truncated[f'RMSE_{var}']
        
        iterations2 = df2_truncated['Iteration']
        rmse_method2 = df2_truncated[f'RMSE_{var}']
        
        # Plot the data
        plt.figure(figsize=(10, 6))
        plt.plot(iterations1, rmse_method1, label=label_1, linewidth=2)
        plt.plot(iterations2, rmse_method2, label=label_2, linewidth=2)
        
        # Add title and labels (in English)
        plt.title(f'RMSE_{var} vs Iterations for Two Methods')
        plt.xlabel('Iterations')
        plt.ylabel(f'RMSE_{var}')
        plt.legend()
        plt.grid(True)
        
        # Save the plot to a file
        gen_folder(os.path.dirname(output_file))
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_file}")
        plt.close()

    
    def plot_data_glory_tempsal_polar(self, save_path='Output/Plot/RealData/Global/Data'):

        data = self.data_loader.read_glory_tempsal(time='202110')
        depths = self.data_loader.read_depths()

        vmin_vmax_list = []

        for i in tqdm(range(40), desc='Plotting'):
            data_plot = data[np.isclose(data[:, 0], depths[i])]

            r, theta, phi, t = unpack_x(data_plot)
            
            for j in range(2):
                for k in ['Arctic', 'Antarctic']:

                    if k == 'Arctic':
                        filtered_data = data_plot[theta.ravel() >= 45]
                    elif k == 'Antarctic':
                        filtered_data = data_plot[theta.ravel() <= -45]
                    column_data = filtered_data[:, 4 + j]
                    vmin_vmax = [np.min(column_data), np.max(column_data)]

                    vmin_vmax_list.append({
                        'depth': depths[i],
                        'variable_index': j,
                        'region': k,
                        'vmin_vmax': vmin_vmax
                    })

                    self.plot_polar_scatter_cartopy(phi, theta, data_plot[:, 4 + j], 
                                                var_name=VAR_UNITS[j], 
                                                figsize=(8, 6),
                                                point_size=0.2,
                                                cmap=VAR_COLORS[j], 
                                                output_path=os.path.join(save_path, k, f'{VAR_NAMES[j]}_{depths[i]:.2f}.png'),
                                                vmin_vmax=vmin_vmax,
                                                region=k,
                                                log_transform=False,
                                                face_color=False,
                                                depth=depths[i])
        
        with open(os.path.join(save_path, 'vmin_vmax_polar.json'), 'w') as f:
            json.dump(vmin_vmax_list, f, indent=4)

    
    def plot_data_glory_tempsal_slice(self, save_path='Output/Plot/RealData/Local/Data', plot_type='all', time='202107'):
        """
        Plot GLORYS temperature and salinity slices.

        Parameters:
            save_path (str): Path to save plots. Defaults to 'Output/Plot/RealData/Local/Data'.
            plot_type (str): Type of plots to generate. Options: 'depth', 'latitude', 'longitude', 'all'. Defaults to 'all'.
        """
        data = self.data_loader.read_glory_tempsal(time=time)

        if self.area == 'Local':
            data = self.data_loader.data_filter(data, self.data_loader.r_min, self.data_loader.r_max, self.data_loader.theta_min, self.data_loader.theta_max, self.data_loader.phi_min, self.data_loader.phi_max, self.data_loader.t_min, self.data_loader.t_max)

        plot_actions = {
            'depth': lambda: self.plot_data_glory_tempsal_slice_depth(data, save_path),
            'latitude': lambda: self.plot_data_glory_tempsal_slice_latitude(data, save_path),
            'longitude': lambda: self.plot_data_glory_tempsal_slice_longitude(data, save_path),
            'all': lambda: [
                self.plot_data_glory_tempsal_slice_depth(data, save_path),
                self.plot_data_glory_tempsal_slice_latitude(data, save_path),
                self.plot_data_glory_tempsal_slice_longitude(data, save_path),
            ],
        }

        if plot_type not in plot_actions:
            raise ValueError(f"Invalid plot_type: {plot_type}. Choose from 'depth', 'latitude', 'longitude', or 'all'.")

        action = plot_actions[plot_type]
        action()

    def plot_data_glory_tempsal_slice_depth(self, data, save_path, generate_vmin_vmax=True, time=None):

        plot_function = self.plot_slice_depth if self.area == 'Local' else self.plot_slice_depth_global

        depths = self.data_loader.read_depths()

        vmin_vmax_list = []

        for depth in tqdm(depths[:40], desc="Processing depths"):

            data_plot = data[np.isclose(data[:, 0], depth)]

            data_plot_coor = self.data_loader.normalization_coordinates(data_plot)

            r, theta, phi, t = unpack_x(data_plot_coor)

            for i in range(2):

                column_data = data_plot[:, 4 + i]

                if generate_vmin_vmax:
                    if self.area == 'Global' and i == 1:
                        vmin_vmax = [np.percentile(column_data, 2), np.percentile(column_data, 98)]
                    else:
                        vmin_vmax = [np.percentile(column_data, 0.1), np.percentile(column_data, 99.9)]

                    vmin_vmax_list.append({
                            'depth': depth,
                            'variable_index': i,
                            'vmin_vmax': vmin_vmax
                        })
                else:
                    vmin_vmax = self.read_vmin_vmax_depth(depth, i, time)

                plot_function(lat=theta, 
                            lon=phi, 
                            values=data_plot[:, 4 + i], 
                            cmap=VAR_COLORS[i], 
                            vmin_vmax=vmin_vmax, 
                            label=VAR_UNITS[i],
                            title=f'{VAR_UNITS[i]} (Depth: {depth:.2f}m)',
                            save_path=os.path.join(save_path, 'Depth', VAR_NAMES[i], f'{VAR_NAMES[i]}_{depth:.2f}.jpg'))
        
        if generate_vmin_vmax:
            with open(os.path.join(save_path, 'vmin_vmax_depth.json'), 'w') as f:
                json.dump(vmin_vmax_list, f, indent=4)

    def plot_data_glory_tempsal_slice_latitude(self, data, save_path, generate_vmin_vmax=True):

        plot_function = self.plot_slice_latitude if self.area == 'Local' else self.plot_slice_latitude_global

        latitudes = [-20, -10, 0, 10, 20] if self.area == 'Local' else list(range(-70, 81, 10))

        unit = SLICE_LAT_UNIT if self.area == 'Local' else SLICE_LAT_UNIT_GLOBAL

        vmin_vmax_list = []

        for index, latitude in enumerate(tqdm(latitudes, desc="Processing latitudes")):

            data_plot = data[np.isclose(data[:, 1], latitude)]

            data_plot_coor = self.data_loader.normalization_coordinates(data_plot)

            r, theta, phi, t = unpack_x(data_plot_coor)

            for i in range(2):

                column_data = data_plot[:, 4 + i]

                if generate_vmin_vmax:
                    vmin_vmax = [np.percentile(column_data, 0.1), np.percentile(column_data, 99.9)]

                    vmin_vmax_list.append({
                            'latitude': latitude,
                            'variable_index': i,
                            'vmin_vmax': vmin_vmax
                        })
                else:
                    vmin_vmax = self.read_vmin_vmax_latitude(latitude, i)

                plot_function(depth=r, 
                            lon=phi, 
                            values=data_plot[:, 4 + i], 
                            cmap=VAR_COLORS[i], 
                            vmin_vmax=vmin_vmax,
                            label=VAR_UNITS[i],
                            title=f'{VAR_UNITS[i]} (Latitude: {unit[index]})',
                            save_path=os.path.join(save_path, 'Latitude', VAR_NAMES[i], f'{VAR_NAMES[i]}_{unit[index]}.jpg'))
        
        if generate_vmin_vmax:
            with open(os.path.join(save_path, 'vmin_vmax_latitude.json'), 'w') as f:
                json.dump(vmin_vmax_list, f, indent=4)


    def plot_data_glory_tempsal_slice_longitude(self, data, save_path, generate_vmin_vmax=True):

        plot_function = self.plot_slice_longitude if self.area == 'Local' else self.plot_slice_longitude_global

        unit = SLICE_LONG_UNIT if self.area == 'Local' else SLICE_LONG_UNIT_GLOBAL

        longitudes = [150, 160, 170, -180, -170, -160, -150, -140, -130, -120, -110] if self.area == 'Local' else list(range(-180, 171, 10))

        vmin_vmax_list = []

        for index, longitude in enumerate(tqdm(longitudes, desc="Processing latitudes")):

            data_plot = data[np.isclose(data[:, 2], longitude)]

            data_plot_coor = self.data_loader.normalization_coordinates(data_plot)

            r, theta, phi, t = unpack_x(data_plot_coor)

            for i in range(2):

                column_data = data_plot[:, 4 + i]

                if generate_vmin_vmax:
                    vmin_vmax = [np.percentile(column_data, 0.1), np.percentile(column_data, 99.9)]

                    vmin_vmax_list.append({
                            'longitude': longitude,
                            'variable_index': i,
                            'vmin_vmax': vmin_vmax
                        })
                else:
                    vmin_vmax = self.read_vmin_vmax_longitude(longitude, i)

                plot_function(depth=r, 
                            lat=theta, 
                            values=data_plot[:, 4 + i], 
                            cmap=VAR_COLORS[i], 
                            vmin_vmax=vmin_vmax,
                            label=VAR_UNITS[i],
                            title=f'{VAR_UNITS[i]} (Longitude: {unit[index]})',
                            save_path=os.path.join(save_path, 'Longitude', VAR_NAMES[i], f'{VAR_NAMES[i]}_{unit[index]}.jpg'))
        
        if generate_vmin_vmax:
            with open(os.path.join(save_path, 'vmin_vmax_longitude.json'), 'w') as f:
                json.dump(vmin_vmax_list, f, indent=4)

    def plot_vtheta_slice_depth(self, 
                      data_path='Data/RealData/Global/Currents/ForPlot/v_phi_plot.npy',
                      save_path='Output/Plot/RealData/Global/Data'):
        
        data = np.load(data_path)

        self.plot_vtheta_slice_depth_(data, save_path)


    def plot_vtheta_slice_depth_(self, data, save_path='Output/Plot/RealData/Global/Data', generate_vmin_vmax=True):
        
        depths = self.data_loader.read_depths()

        vmin_vmax_list = []

        for depth in tqdm(depths[:40], desc="Processing depths"):

            data_plot = data[np.isclose(data[:, 0], depth)]

            data_plot_coor = self.data_loader.normalization_coordinates(data_plot)

            r, theta, phi, t = unpack_x(data_plot_coor)

            column_data = data_plot[:, 4]

            if generate_vmin_vmax:
                abs_column_data = np.abs(column_data)
                percentile = np.percentile(abs_column_data, 99.5)
                vmin_vmax = [-percentile, percentile]
                vmin_vmax_list.append({
                        'depth': depth,
                        'vmin_vmax': vmin_vmax
                    })
            else:
                vmin_vmax = self.read_vmin_vmax_depth_vtheta(depth)

            self.plot_slice_depth_global(lat=theta, 
                        lon=phi, 
                        values=data_plot[:, 4], 
                        cmap=VAR_COLORS[3], 
                        vmin_vmax=vmin_vmax, 
                        label=VAR_UNITS[3],
                        title=f'{VAR_UNITS[3]} (Depth: {depth:.2f}m)',
                        save_path=os.path.join(save_path, 'Depth', VAR_NAMES[3], f'{VAR_NAMES[3]}_{depth:.2f}.jpg'))
        
        if generate_vmin_vmax:
            with open(os.path.join(save_path, 'vmin_vmax_depth_vtheta.json'), 'w') as f:
                json.dump(vmin_vmax_list, f, indent=4)

    def read_vmin_vmax_polar(self, region, depth, variable_index, json_path='Output/Plot/RealData/Global/Data/vmin_vmax_polar.json'):
        """
        Reads the vmin_vmax values for a specific region, depth, and variable index from a JSON file.

        Parameters:
        json_path (str): Path to the JSON file.
        region (str): The region (e.g., 'Arctic' or 'Antarctic').
        depth (float): The depth value.
        variable_index (int): The variable index.

        Returns:
        list: The vmin_vmax values as [vmin, vmax]. Returns None if no match is found.
        """
        # Check if the file exists
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"File {json_path} does not exist")

        # Read the JSON file
        with open(json_path, 'r') as f:
            vmin_vmax_list = json.load(f)

        # Find the matching entry
        for entry in vmin_vmax_list:
            if (
                entry['region'] == region
                and entry['depth'] == depth
                and entry['variable_index'] == variable_index
            ):
                return entry['vmin_vmax']

        # Return None if no match is found
        return None

    def read_vmin_vmax_depth(self, depth, variable_index, time=None):
        """
        Reads the vmin_vmax values for a specific depth and variable index from a JSON file.

        Parameters:
        json_path (str): Path to the JSON file.
        depth (float): The depth value.
        variable_index (int): The variable index.

        Returns:
        list: The vmin_vmax values as [vmin, vmax]. Returns None if no match is found.
        """

        json_path = f'Output/Plot/RealData/{self.area}/Data{time}/vmin_vmax_depth.json'

        # Check if the file exists
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"File {json_path} does not exist")

        # Read the JSON file
        with open(json_path, 'r') as f:
            vmin_vmax_list = json.load(f)

        # Find the matching entry
        for entry in vmin_vmax_list:
            if (
                entry['depth'] == depth
                and entry['variable_index'] == variable_index
            ):
                return entry['vmin_vmax']

        # Return None if no match is found
        return None

    def read_vmin_vmax_depth_vtheta(self, depth):

        json_path = f'Output/Plot/RealData/{self.area}/Data/vmin_vmax_depth_vtheta.json'

        # Check if the file exists
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"File {json_path} does not exist")

        # Read the JSON file
        with open(json_path, 'r') as f:
            vmin_vmax_list = json.load(f)

        # Find the matching entry
        for entry in vmin_vmax_list:
            if (
                entry['depth'] == depth
            ):
                return entry['vmin_vmax']

        # Return None if no match is found
        return None

    def read_vmin_vmax_latitude(self, latitude, variable_index):
        """
        Reads the vmin_vmax values for a specific latitude and variable index from a JSON file.

        Parameters:
        json_path (str): Path to the JSON file.
        latitude (float): The latitude value.
        variable_index (int): The variable index.

        Returns:
        list: The vmin_vmax values as [vmin, vmax]. Returns None if no match is found.
        """

        json_path = f'Output/Plot/RealData/{self.area}/Data/vmin_vmax_latitude.json'

        # Check if the file exists
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"File {json_path} does not exist")

        # Read the JSON file
        with open(json_path, 'r') as f:
            vmin_vmax_list = json.load(f)

        # Find the matching entry
        for entry in vmin_vmax_list:
            if (
                entry['latitude'] == latitude
                and entry['variable_index'] == variable_index
            ):
                return entry['vmin_vmax']

        # Return None if no match is found
        return None

    def read_vmin_vmax_longitude(self, longitude, variable_index):
        """
        Reads the vmin_vmax values for a specific longitude and variable index from a JSON file.

        Parameters:
        json_path (str): Path to the JSON file.
        longitude (float): The longitude value.
        variable_index (int): The variable index.

        Returns:
        list: The vmin_vmax values as [vmin, vmax]. Returns None if no match is found.
        """

        json_path = f'Output/Plot/RealData/{self.area}/Data/vmin_vmax_longitude.json'

        # Check if the file exists
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"File {json_path} does not exist")

        # Read the JSON file
        with open(json_path, 'r') as f:
            vmin_vmax_list = json.load(f)

        # Find the matching entry
        for entry in vmin_vmax_list:
            if (
                entry['longitude'] == longitude
                and entry['variable_index'] == variable_index
            ):
                return entry['vmin_vmax']

        # Return None if no match is found
        return None
    
    def plot_data_seaice(self, data_path='Data/RealData/Global/SeaIce', save_path='Output/Plot/RealData/Global/DataSeaIce'):
        seaice_all_path = os.path.join(data_path, 'seaice_all.npy')
        seaice_all = np.load(seaice_all_path)
        seaice_all = seaice_all[seaice_all[:, 3] > 0]

        vmin_vmax = [np.min(seaice_all[:, 3]), np.max(seaice_all[:, 3])]

        t_list = np.unique(seaice_all[:, 2])
        for i, t in enumerate(tqdm(t_list, desc="Processing timestamps")):
            seaice_all_plot = seaice_all[seaice_all[:, 2] == t]
            seaice_all_plot = seaice_all_plot[seaice_all_plot[:, 3] > 0]

            lats = seaice_all_plot[:, 0]
            lons = seaice_all_plot[:, 1]
            values = seaice_all_plot[:, 3]

            date = datetime.fromtimestamp(t, tz=timezone.utc)

            self.plot_polar_scatter_cartopy(lons, lats, values, 
                                             var_name='Sea Ice Thickness', 
                                             cmap='PuBu_r',
                                             figsize=(8, 6),
                                             point_size=0.2,
                                             central_lon=0,
                                             output_path=os.path.join(save_path, 'Arctic', date.strftime('%Y-%m-%d %H:%M:%S')),
                                             vmin_vmax=vmin_vmax,
                                             region='Arctic')
            self.plot_polar_scatter_cartopy(lons, lats, values, 
                                             var_name='Sea Ice Thickness', 
                                             cmap='PuBu_r',
                                             figsize=(8, 6),
                                             point_size=0.2,
                                             central_lon=0,
                                             output_path=os.path.join(save_path, 'Antarctic', date.strftime('%Y-%m-%d %H:%M:%S')),
                                             vmin_vmax=vmin_vmax,
                                             region='Antarctic')

    
    # Cartopy version (recommended)
    def plot_polar_scatter_cartopy(self, lons, lats, values, 
                                var_name='Variable', 
                                cmap='viridis',
                                figsize=(8, 6),
                                point_size=20,
                                central_lon=0,
                                output_path=None,
                                vmin_vmax=[None, None],
                                region='Arctic',
                                log_transform=True,
                                face_color=True,
                                depth=None):
        """
        Polar scatter plot using Cartopy (recommended for accurate projection)
        
        Parameters:
        lons (array): Longitude values (-180 to 180 or 0-360)
        lats (array): Latitude values (60-90°N for Arctic, 60-90°S for Antarctic)
        region (str): 'Arctic' or 'Antarctic' to specify polar region
        """

        # Convert longitude if needed
        if np.any(lons > 180):
            lons = (lons + 180) % 360 - 180

        # Create projection based on region
        if region.lower() == 'arctic':
            proj = ccrs.NorthPolarStereo(central_longitude=central_lon)
            lat_limit = 50 if region.lower() == 'arctic' else -50
            extent = [-180, 180, 50, 90]
            title_region = 'Arctic'
        elif region.lower() == 'antarctic':
            proj = ccrs.SouthPolarStereo(central_longitude=central_lon)
            extent = [-180, 180, -90, -55]
            title_region = 'Antarctic'
            # Flip latitudes for southern hemisphere
            # lats = -np.abs(lats)
        else:
            raise ValueError("Region must be 'Arctic' or 'Antarctic'")

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection=proj)
        
        # Set map extent
        ax.set_extent(extent, crs=ccrs.PlateCarree())
        
        # Plot data
        if log_transform:
            sc = ax.scatter(lons, lats, c=values, 
                            cmap=cmap, s=point_size,
                            transform=ccrs.PlateCarree(),
                            edgecolor='none',
                            norm=LogNorm(vmin=vmin_vmax[0], vmax=vmin_vmax[1]))
        else:
            sc = ax.scatter(lons, lats, c=values, 
                            cmap=cmap, s=point_size,
                            transform=ccrs.PlateCarree(),
                            edgecolor='none',
                            vmin=vmin_vmax[0], vmax=vmin_vmax[1])

        # Add features
        cmap_obj = plt.get_cmap(cmap)
        start_color = cmap_obj(0.0)
        end_color = cmap_obj(1.0)
        if face_color:
            ax.add_feature(cfeature.OCEAN, facecolor=start_color)
        else:
            ax.add_feature(cfeature.OCEAN, facecolor='white')
        ax.add_feature(cfeature.LAND, facecolor='white')
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        
        # Add gridlines
        gl = ax.gridlines(linestyle='--', color='gray', linewidth=0.5,
                        draw_labels=True, xlocs=range(-180, 181, 30))
        gl.top_labels = False
        if region.lower() == 'antarctic':
            gl.xlabel_style = {'rotation': 20}  # Adjust label rotation for southern labels
        
        # Colorbar and title
        cbar = plt.colorbar(sc, ax=ax, shrink=0.8)
        cbar.set_label(var_name)

        if depth is None:
            title = f'{title_region} {var_name} Distribution\n({"North" if region.lower() == "arctic" else "South"} Polar Stereographic)'
        else:
            title = f'{title_region} {var_name} Distribution\n({"North" if region.lower() == "arctic" else "South"} Polar Stereographic, depth: {depth:.2f}m)'
        plt.title(title)
        
        if output_path:
            gen_folder(os.path.dirname(output_path))
            plt.savefig(output_path, bbox_inches='tight', dpi=300)
        else:
            plt.show()
        plt.close()

    
    def plot_slice_depth(self, lat, lon, values, cmap, vmin_vmax, label, title, save_path):

        plt.rcParams['font.size'] = 15

        fig, ax = plt.subplots(figsize=(10, 4), dpi=300)

        im = ax.scatter(lon, lat, c=values, cmap=cmap, vmin=vmin_vmax[0], vmax=vmin_vmax[1], s=1.5, linewidths=0)
                
        cbar = fig.colorbar(im, label=label, pad=0.03)
        # if k in [0, 1]:
        #     cbar.ax.set_position([0.78, 0.085, 0.028, 0.8])
        ax.set_ylabel('Latitude')
        ax.set_xlabel('Longitude')
        ax.set_title(title)
        ax.set_yticks(np.deg2rad([110, 100, 90, 80, 70]))
        ax.set_yticklabels(['20°S', '10°S', '0°', '10°N', '20°N'])
        ax.set_xticks(np.deg2rad([0, 20, 40, 60, 80, 100]))
        ax.set_xticklabels(['150°E', '170°E', '170°W', '150°W', '130°W', '110°W'])
        ax.set_xlim(np.deg2rad([0, 100]))
        ax.set_ylim(np.deg2rad([110, 70]))

        gen_folder(os.path.dirname(save_path))
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()

    def plot_slice_depth_global(self, lat, lon, values, cmap, vmin_vmax, label, title, save_path):

        plt.rcParams['font.size'] = 15

        fig, ax = plt.subplots(figsize=(14, 7), dpi=300)

        im = ax.scatter(lon, lat, c=values, cmap=cmap, vmin=vmin_vmax[0], vmax=vmin_vmax[1], s=0.2, linewidths=0)
                
        cbar = fig.colorbar(im, label=label, pad=0.03)
        # if k in [0, 1]:
        #     cbar.ax.set_position([0.78, 0.085, 0.028, 0.8])
        ax.set_ylabel('Latitude')
        ax.set_xlabel('Longitude')
        ax.set_title(title)
        ax.set_yticks(np.deg2rad([0, 45, 90, 135, 180][::-1]))
        ax.set_yticklabels(['90°S', '45°S', '0°', '45°N', '90°N'])
        ax.set_xticks(np.deg2rad([0, 45, 90, 135, 180, 225, 270, 315, 360]))
        ax.set_xticklabels(['180°', '135°W', '90°W', '45°W', '0°', '45°E', '90°E', '135°E', '180°'])
        
        ax.set_ylim(np.deg2rad([180, 0]))
        ax.set_xlim(np.deg2rad([0, 360]))

        gen_folder(os.path.dirname(save_path))
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()



    def plot_slice_latitude(self, depth, lon, values, cmap, vmin_vmax, label, title, save_path):

        plt.rcParams['font.size'] = 10

        fig, ax = plt.subplots(figsize=(10, 2), dpi=300)

        im = ax.scatter(lon, depth, c=values, cmap=cmap, vmin=vmin_vmax[0], vmax=vmin_vmax[1], s=2, linewidths=0)
                
        cbar = fig.colorbar(im, label=label, pad=0.02)
        # if k in [0, 1]:
        #     cbar.ax.set_position([0.78, 0.085, 0.028, 0.8])
        ax.set_ylabel('Depth')
        ax.set_xlabel('Longitude')
        ax.set_title(title)
        ax.set_yticks([-0.4, -0.2, 0])
        ax.set_yticklabels(['2000m', '1000m', '0m'])
        ax.set_xticks(np.deg2rad([0, 20, 40, 60, 80, 100]))
        ax.set_xticklabels(['150°E', '170°E', '170°W', '150°W', '130°W', '110°W'])
        ax.set_ylim([-0.4, 0])
        ax.set_xlim(np.deg2rad([0, 100]))

        gen_folder(os.path.dirname(save_path))
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()


    def plot_slice_latitude_global(self, depth, lon, values, cmap, vmin_vmax, label, title, save_path):

        plt.rcParams['font.size'] = 10

        fig, ax = plt.subplots(figsize=(10, 2), dpi=300)

        im = ax.scatter(lon, depth, c=values, cmap=cmap, vmin=vmin_vmax[0], vmax=vmin_vmax[1], s=1, linewidths=0)
                
        cbar = fig.colorbar(im, label=label, pad=0.02)
        # if k in [0, 1]:
        #     cbar.ax.set_position([0.78, 0.085, 0.028, 0.8])
        ax.set_ylabel('Depth')
        ax.set_xlabel('Longitude')
        ax.set_title(title)
        ax.set_yticks([-0.4, -0.2, 0])
        ax.set_yticklabels(['2000m', '1000m', '0m'])
        ax.set_xticks(np.deg2rad([0, 45, 90, 135, 180, 225, 270, 315, 360]))
        ax.set_xticklabels(['180°', '135°W', '90°W', '45°W', '0°', '45°E', '90°E', '135°E', '180°'])
        ax.set_ylim([-0.4, 0])
        ax.set_xlim(np.deg2rad([0, 360]))

        gen_folder(os.path.dirname(save_path))
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()

    def plot_slice_longitude(self, depth, lat, values, cmap, vmin_vmax, label, title, save_path):

        plt.rcParams['font.size'] = 10

        fig, ax = plt.subplots(figsize=(4, 2), dpi=300)

        im = ax.scatter(lat, depth, c=values, cmap=cmap, vmin=vmin_vmax[0], vmax=vmin_vmax[1], s=2, linewidths=0)
                
        cbar = fig.colorbar(im, label=label, pad=0.05)

        ax.set_ylabel('Depth')
        ax.set_xlabel('Latitude')
        ax.set_title(title)
        ax.set_yticks([-0.4, -0.2, 0])
        ax.set_yticklabels(['2000m', '1000m', '0m'])
        ax.set_xticks(np.deg2rad([110, 100, 90, 80, 70]))
        ax.set_xticklabels(['20°S', '10°S', '0°', '10°N', '20°N'])
        ax.set_ylim([-0.4, 0])
        ax.set_xlim(np.deg2rad([110, 70]))

        gen_folder(os.path.dirname(save_path))
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()

    def plot_slice_longitude_global(self, depth, lat, values, cmap, vmin_vmax, label, title, save_path):

        plt.rcParams['font.size'] = 10

        fig, ax = plt.subplots(figsize=(6, 2), dpi=300)

        im = ax.scatter(lat, depth, c=values, cmap=cmap, vmin=vmin_vmax[0], vmax=vmin_vmax[1], s=1, linewidths=0)
                
        cbar = fig.colorbar(im, label=label, pad=0.05)

        ax.set_ylabel('Depth')
        ax.set_xlabel('Latitude')
        ax.set_title(title)

        ax.set_yticks([-0.4, -0.2, 0])
        ax.set_yticklabels(['2000m', '1000m', '0m'])
        ax.set_xticks(np.deg2rad([0, 45, 90, 135, 180][::-1]))
        ax.set_xticklabels(['90°S', '45°S', '0°', '45°N', '90°N'])
        ax.set_ylim([-0.4, 0])
        ax.set_xlim(np.deg2rad([180, 0]))

        gen_folder(os.path.dirname(save_path))
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()

    
    def plot_rmse_distribution_spacial(self, model_pre_path, grid_step=2, save_path='Output/Plot/RealData/RMSE', method_name='MAD-REF Net', plot_seperate=False):
        save_path = os.path.join(save_path, method_name, 'Horizontal')

        # Read test data and prediction results
        data_argo_test, data_cur_test, data_wcur_test = self.data_loader.read_data(type='test')
        x_argo, y_temp, y_sal, x_cur, y_v_theta, y_v_phi, x_wcur, y_w = data_split(data_argo_test, data_cur_test, data_wcur_test)
        temp_pre, sal_pre, w_pre, v_theta_pre, v_phi_pre = self.data_loader.read_pred_results(model_pre_path)

        r_min, r_max, theta_min, theta_max, phi_min, phi_max, t_min, t_max = self.data_loader.get_parameters()

        def calculate_rmse_grid(y_true, y_pred, lats, lons, step):
            # Use a safer boundary calculation method
            lat_min = theta_min
            lat_max = theta_max
            lon_min = phi_min
            lon_max = phi_max

            # Add a protective offset when generating grid boundaries to avoid floating-point errors
            lat_edges = np.arange(lat_min, lat_max + 1e-9, step)  
            lon_edges = np.arange(lon_min, lon_max + 1e-9, step)
            
            # Correct index calculation
            lat_idx = np.clip(np.digitize(lats, lat_edges) - 1, 0, len(lat_edges)-2)
            lon_idx = np.clip(np.digitize(lons, lon_edges) - 1, 0, len(lon_edges)-2)
            
            # Ensure the matrix shape is correct
            shape = (len(lat_edges)-1, len(lon_edges)-1)
            
            data = (y_true - y_pred)**2
            sum_matrix = coo_matrix((data, (lat_idx, lon_idx)), shape=shape).toarray()
            count_matrix = coo_matrix((np.ones_like(data), (lat_idx, lon_idx)), shape=shape).toarray()
            
            # Handle empty grids
            valid_mask = count_matrix > 0
            rmse_grid = np.full(shape, np.nan)
            rmse_grid[valid_mask] = np.sqrt(sum_matrix[valid_mask] / count_matrix[valid_mask])
            
            return rmse_grid, lat_edges, lon_edges

        def plot_grid(rmse_grid, lat_bins, lon_bins, title, save_path, label='RMSE', vmin_vmax=[None, None], log_scale=True):
            plt.figure(figsize=(10, 5), dpi=300)
            
            # Plot the data with optional log transform
            if log_scale:
                norm = colors.LogNorm(vmin=vmin_vmax[0], vmax=vmin_vmax[1])
            else:
                norm = colors.Normalize(vmin=vmin_vmax[0], vmax=vmin_vmax[1])
            
            mesh = plt.pcolormesh(lon_bins, lat_bins, rmse_grid, shading='flat', cmap='YlGnBu_r', norm=norm)
            cbar = plt.colorbar(mesh, label=label)
            
            # Set axis labels with E/W and N/S notation
            # Control the density of longitude labels (every 45 degrees)
            lon_ticks = np.arange(np.floor(lon_bins.min() / 45) * 45, 
                                np.ceil(lon_bins.max() / 45) * 45 + 1, 45)
            lon_labels = [f'{abs(lon):.0f}°W' if lon < 0 else f'{lon:.0f}°E' for lon in lon_ticks]
            lon_labels = ['0°' if lon == 0 else lon for lon in lon_labels]  # Handle 0 longitude
            plt.xticks(lon_ticks, labels=lon_labels, rotation=0, ha='center')  # No rotation
            
            # Control the density of latitude labels (every 45 degrees)
            lat_ticks = np.arange(np.floor(lat_bins.min() / 45) * 45, 
                                np.ceil(lat_bins.max() / 45) * 45 + 1, 45)
            lat_labels = [f'{abs(lat):.0f}°S' if lat < 0 else f'{lat:.0f}°N' for lat in lat_ticks]
            lat_labels = ['0°' if lat == 0 else lat for lat in lat_labels]  # Handle 0 latitude
            plt.yticks(lat_ticks, labels=lat_labels)
            
            # Add labels and title
            plt.xlabel('Longitude')
            plt.ylabel('Latitude')
            plt.title(title)
            
            # Save the figure
            gen_folder(os.path.dirname(save_path))
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()

        variable_weights = self.var_weights(y_temp, y_sal, y_w, y_v_theta, y_v_phi)
        rmse_grids = {}

        # Process and plot for each variable
        # 1. Temperature
        rmse_temp, lat_bins, lon_bins = calculate_rmse_grid(
            y_temp, temp_pre, x_argo[:,1], x_argo[:,2], grid_step
        )
        rmse_grids['temp'] = rmse_temp
        # 2. Salinity
        rmse_sal, _, _ = calculate_rmse_grid(
            y_sal, sal_pre, x_argo[:,1], x_argo[:,2], grid_step
        )
        rmse_grids['sal'] = rmse_sal
        # 3. Vertical velocity (assuming the use of x_wcur's latitude and longitude)
        rmse_w, lat_bins_w, lon_bins_w = calculate_rmse_grid(
            y_w, w_pre, x_wcur[:,1], x_wcur[:,2], grid_step
        )
        rmse_grids['w'] = rmse_w
        # 4. Meridional velocity (assuming the use of x_cur's latitude and longitude)
        rmse_v_theta, lat_bins_cur, lon_bins_cur = calculate_rmse_grid(
            y_v_theta, v_theta_pre, x_cur[:,1], x_cur[:,2], grid_step
        )
        rmse_grids['v_theta'] = rmse_v_theta
        # 5. Zonal velocity
        rmse_v_phi, _, _ = calculate_rmse_grid(
            y_v_phi, v_phi_pre, x_cur[:,1], x_cur[:,2], grid_step
        )
        rmse_grids['v_phi'] = rmse_v_phi

        if plot_seperate:
            plot_grid(rmse_temp, lat_bins, lon_bins, 'Temperature RMSE Distribution', save_path=os.path.join(save_path, 'temp_rmse_grid.png'))
            plot_grid(rmse_sal, lat_bins, lon_bins, 'Salinity RMSE Distribution', save_path=os.path.join(save_path, 'sal_rmse_grid.png'))
            plot_grid(rmse_w, lat_bins_w, lon_bins_w, 'Vertical Velocity RMSE Distribution', save_path=os.path.join(save_path, 'w_rmse_grid.png'))
            plot_grid(rmse_v_theta, lat_bins_cur, lon_bins_cur, 'Meridional Velocity RMSE Distribution', save_path=os.path.join(save_path, 'v_theta_rmse_grid.png'))
            plot_grid(rmse_v_phi, lat_bins_cur, lon_bins_cur, 'Zonal Velocity RMSE Distribution', save_path=os.path.join(save_path, 'v_phi_rmse_grid.png'))


        def calculate_integrated_rmse(rmse_dict, weights):
            # Initialize valid weights sum and weighted sum matrices
            valid_weights_sum = np.zeros_like(rmse_dict['temp'])
            weighted_sum = np.zeros_like(rmse_dict['temp'])
            
            # Iterate over each variable
            for var in rmse_dict:
                # Get the RMSE matrix for the current variable
                rmse = rmse_dict[var]
                # Create a valid weight mask (True for non-NaN positions)
                valid_mask = ~np.isnan(rmse)
                # Calculate the effective weights for the current variable
                curr_weights = np.where(valid_mask, variable_weights[var], 0.0)
                
                # Accumulate the effective weights
                valid_weights_sum += curr_weights
                # Accumulate the weighted RMSE (replace NaN with 0 for calculation)
                weighted_sum += np.nan_to_num(rmse) * curr_weights
            
            # Calculate the integrated RMSE (handle division by zero)
            integrated = np.full_like(weighted_sum, np.nan)
            valid_cells = valid_weights_sum == sum(weights.values())
            integrated[valid_cells] = weighted_sum[valid_cells] / valid_weights_sum[valid_cells]
            
            return integrated

        integrated_rmse = calculate_integrated_rmse(rmse_grids, variable_weights)

        plot_grid(integrated_rmse, lat_bins, lon_bins, 
                f'Variation of sRMSE with respect to latitude and longitude',
                os.path.join(save_path, 'integrated_rmse_grid.png'),
                label='sRMSE',
                vmin_vmax=[1e-2, 5e-1])

    def plot_rmse_temporal(self, model_pre_path, time_step='ME', 
                      save_path='Output/Plot/RealData/RMSE', 
                      method_name='MAD-REF Net',
                      plot_seperate=False):
        save_path = os.path.join(save_path, method_name, 'Temporal')
        gen_folder(save_path)

        # Read test data and prediction results
        data_argo_test, data_cur_test, data_wcur_test = self.data_loader.read_data(type='test')
        x_argo, y_temp, y_sal, x_cur, y_v_theta, y_v_phi, x_wcur, y_w = data_split(data_argo_test, data_cur_test, data_wcur_test)
        temp_pre, sal_pre, w_pre, v_theta_pre, v_phi_pre = self.data_loader.read_pred_results(model_pre_path)

        # Extract timestamps for each dataset
        timestamps_argo = pd.to_datetime(x_argo[:, 3], unit='s')
        timestamps_cur = pd.to_datetime(x_cur[:, 3], unit='s')
        timestamps_wcur = pd.to_datetime(x_wcur[:, 3], unit='s')

        r_min, r_max, theta_min, theta_max, phi_min, phi_max, t_min, t_max = self.data_loader.get_parameters()
        t_min = pd.to_datetime(t_min, unit='s')
        t_max = pd.to_datetime(t_max, unit='s')

        def calculate_rmse_time(y_true, y_pred, timestamps, time_step):
            # Create time bins
            df = pd.DataFrame({
                'true': y_true,
                'pred': y_pred,
                'time': timestamps
            })
            
            # Resample to time bins
            resampled = df.resample(time_step, on='time').agg({
                'true': lambda x: np.sqrt(np.mean((x - y_pred[x.index])**2)),
                'pred': 'count'
            })
            
            # Handle empty bins
            resampled['rmse'] = resampled['true']
            resampled.loc[resampled['pred'] == 0, 'rmse'] = np.nan
            
            return resampled.index.to_numpy(), resampled['rmse'].to_numpy()

        def plot_time_series(time_bins, rmse_values, title, save_path, log_scale=False):
            plt.figure(figsize=(12, 3.5), dpi=300)
            
            # Plot time series
            plt.plot(time_bins, rmse_values, 
                    marker='o', linestyle='-', 
                    color='#2c7bb6', linewidth=2, 
                    markersize=6)
            
            # Formatting
            plt.xlabel('Time')
            plt.ylabel('RMSE')
            plt.title(title)
            plt.grid(True, alpha=0.3)
            # plt.xticks(rotation=45)
            plt.tight_layout()
            
            # Apply log scale to y-axis if enabled
            if log_scale:
                plt.yscale('log')
                plt.ylabel('RMSE (log scale)')

            # Set y-axis to scientific notation
            formatter = ticker.ScalarFormatter(useMathText=True)
            formatter.set_scientific(True)
            formatter.set_powerlimits((-3, 3))
            plt.gca().yaxis.set_major_formatter(formatter)
            
            # Save figure
            gen_folder(os.path.dirname(save_path))
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()

            print(f"Figure saved to '{save_path}'")

        # Calculate RMSE time series for each variable
        variables = {
            'temp': (y_temp, temp_pre, timestamps_argo),
            'sal': (y_sal, sal_pre, timestamps_argo),
            'w': (y_w, w_pre, timestamps_wcur),
            'v_theta': (y_v_theta, v_theta_pre, timestamps_cur),
            'v_phi': (y_v_phi, v_phi_pre, timestamps_cur)
        }

        weights = self.var_weights(y_temp, y_sal, y_w, y_v_theta, y_v_phi)

        if plot_seperate:
            # Plot time series for each variable
            for var_name, (y_true, y_pred, timestamps) in variables.items():
                time_bins, rmse_values = calculate_rmse_time(y_true, y_pred, timestamps, time_step)

                print(f"RMSE for {var_name}: {np.mean(rmse_values)}")
                print(rmse_values)

                plot_time_series(time_bins, rmse_values, 
                                f'{var_name.capitalize()} RMSE Temporal Distribution - {method_name}',
                                os.path.join(save_path, f'{var_name}_rmse_temporal.png'))

        # Calculate integrated sRMSE time series
        def calculate_integrated_srmse(time_step):
            all_rmse = {}
            
            # Get time series for each variable
            for var_name, (y_true, y_pred, timestamps) in variables.items():
                time_bins, rmse_values = calculate_rmse_time(y_true, y_pred, timestamps, time_step)
                all_rmse[var_name] = pd.Series(rmse_values, index=time_bins)
            
            # Create DataFrame
            df = pd.DataFrame(all_rmse)
            
            # Weighted calculation
            df_weighted = df.mul(pd.Series(weights), axis=1)
            df['sRMSE'] = df_weighted.sum(axis=1) / sum(weights.values())
            
            return df.index.to_numpy(), df['sRMSE'].to_numpy()

        # Plot integrated sRMSE
        time_bins_, srmse_values_ = calculate_integrated_srmse(time_step)
        plot_time_series(time_bins_, srmse_values_,
                        f'sRMSE Temporal Distribution of {method_name}',
                        os.path.join(save_path, 'integrated_srmse_temporal.png'))

    
    def plot_combined_rmse_temporal(self, method_info_list, time_step='ME',
                               save_path='Output/Plot/RealData/RMSE/Combined',
                               fig_name='combined_rmse_comparison.png'):
        
        """
        Plot combined sRMSE temporal curves for multiple methods in one figure.
        
        Args:
            method_info_list: List of tuples containing (model_pre_path, method_name)
            time_step: Time frequency for resampling (default 'ME' for monthly)
            save_path: Path to save the combined plot
            fig_name: Filename for the output figure
        """

        # Create save directory
        gen_folder(save_path)
        
        # Store data for all methods
        all_data = {}

        # Reuse existing calculation logic from plot_rmse_temporal
        data_argo_test, data_cur_test, data_wcur_test = self.data_loader.read_data(type='test')
        x_argo, y_temp, y_sal, x_cur, y_v_theta, y_v_phi, x_wcur, y_w = data_split(
            data_argo_test, data_cur_test, data_wcur_test)

        # Extract timestamps
        timestamps_argo = pd.to_datetime(x_argo[:, 3], unit='s')
        timestamps_cur = pd.to_datetime(x_cur[:, 3], unit='s')
        timestamps_wcur = pd.to_datetime(x_wcur[:, 3], unit='s')

        weights = self.var_weights(y_temp, y_sal, y_w, y_v_theta, y_v_phi)
        
        # Process each method
        for item in tqdm(method_info_list, desc='Processing methods'):
            
            model_pre_path, method_name = item['model_pre_path'], item['method_name']

            temp_pre, sal_pre, w_pre, v_theta_pre, v_phi_pre = self.data_loader.read_pred_results(model_pre_path)

            # Define variables and weights
            variables = {
                'temp': (y_temp, temp_pre, timestamps_argo),
                'sal': (y_sal, sal_pre, timestamps_argo),
                'w': (y_w, w_pre, timestamps_wcur),
                'v_theta': (y_v_theta, v_theta_pre, timestamps_cur),
                'v_phi': (y_v_phi, v_phi_pre, timestamps_cur)
            }


            # Calculate integrated sRMSE
            def _calculate_rmse(y_true, y_pred, timestamps):
                """Reusable RMSE calculation helper"""
                df = pd.DataFrame({
                    'true': y_true,
                    'pred': y_pred,
                    'time': timestamps
                })
                resampled = df.resample(time_step, on='time').agg({
                    'true': lambda x: np.sqrt(np.mean((x - y_pred[x.index])**2)),
                    'pred': 'count'
                })
                resampled['rmse'] = resampled['true']
                resampled.loc[resampled['pred'] == 0, 'rmse'] = np.nan
                return resampled['rmse']

            # Calculate weighted sRMSE
            rmse_components = []
            for var_name, (y_true, y_pred, timestamps) in tqdm(variables.items(), desc='Calculating RMSE'):
                rmse = _calculate_rmse(y_true, y_pred, timestamps)
                weighted_rmse = rmse * weights[var_name]
                rmse_components.append(weighted_rmse)
            
            total_weights = sum(weights.values())
            srmse = pd.concat(rmse_components, axis=1).sum(axis=1) / total_weights
            
            # Store results
            all_data[method_name] = {
                'time': srmse.index.to_numpy(),
                'srmse': srmse.to_numpy()
            }

        # Create combined plot
        plt.figure(figsize=(12, 4), dpi=300)
        
        # Color palette for methods
        colors = [METHOD_COLORS['MAD-REF Net'], METHOD_COLORS['Neural Field'], METHOD_COLORS['LightGBM'], METHOD_COLORS['SparseGP']]
        
        # Plot each method's curve
        for (method_name, data), color in zip(all_data.items(), colors):
            plt.plot(data['time'], data['srmse'],
                    marker=METHOD_MARKERS[method_name],
                    linestyle='-',
                    linewidth=2,
                    markersize=6,
                    color=color,
                    label=method_name)

        # Plot formatting
        plt.xlabel('Time', fontsize=12)
        plt.ylabel('sRMSE', fontsize=12)
        plt.title('Variation of sRMSE with respect to time', fontsize=14)
        plt.legend(fontsize=10, loc='upper right')
        plt.grid(True, alpha=0.3)
        
        # Scientific notation for y-axis
        plt.gca().yaxis.set_major_formatter(
            ticker.ScalarFormatter(useMathText=True))
        plt.gca().yaxis.set_major_formatter(
            ticker.FuncFormatter(lambda x, pos: f"{x:.1e}"))

        # Apply log transformation to y-axis
        plt.yscale('log')

        # Scientific notation for y-axis in log scale
        plt.gca().yaxis.set_major_formatter(ticker.LogFormatterSciNotation())
        plt.gca().yaxis.set_minor_formatter(ticker.LogFormatterSciNotation(minor_thresholds=(2, 0.4)))

        plt.ylim(bottom=4e-2)

        # Save combined plot
        save_file = os.path.join(save_path, fig_name)
        plt.savefig(save_file, bbox_inches='tight')
        plt.close()
        print(f"Combined RMSE plot saved to: {save_file}")

    
    def plot_rmse_depth(self, model_pre_path, depth_bins=20, depth_bin_power=2,
                        save_path='Output/Plot/RealData/RMSE', 
                        method_name='MAD-REF Net',
                        plot_seperate=False):
        save_path = os.path.join(save_path, method_name, 'Depth')
        gen_folder(save_path)

        # Read test data and prediction results
        data_argo_test, data_cur_test, data_wcur_test = self.data_loader.read_data(type='test')
        x_argo, y_temp, y_sal, x_cur, y_v_theta, y_v_phi, x_wcur, y_w = data_split(data_argo_test, data_cur_test, data_wcur_test)
        temp_pre, sal_pre, w_pre, v_theta_pre, v_phi_pre = self.data_loader.read_pred_results(model_pre_path)

        # Extract depth data for each variable
        depths_argo = x_argo[:, 0]
        depths_cur = x_cur[:, 0]
        depths_wcur = x_wcur[:, 0]

        # Generate non-uniform bins (more dense near the maximum depth)
        depth_min = min(np.min(depths_argo), np.min(depths_cur), np.min(depths_wcur))
        depth_max = max(np.max(depths_argo), np.max(depths_cur), np.max(depths_wcur))
        
        # Use a power function to create non-linear bins
        n_bins = depth_bins + 1
        x = np.linspace(0, 1, n_bins)
        s = 1 - (1 - x) ** depth_bin_power  # Bins become denser as depth increases
        depth_bins_edges = depth_min + (depth_max - depth_min) * s

        weights = self.var_weights(y_temp, y_sal, y_w, y_v_theta, y_v_phi)

        def calculate_rmse_depth(y_true, y_pred, depths, bins):
            df = pd.DataFrame({'true': y_true, 'pred': y_pred, 'depth': depths})
            df['depth_bin'] = pd.cut(df['depth'], bins=bins, include_lowest=True)
            grouped = df.groupby('depth_bin', observed=False).apply(
                lambda x: np.sqrt(np.mean((x['true'] - x['pred'])**2)) if len(x) > 0 else np.nan
            )
            grouped = grouped.reset_index()
            grouped['depth_mid'] = grouped['depth_bin'].apply(lambda x: x.mid)
            grouped = grouped.sort_values('depth_mid')
            return grouped['depth_mid'].values, grouped[0].values

        # Core logic for calculating weighted sRMSE
        def calculate_integrated_srmse(bins_edges):
            srmse_list = []
            depth_mids = []
            
            # Iterate over each bin interval
            for i in tqdm(range(len(bins_edges)-1), desc='Calculating integrated sRMSE'):
                left, right = bins_edges[i], bins_edges[i+1]
                mid = (left + right) / 2  # Calculate the midpoint of the bin
                
                total_weighted_rmse = 0.0
                total_weight = 0.0
                
                # Iterate over each variable
                for var_name in variables.keys():
                    y_true, y_pred, depths = variables[var_name]
                    
                    # Filter data points within the current bin
                    mask = (depths >= left) & (depths < right)
                    if i == len(bins_edges)-2:  # The last interval includes the right endpoint
                        mask = (depths >= left) & (depths <= right)
                    
                    y_true_bin = y_true[mask]
                    y_pred_bin = y_pred[mask]
                    
                    # Skip empty bins
                    if len(y_true_bin) == 0:
                        continue
                    
                    # Calculate RMSE and weight for the current variable
                    rmse = np.sqrt(np.mean((y_true_bin - y_pred_bin)**2))
                    weight = weights[var_name]
                    
                    # Accumulate weighted RMSE
                    total_weighted_rmse += rmse * weight
                    total_weight += weight
                
                # Calculate sRMSE for the current bin
                if total_weight == sum(weights.values()):
                    srmse = total_weighted_rmse / total_weight
                    srmse_list.append(srmse)
                    depth_mids.append(mid)
            
            return np.array(depth_mids), np.array(srmse_list)

        def plot_depth_series(depth_mids, rmse_values, title, save_path):
            plt.figure(figsize=(12, 3.5), dpi=300)
            plt.plot(depth_mids, rmse_values, 
                    marker='o', linestyle='-', 
                    color='#2c7bb6', linewidth=2, 
                    markersize=6)
            plt.xlabel('Depth (m)')
            plt.ylabel('sRMSE')
            plt.title(title)
            plt.gca().invert_xaxis()  # Depth increases downward
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            gen_folder(os.path.dirname(save_path))
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()

        # Variable data collection
        variables = {
            'temp': (y_temp, temp_pre, depths_argo),
            'sal': (y_sal, sal_pre, depths_argo),
            'w': (y_w, w_pre, depths_wcur),
            'v_theta': (y_v_theta, v_theta_pre, depths_cur),
            'v_phi': (y_v_phi, v_phi_pre, depths_cur)
        }

        if plot_seperate:
            # Plot RMSE distribution for each variable
            for var_name, (y_true, y_pred, depths) in variables.items():
                depth_mids, rmse_values = calculate_rmse_depth(y_true, y_pred, depths, depth_bins_edges)
                valid = ~np.isnan(rmse_values)
                if np.sum(valid) == 0:
                    print(f"Warning: No valid data points for {var_name}")
                    continue
                plot_depth_series(depth_mids[valid], rmse_values[valid],
                                f'{var_name.capitalize()} RMSE Depth Distribution - {method_name}',
                                os.path.join(save_path, f'{var_name}_rmse_depth.png'))

        # Calculate and plot integrated sRMSE
        depth_mids, srmse_values = calculate_integrated_srmse(depth_bins_edges)
        valid = ~np.isnan(srmse_values)
        plot_depth_series(depth_mids[valid], srmse_values[valid],
                        f'sRMSE Depth Distribution of {method_name}',
                        os.path.join(save_path, 'integrated_srmse_depth.png'))

    def var_weights(self, y_temp, y_sal, y_w, y_v_theta, y_v_phi):
        # Combine velocity data
        velocity_data = np.concatenate([y_w, y_v_theta, y_v_phi])

        # Calculate the joint standard deviation
        velocity_std = np.std(velocity_data)

        # Calculate the unified weight
        velocity_weight = 1.0 / velocity_std

        # Define the weights dictionary
        weights = {
            'temp': 1.0 / np.std(y_temp),  # Weight for temperature
            'sal': 1.0 / np.std(y_sal),    # Weight for salinity
            'w': velocity_weight,          # Weight for vertical velocity
            'v_theta': velocity_weight,    # Weight for horizontal velocity (theta direction)
            'v_phi': velocity_weight       # Weight for horizontal velocity (phi direction)
        }

        return weights

    def plot_combined_rmse_depth(self, methods_list, depth_bins=20, depth_bin_power=2,
                             save_path='Output/Plot/RealData/RMSE'):
        """
        绘制多模型对比曲线（不使用插值）

        改进点：
        1. 统一分箱策略：基于所有模型的全局深度范围生成统一分箱
        2. 直接使用原始分箱中点绘制曲线
        3. 自动处理不同模型的数据覆盖范围差异
        """
        gen_folder(save_path)
        plt.figure(figsize=(12, 4), dpi=300)
        
        # 样式配置
        style_config = [
            {'color': METHOD_COLORS['MAD-REF Net'], 'marker': 'o', 'ls': '-'},  # MAD-REF Net
            {'color': METHOD_COLORS['Neural Field'], 'marker': 's', 'ls': '-'},  # Neural Field
            {'color': METHOD_COLORS['LightGBM'], 'marker': 'D', 'ls': '-'},   # LightGBM
            {'color': METHOD_COLORS['SparseGP'], 'marker': 'p', 'ls': '-'}   # SparseGP
        ]
        
        # 阶段1: 收集全局深度范围 ------------------------------------------------
        all_depths = []
        for method in methods_list:
            # 读取原始数据（不加载预测结果以节省内存）
            data_argo, data_cur, data_wcur = self.data_loader.read_data(type='test')
            all_depths.extend(data_argo[:,0].tolist())
            all_depths.extend(data_cur[:,0].tolist())
            all_depths.extend(data_wcur[:,0].tolist())
        
        # 计算全局分箱
        depth_min, depth_max = np.min(all_depths), np.max(all_depths)
        n_bins = depth_bins + 1
        x = np.linspace(0, 1, n_bins)
        s = 1 - (1 - x) ** depth_bin_power
        global_bins = depth_min + (depth_max - depth_min) * s
        bin_mids = [(global_bins[i] + global_bins[i+1])/2 for i in range(len(global_bins)-1)]
        
        # 阶段2: 计算各模型sRMSE ------------------------------------------------
        results = []
        for idx, method in enumerate(methods_list):
            # 读取数据和预测结果
            data_argo, data_cur, data_wcur = self.data_loader.read_data(type='test')
            temp_pre, sal_pre, w_pre, v_theta_pre, v_phi_pre = self.data_loader.read_pred_results(method['model_pre_path'])
            
            # 组织变量数据
            x_argo, y_temp, y_sal, x_cur, y_v_theta, y_v_phi, x_wcur, y_w = data_split(data_argo, data_cur, data_wcur)
            variables = {
                'temp': (y_temp, temp_pre, x_argo[:,0]),
                'sal': (y_sal, sal_pre, x_argo[:,0]),
                'w': (y_w, w_pre, x_wcur[:,0]),
                'v_theta': (y_v_theta, v_theta_pre, x_cur[:,0]),
                'v_phi': (y_v_phi, v_phi_pre, x_cur[:,0])
            }
            weights = self.var_weights(y_temp, y_sal, y_w, y_v_theta, y_v_phi)
            
            # 计算分箱sRMSE
            srmse_values = []
            for i in range(len(global_bins)-1):
                left, right = global_bins[i], global_bins[i+1]
                total_rmse, total_weight = 0.0, 0.0
                
                # 遍历所有变量
                for var in ['temp', 'sal', 'w', 'v_theta', 'v_phi']:
                    y_true, y_pred, depths = variables[var]
                    
                    # 获取当前分箱数据
                    mask = (depths >= left) & (depths < right)
                    if i == len(global_bins)-2:  # 最后一个分箱包含右端点
                        mask = (depths >= left) & (depths <= right)
                    
                    if np.sum(mask) == 0:
                        continue
                    
                    # 计算贡献值
                    rmse = np.sqrt(np.mean((y_true[mask]-y_pred[mask])**2))
                    total_rmse += rmse * weights[var]
                    total_weight += weights[var]

                srmse = total_rmse / total_weight if total_weight == sum(weights.values()) else np.nan
                srmse_values.append(srmse)
            
            # 保存结果
            results.append({
                'name': method['method_name'],
                'depth': np.array(bin_mids),
                'srmse': np.array(srmse_values),
                'style': style_config[idx]
            })
        
        # 阶段3: 可视化 ------------------------------------------------
        for res in results:
            valid = ~np.isnan(res['srmse'])
            plt.plot(res['depth'][valid], res['srmse'][valid],
                    marker=res['style']['marker'],
                    linestyle=res['style']['ls'],
                    color=res['style']['color'],
                    linewidth=2,
                    markersize=6,
                    markevery=1,
                    label=res['name'])
        
        plt.xlabel('Depth (m)', fontsize=12)
        plt.ylabel('sRMSE', fontsize=12)
        plt.yscale('log')
        plt.title('Variation of sRMSE with respect to depth', fontsize=14)
        plt.gca().invert_xaxis()
        plt.grid(True, alpha=0.3)
        plt.legend(loc='upper right', frameon=False)

        save_file = os.path.join(save_path, 'combined_srmse_depth_unified.png')
        plt.savefig(save_file, bbox_inches='tight', dpi=300)
        plt.close()
        print(f'>> Combined plot saved to: {save_file}')


    def plot_rmse_distribution(self, model_pre_path, method_name):
        self.plot_rmse_distribution_spacial(model_pre_path=model_pre_path,
                                       method_name=method_name)
        # self.plot_rmse_temporal(model_pre_path=model_pre_path,
        #                     method_name=method_name)
        # self.plot_rmse_depth(model_pre_path=model_pre_path, 
        #                 depth_bins=25,
        #                 method_name=method_name)



    def create_alternating_comparison_grid(
        self,
        folder1: str,
        folder2: str,
        model1_name: str,
        model2_name: str,
        output_path: str,
        cols: int = 4,
        font_path: Optional[str] = None,
        max_images: Optional[int] = None,
        scale_factor: float = 1.0,
        header_height: int = 30,
        font_size: int = 20,
        max_rows_per_image: Optional[int] = None,
        output_prefix: str = "comparison",
        sort_by: Literal['depth', 'latitude', 'longitude'] = 'depth',
        sort_direction: Literal['asc', 'desc'] = 'asc'
    ):
        """
        Create comparison grids supporting both depth and coordinate sorting.
        
        Args:
            folder1: Path to first model's images
            folder2: Path to second model's images  
            model1_name: Name for first model
            model2_name: Name for second model
            output_path: Output directory path
            cols: Number of columns (must be even)
            font_path: Optional custom font
            max_images: Max images to process
            scale_factor: Image scaling (0.1-1.0)
            header_height: Header row height
            font_size: Header font size
            max_rows_per_image: Max rows per output image
            output_prefix: Filename prefix for output images
            sort_by: Sorting method ('depth', 'latitude', 'longitude')
            sort_direction: 'asc' for ascending, 'desc' for descending
        """
        # Validate inputs
        if cols % 2 != 0:
            raise ValueError("Number of columns must be even")
        if not 0.1 <= scale_factor <= 1.0:
            raise ValueError("Scale factor should be between 0.1 and 1.0")
        
        os.makedirs(output_path, exist_ok=True)

        def extract_value(filename: str) -> float:
            """Extract numeric value from filename with flexible prefix
            
            Supports:
            - Depth format: PREFIX_<number>.jpg (e.g., "sal_-1.2.jpg")
            - Coordinate format: PREFIX_<lat>°<NS>[_<lon>°<EW>].jpg (e.g., "depth_10°N_50°E.jpg")
            """
            if sort_by == 'depth':
                # Match any prefix followed by number: PREFIX_<number>.jpg
                match = re.search(r'^.*?_([-+]?\d+\.?\d*)\.jpg$', filename, re.IGNORECASE)
                return float(match.group(1)) if match else 0.0
            
            # For coordinate sorting
            pattern = r'^.*?_([-+]?\d+\.?\d*)°([NS])(?:_([-+]?\d+\.?\d*)°([EW]))?\.jpg$'
            match = re.search(pattern, filename, re.IGNORECASE)
            if not match:
                return 0.0
                
            if sort_by == 'latitude':
                lat_val = float(match.group(1))
                lat_dir = match.group(2).upper()
                return lat_val if lat_dir == 'N' else -lat_val
            else:  # longitude
                if not match.group(3):  # No longitude in filename
                    return 0.0
                lon_val = float(match.group(3))
                lon_dir = match.group(4).upper()
                return lon_val if lon_dir == 'E' else -lon_val

        # Get and sort files
        files1 = sorted(
            [f for f in os.listdir(folder1) if f.lower().endswith('.jpg')],
            key=extract_value,
            reverse=(sort_direction == 'desc')
        )
        files2 = sorted(
            [f for f in os.listdir(folder2) if f.lower().endswith('.jpg')],
            key=extract_value,
            reverse=(sort_direction == 'desc')
        )

        if files1 != files2:
            raise ValueError("Image files in the two folders don't match")
        
        if max_images is not None:
            files1 = files1[:max_images]
            files2 = files2[:max_images]

        # Get image dimensions
        with Image.open(os.path.join(folder1, files1[0])) as sample_img:
            orig_width, orig_height = sample_img.size
            scaled_width = int(orig_width * scale_factor)
            scaled_height = int(orig_height * scale_factor)

        # Setup font
        try:
            font = ImageFont.truetype(font_path, font_size) if font_path else ImageFont.load_default(font_size)
        except:
            font = ImageFont.load_default()

        # Calculate grid parameters
        total_pairs = len(files1)
        pairs_per_row = cols // 2
        total_rows = math.ceil(total_pairs / pairs_per_row)
        
        if max_rows_per_image is None:
            max_rows_per_image = max(1, 8000 // (scaled_height + header_height))
        
        rows_per_image = min(max_rows_per_image, total_rows)
        num_output_images = math.ceil(total_rows / rows_per_image)

        print(f"Creating {num_output_images} output images sorted by {sort_by} ({sort_direction})")

        # Process in batches
        for image_num in range(num_output_images):
            start_row = image_num * rows_per_image
            end_row = min((image_num + 1) * rows_per_image, total_rows)
            
            rows_in_this_image = end_row - start_row
            grid_width = cols * scaled_width
            grid_height = rows_in_this_image * scaled_height + header_height
            
            canvas = Image.new('RGB', (grid_width, grid_height), color='white')
            draw = ImageDraw.Draw(canvas)
            
            # Draw headers
            for c in range(cols):
                x = c * scaled_width
                model_name = model1_name if c % 2 == 0 else model2_name
                draw.rectangle([(x, 0), (x + scaled_width - 1, header_height - 1)], fill='white')
                
                text_width = font.getlength(model_name)
                text_x = x + (scaled_width - text_width) / 2
                draw.text(
                    (text_x, header_height//2 - font_size//2),
                    model_name,
                    fill='black',
                    font=font
                )
            
            # Place images
            for row_in_image in range(rows_in_this_image):
                absolute_row = start_row + row_in_image
                y_pos = row_in_image * scaled_height + header_height
                
                for pair_in_row in range(pairs_per_row):
                    pair_index = absolute_row * pairs_per_row + pair_in_row
                    if pair_index >= total_pairs:
                        break
                    
                    col = pair_in_row * 2
                    x_pos = col * scaled_width
                    
                    filename = files1[pair_index]
                    with Image.open(os.path.join(folder1, filename)) as img1:
                        img1 = img1.resize((scaled_width, scaled_height), Image.LANCZOS)
                        canvas.paste(img1, (x_pos, y_pos))
                    
                    with Image.open(os.path.join(folder2, filename)) as img2:
                        img2 = img2.resize((scaled_width, scaled_height), Image.LANCZOS)
                        canvas.paste(img2, (x_pos + scaled_width, y_pos))
            
            # Save with metadata about sorting
            output_filename = f"{output_prefix}_{image_num+1:02d}.jpg"
            full_output_path = os.path.join(output_path, output_filename)
            canvas.save(full_output_path, quality=95)
            print(f"Saved {full_output_path}")

    def plot_merge_slice(self, mdrf_ensemble):
        font_sizes = [100, 50, 50]
        colss = [4, 2, 4]
        header_heights = [175, 75, 75]
        max_rows_per_images = [7, 6, 9]

        for i, slice_direction in enumerate(['Depth', 'Latitude', 'Longitude']):
            font_size = font_sizes[i]
            cols = colss[i]
            header_height = header_heights[i]
            max_rows_per_image = max_rows_per_images[i]

            for var in ['temp', 'sal']:
                self.create_alternating_comparison_grid(
                        folder1=f'Output/Plot/RealData/Global/Data/{slice_direction}/{var}',
                        folder2=f'Output/Plot/RealData/{mdrf_ensemble.ensemble_task_name}/Slice/{slice_direction}/{var}',
                        model1_name='Ground Truth',
                        model2_name='MAD-REF Net',
                        output_path='Output/Plot/RealData/Global/ComparisonGrid',
                        cols=cols,
                        scale_factor=0.5,  # Reduce image sizes to 50%
                        max_images=None,
                        font_size=font_size,
                        header_height=header_height,
                        max_rows_per_image=max_rows_per_image,
                        output_prefix=f"{slice_direction.lower()}_{var}",
                        sort_by = slice_direction.lower(),
                        sort_direction='desc'
                    )

    # def plot_3dfit_merge(self, mdrf_ensemble):
    #     image_files = ['1.jpg', '2.jpg', '3.jpg', '4.jpg', '5.jpg']
    #     grid_image = create_image_grid(image_files, cols=3)
    #     grid_image.save('output_grid.jpg')
    #     grid_image.show()