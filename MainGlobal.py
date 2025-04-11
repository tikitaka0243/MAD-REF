import os
from sklearn.metrics import root_mean_squared_error
import torch
from Code.DataProcess import RealData
from Code.Model import EnsembleModel, RealDataModel
from Code.Plot import RealDataPlot
from Code.utils import TIME_LIST, TIME_LIST_SHINY, data_old2new_global_argo

# -------------------------- Load Data --------------------------



data_loader = RealData(data_path='Data/RealData/Global')
data_loader.set_parameters(r_min=-2000, r_max=0, 
                           theta_min=-90, theta_max=90, 
                           phi_min=-180, phi_max=180,
                           t_min='2021-01-01T00:00:00Z', 
                           t_max='2023-01-01T00:00:00Z')
data_loader.gen_new_x(density_r=41, 
                      density_theta=361, 
                      density_phi=721, 
                      t_list=TIME_LIST, 
                      save_path='Data/RealData/Global/new_x_plot.npy')
data_loader.gen_new_x(density_theta=901, 
                    density_phi=1801, 
                    t_list=["2021-10-16T12:00:00Z"],
                    r_range=data_loader.read_depths()[:40],
                    save_path='Data/RealData/Global/new_x_plot_polar.npy')
data_loader.gen_new_x(density_theta=1441, 
                      density_phi=2881, 
                      t_list=["2023-01-16T12:00:00Z"], 
                      save_path='Data/RealData/Global/new_x_plot_depth_pre.npy',
                      r_range=data_loader.read_depths()[:40])
data_loader.gen_new_x(density_r=41, 
                      density_theta=361, 
                      density_phi=721, 
                      t_list=TIME_LIST_SHINY, 
                      save_path='Data/RealData/Global/new_x_plot_shiny.npy')
data_loader.gen_new_x_slice_global()

data_loader.load_data_currents(currents_data_path='Data/RealData/RawFiles/Currents', 
                               currents_save_path='Data/RealData/Global/Currents', 
                               train_vali_test=[8, 1, 1], 
                               ratio=0.1,
                               convert_and_filter=True)
data_loader.load_data_seaice(data_path='Data/RealData/RawFiles/SeaIce', save_path='Data/RealData/Global/SeaIce')
data_loader.load_data_tempsal(data_path='Data/RealData/RawFiles/GLOBAL_MULTIYEAR_PHY_001_030', save_path='Data/RealData/Global/TempSal')
data_loader.load_vphi_plot()


plotter = RealDataPlot(data_loader)
plotter.plot_data(datapath='Data/RealData/Global', 
                  save_path='Output/Plot/RealData/Global/Data',
                  area='Global',
                  rotate=False)
plotter.plot_data(datapath='Data/RealData/Global', 
                  save_path='Output/Plot/RealData/Global/DataRotate',
                  area='Global',
                  rotate=True)

plotter.plot_data_time_distribution(data_path='Data/RealData/Global', 
                                    save_path='Output/Plot/RealData/Global/Data/TimeDistribution')

plotter.plot_data_seaice()
plotter.plot_data_glory_tempsal_polar()
plotter.plot_data_glory_tempsal_slice(save_path='Output/Plot/RealData/Global/Data202301', plot_type='depth', time='202301')
plotter.plot_vtheta_slice_depth()

# ------------------------------ Train Model ------------------------------


mdrf = RealDataModel(task_name='model', 
                     dataloader=data_loader, 
                     plotter=plotter,
                     hard_constraint=True, 
                     meta_learning=False, 
                     output_dir=f'Output/Model/RealData',
                     layer_width=[256] * 6, 
                     sub_learner=0,
                     activation_func='sin')

mdrf.train(iters=[5000000, 5000000], 
           save_period=5000,
           lrs=[1e-4, 1e-4], 
           num_domain=1500, 
           num_boundary=1000, 
           batch_size=40000,
           new_x_path='Data/RealData/Global/new_x_plot.npy',
           loss_weight=None,
           train_var='all',
           reinitialize=False,
           external_source=False, 
           skip_step_1=False)

mdrf.inference_and_plot(new_x_path='Data/RealData/Global/new_x_plot.npy', 
                        save_path=mdrf.step_1_plot_pre_path,
                        t_list=TIME_LIST[1:2])
mdrf.test(save_pre_bool=True)

mdrf.validate(step_num=1, iter_num=[0, 545000])

# ----------------- Ensemble Model -----------------

mdrf_1 = RealDataModel(task_name='model_1', 
                     dataloader=data_loader, 
                     plotter=plotter,
                     hard_constraint=True, 
                     meta_learning=False, 
                     output_dir=f'Output/Model/RealData',
                     layer_width=[256] * 6, 
                     sub_learner=1,
                     activation_func='sin')

mdrf_2 = RealDataModel(task_name='model_1', 
                     dataloader=data_loader, 
                     plotter=plotter,
                     hard_constraint=True, 
                     meta_learning=False, 
                     output_dir=f'Output/Model/RealData',
                     layer_width=[256] * 6, 
                     sub_learner=2,
                     activation_func='sin')

mdrf_ensemble = EnsembleModel(sublearner_1=mdrf_1, sublearner_2=mdrf_2, plotter=plotter, dataloader=data_loader)

mdrf_ensemble.inference(new_x_path='Data/RealData/Global/new_x_plot.npy', 
                        plot_bool=True, 
                        t_list=TIME_LIST,
                        plot_range=range(7))


