import os
from sklearn.metrics import root_mean_squared_error
import torch
from Code.DataProcess import RealData
from Code.Model import EnsembleModel, RealDataModel
from Code.Plot import RealDataPlot
from Code.utils import TIME_LIST, TIME_LIST_SHINY, data_old2new_global_argo

# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
torch.cuda.set_device(4)

# -------------------------- Load Data --------------------------


# for data_type in ['train', 'vali', 'test']:
#     old_path = f'Data/RealData/Global/Argo/Old/argo_{data_type}.npy'
#     new_path = f'Data/RealData/Global/Argo/argo_{data_type}.npy'
#     data_old2new_global_argo(old_path, new_path)


data_loader = RealData(data_path='Data/RealData/Global')
data_loader.set_parameters(r_min=-2000, r_max=0, 
                           theta_min=-90, theta_max=90, 
                           phi_min=-180, phi_max=180,
                           t_min='2021-01-01T00:00:00Z', 
                           t_max='2023-01-01T00:00:00Z')
# data_loader.gen_new_x(density_r=41, 
#                       density_theta=361, 
#                       density_phi=721, 
#                       t_list=TIME_LIST, 
#                       save_path='Data/RealData/Global/new_x_plot.npy')
# data_loader.gen_new_x(density_theta=901, 
#                     density_phi=1801, 
#                     t_list=["2021-10-16T12:00:00Z"],
#                     r_range=data_loader.read_depths()[:40],
#                     save_path='Data/RealData/Global/new_x_plot_polar.npy')
# data_loader.gen_new_x(density_theta=1441, 
#                       density_phi=2881, 
#                       t_list=["2023-01-16T12:00:00Z"], 
#                       save_path='Data/RealData/Global/new_x_plot_depth_pre.npy',
#                       r_range=data_loader.read_depths()[:40])
# data_loader.gen_new_x(density_r=41, 
#                       density_theta=361, 
#                       density_phi=721, 
#                       t_list=TIME_LIST_SHINY, 
#                       save_path='Data/RealData/Global/new_x_plot_shiny.npy')
# data_loader.gen_new_x_slice_global()

# data_loader.load_data_currents(currents_data_path='Data/RealData/RawFiles/Currents', 
#                                currents_save_path='Data/RealData/Global/Currents', 
#                                train_vali_test=[8, 1, 1], 
#                                ratio=0.1,
#                                convert_and_filter=True)
# data_loader.load_data_seaice(data_path='Data/RealData/RawFiles/SeaIce', save_path='Data/RealData/Global/SeaIce')
# data_loader.load_data_tempsal(data_path='Data/RealData/RawFiles/GLOBAL_MULTIYEAR_PHY_001_030', save_path='Data/RealData/Global/TempSal')
# data_loader.load_vphi_plot()


plotter = RealDataPlot(data_loader)
# plotter.plot_data(datapath='Data/RealData/Global', 
#                   save_path='Output/Plot/RealData/Global/Data',
#                   area='Global',
#                   rotate=False)
# plotter.plot_data(datapath='Data/RealData/Global', 
#                   save_path='Output/Plot/RealData/Global/DataRotate',
#                   area='Global',
#                   rotate=True)

# plotter.plot_data_time_distribution(data_path='Data/RealData/Global', 
#                                     save_path='Output/Plot/RealData/Global/Data/TimeDistribution')

# plotter.plot_data_seaice()
# plotter.plot_data_glory_tempsal_polar()
# plotter.plot_data_glory_tempsal_slice(save_path='Output/Plot/RealData/Global/Data202301', plot_type='depth', time='202301')
# plotter.plot_vtheta_slice_depth()

# ------------------------------ Train Model ------------------------------
# 'Global/_2Steps_lr1e-4_sin_largeBC/res95s1_lr1e-4_w_reinit_hc'
# hc: Hard Constraint

mdrf = RealDataModel(task_name='Global/no_ensemble_lr1e-4_sin_largeBC', 
                     dataloader=data_loader, 
                     plotter=plotter,
                     hard_constraint=True, 
                     meta_learning=False, 
                     output_dir=f'Output/Model/RealData',
                     layer_width=[256] * 6, 
                     sub_learner=0,
                     activation_func='sin')

# mdrf.set_restore(restore_main=['Global/no_ensemble_lr1e-4_sin_largeBC', 1, 400000],
#                     restore_sub=None)

# mdrf.train(iters=[5000000, 5000000], 
#            save_period=5000,
#            lrs=[1e-4, 1e-4], 
#            num_domain=1500, 
#            num_boundary=1000, 
#            batch_size=40000,
#            new_x_path='Data/RealData/Global/new_x_plot.npy',
#            loss_weight=None,
#            train_var='all',
#            reinitialize=False,
#            external_source=False, 
#            skip_step_1=False)
# ['Global/2Steps_lr1e-4_sin_largeBC', 'Step1', 950000]

# mdrf.inference_and_plot(new_x_path='Data/RealData/Global/new_x_plot.npy', 
#                         save_path=mdrf.step_1_plot_pre_path,
#                         t_list=TIME_LIST[1:2])
# mdrf.test(save_pre_bool=True)

# mdrf.validate(step_num=1, iter_num=[0, 545000])

# ----------------- Ensemble Model -----------------

mdrf_1 = RealDataModel(task_name='Global/2Steps_lr1e-4_sin_largeBC/res95s1_lr1e-4_combine', 
                     dataloader=data_loader, 
                     plotter=plotter,
                     hard_constraint=True, 
                     meta_learning=False, 
                     output_dir=f'Output/Model/RealData',
                     layer_width=[256] * 6, 
                     sub_learner=1,
                     activation_func='sin')
mdrf_1.set_restore(restore_main=['Global/2Steps_lr1e-4_sin_largeBC/res95s1_lr1e-4_w_reinit_hc', 1, 1390000],
                 restore_sub=[['Global/2Steps_lr1e-4_sin_largeBC/res95s1_lr1e-4_vtheta_reinit', 1, 2190000, 'v_theta']])
# mdrf_1.inference_and_plot(new_x_path='Data/RealData/Global/new_x_plot.npy', 
#                         save_path=mdrf_1.step_1_plot_pre_path,
#                         t_list=TIME_LIST[1:2])

mdrf_2 = RealDataModel(task_name='Global/_2Steps_lr1e-4_sin_largeBC/res92s1_lr1e-4_combine', 
                     dataloader=data_loader, 
                     plotter=plotter,
                     hard_constraint=True, 
                     meta_learning=False, 
                     output_dir=f'Output/Model/RealData',
                     layer_width=[256] * 6, 
                     sub_learner=2,
                     activation_func='sin')
mdrf_2.set_restore(restore_main=['Global/_2Steps_lr1e-4_sin_largeBC/res92s1_lr1e-4_w_reinit_hc', 1, 195000],
                    restore_sub=[['Global/_2Steps_lr1e-4_sin_largeBC/res92s1_lr1e-4_vtheta_reinit', 1, 1985000, 'v_theta'], ['Global/_2Steps_lr1e-4_sin_largeBC/res92s1_lr1e-4_temp_seaice', 2, 750000, 'temp']])
# mdrf_2.inference_and_plot(new_x_path='Data/RealData/Global/new_x_plot.npy', 
#                         save_path=mdrf_2.step_1_plot_pre_path,
#                         t_list=TIME_LIST[1:2])
# mdrf_2.test()

mdrf_ensemble = EnsembleModel(sublearner_1=mdrf_1, sublearner_2=mdrf_2, plotter=plotter, dataloader=data_loader)

mdrf_ensemble.inference(new_x_path='Data/RealData/Global/new_x_plot.npy', 
                        plot_bool=True, 
                        t_list=TIME_LIST,
                        plot_range=range(7))
# mdrf_ensemble.inference(new_x_path='Data/RealData/Global/new_x_plot_shiny.npy', 
#                         save_name='prediction_plot_shiny.npy',
#                         plot_bool=True, 
#                         t_list=TIME_LIST_SHINY,
#                         plot_range=range(7), 
#                         fig_save_path='Shiny')
# mdrf_ensemble.inference_and_plot_polar()
# mdrf_ensemble.inference_and_plot_slice()
# mdrf_ensemble.inference_and_plot_slice_pre(time='202301')
# mdrf_ensemble.inference_and_plot_slice_depth_vtheta()
# mdrf_ensemble.test(save_pre_bool=True)

# ----------------------- Plot -----------------------

# plotter.read_csv_and_plot_rmse(task_name_1='Global/2Steps_lr1e-4_sin_largeBC', 
#                                task_name_2='Global/2Steps_lr1e-4_sin_largeBC/res95s1_lr1e-4_vtheta_reinit', 
#                                label_1='Without Branchwise Training',
#                                label_2='With Branchwise Training',
#                                var='v_theta', 
#                                output_file='Output/Plot/RealData/Global/MethodComparison/RMSE_v_theta.png')

plotter.plot_merge_slice(mdrf_ensemble)

