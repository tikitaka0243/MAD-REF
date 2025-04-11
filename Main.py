from pathlib import Path
import numpy as np
from Code.DataProcess import RealData
from Code.Model import RealDataModel
from Code.Plot import RealDataPlot
from Code.utils import TIME_LIST, data_old_split, ndarray_check


# ------------------ Load Data ------------------


# data_old_split(data_path='Data/RealData/Local/Argo/argo_all.npy', train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, random_seed=42)


data_loader = RealData(data_path='Data/RealData/Local')
data_loader.set_parameters(r_min=-2000, r_max=0, 
                           theta_min=-20, theta_max=20, 
                           phi_min=150, phi_max=-110,
                           t_min='2021-01-01T00:00:00Z', 
                           t_max='2023-01-01T00:00:00Z')

# data_loader.gen_new_x(density_r=41, 
#                       density_theta=161, 
#                       density_phi=401, 
#                       t_list=TIME_LIST, 
#                       save_path='Data/RealData/Local/new_x_plot.npy')
# data_loader.gen_new_x_slice(save_folder='Data/RealData/Local')

# data_loader.load_data_currents(currents_data_path='Data/RealData/RawFiles/Currents', 
#                                currents_save_path='Data/RealData/Local/Currents', 
#                                train_vali_test=[8, 1, 1], 
#                                ratio=1)

plotter = RealDataPlot(data_loader)
# plotter.plot_data(dataloader=data_loader, 
#                   datapath='Data/RealData/Local', 
#                   save_path='Output/Plot/RealData/Local/Data')
# plotter.plot_data_time_distribution(data_path='Data/RealData/Local', 
#                                     save_path='Output/Plot/RealData/Local//Data/TimeDistribution')
# plotter.plot_data_glory_tempsal_slice(plot_type='all', time='202107')

# ------------------ Train Model ------------------

# res140s1: restore from step 1 (1,400,000 iters)
# lw: layer_width

# 'Local/2Steps_lr1e-4_siren_largeBC/res140s1_lr1e-5_w_reinit'

mdrf = RealDataModel(task_name='Local/2Steps_lr1e-4_siren_largeBC/res140s1_lr1e-4_combine', 
                     dataloader=data_loader, 
                     plotter=plotter,
                     hard_constraint=True, 
                     meta_learning=False, 
                     output_dir=f'Output/Model/RealData', 
                     n_layers=[4] * 2 + [6] * 4, 
                     layer_width=[128] * 6)
mdrf.set_restore(restore_main=['Local/2Steps_lr1e-4_siren_largeBC/res140s1_lr1e-4_vtheta_reinit/res141_lr1e-5_vtheta', 1, 2378000],
                 restore_sub=[['Local/2Steps_lr1e-4_siren_largeBC/res140s1_lr1e-5_w_reinit', 1, 2374000, 'w']])

# mdrf.train(iters=[5000000, 2000], 
#            save_period=2000,
#            lrs=[1e-5, 1e-5], 
#            num_domain=1500, 
#            num_boundary=1000, 
#            batch_size=400000,
#            new_x_path='Data/RealData/Local/new_x_plot.npy',
#            loss_weight=None, 
#            restore_1=['Local/2Steps_lr1e-4_siren_largeBC/res140s1_lr1e-4_vtheta_reinit', 'Step1', 1410000],
#            train_var='v_theta',
#            reinitialize=False)
# ['Local/2Steps_lr1e-4_siren_largeBC', 'Step1', 1200000]

# mdrf.inference_and_plot(new_x_path='Data/RealData/Local/new_x_plot.npy', 
#                         save_path=mdrf.step_1_plot_pre_path, 
#                         t_list=TIME_LIST)
mdrf.inference_and_plot_slice()
# mdrf.test()

# mdrf.validate(step_num=1, iter_num=[0, 402000])


