from datetime import datetime
import os

from sklearn.metrics import root_mean_squared_error
from tqdm import tqdm
from Code.DataProcess import RealData
from Code.OtherMethods import OtherMethods, predict_with_saved_model_gp
from Code.Plot import RealDataPlot
from Code.utils import VAR_COLORS, VAR_NAMES, VAR_UNITS, ndarray_check, unpack_x


os.environ["CUDA_VISIBLE_DEVICES"] = "4"


data_loader = RealData(data_path='Data/RealData/Global')
data_loader.set_parameters(r_min=-2000, r_max=0, 
                           theta_min=-90, theta_max=90, 
                           phi_min=-180, phi_max=180,
                           t_min='2021-01-01T00:00:00Z', 
                           t_max='2023-01-01T00:00:00Z')


othermethods = OtherMethods(data_loader)
# othermethods.test(method='lightgbm', save_path='Output/Prediction/RealData/OtherMethods/LightGBM', tasks_to_run=['v_phi', 'w'])
# othermethods.test(method='sparse_gp', save_path='Output/Prediction/RealData/OtherMethods/SparseGP', tasks_to_run=['v_phi'])


# othermethods.load_and_inference()



plotter = RealDataPlot(data_loader)

methods = [
    {
        'model_pre_path': 'Output/Prediction/RealData/Global/Ensemble/2Steps_lr1e-4_sin_largeBC-res95s1_lr1e-4_combine|_2Steps_lr1e-4_sin_largeBC-res92s1_lr1e-4_combine',
        'method_name': 'MAD-REF net'
    },
    {
        'model_pre_path': 'Output/Prediction/RealData/Global/no_ensemble_lr1e-4_sin_largeBC',
        'method_name': 'Neural Field'
    },
    {
        'model_pre_path': 'Output/Prediction/RealData/OtherMethods/LightGBM',
        'method_name': 'LightGBM'
    },
    {
        'model_pre_path': 'Output/Prediction/RealData/OtherMethods/SparseGP',
        'method_name': 'SparseGP'
    }
]

for method in tqdm(methods, desc='Processing Methods'):
    plotter.plot_rmse_distribution(model_pre_path=method['model_pre_path'], 
                                   method_name=method['method_name'])


plotter.plot_combined_rmse_temporal(method_info_list = methods, 
                                    time_step='ME',
                                    save_path='Output/Plot/RealData/RMSE/',
                                    fig_name='combined_temperal_rmse_comparison.png')
plotter.plot_combined_rmse_depth(methods_list=methods, 
                                 depth_bins=25)
