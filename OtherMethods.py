from datetime import datetime
import os

from sklearn.metrics import root_mean_squared_error
from tqdm import tqdm
from Code.DataProcess import RealData
from Code.OtherMethods import OtherMethods, predict_with_saved_model_gp
from Code.Plot import RealDataPlot
from Code.utils import VAR_COLORS, VAR_NAMES, VAR_UNITS, ndarray_check, unpack_x



data_loader = RealData(data_path='Data/RealData/Global')
data_loader.set_parameters(r_min=-2000, r_max=0, 
                           theta_min=-90, theta_max=90, 
                           phi_min=-180, phi_max=180,
                           t_min='2021-01-01T00:00:00Z', 
                           t_max='2023-01-01T00:00:00Z')


othermethods = OtherMethods(data_loader)
othermethods.test(method='lightgbm', save_path='Output/Prediction/RealData/OtherMethods/LightGBM', tasks_to_run=['v_phi', 'w'])
othermethods.test(method='sparse_gp', save_path='Output/Prediction/RealData/OtherMethods/SparseGP', tasks_to_run=['v_phi'])

