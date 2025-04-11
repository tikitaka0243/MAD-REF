import datetime
import os
from pathlib import Path
# import copernicusmarine
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import netCDF4 as nc
import xarray as xr

from Code.utils import TIME_LIST, VAR_NAMES, convert_to_timestamp, find_closest_vector, gen_folder, get_all_files, ndarray_check, normalize_t, proportional_sampling, simulation_data_domain_check, unpack_x, unpack_y


class SimulationData():
    def __init__(self, eta, zeta, zeta_tau):
        self.eta = eta
        self.zeta = zeta
        self.zeta_tau = zeta_tau

    def taylor_green_vortex(self, x, z, t, save=False, save_path='Data/simulation_solution.npy'):
        tau = np.sin(2 * np.pi * z) * np.exp(-4 * np.pi**2 * self.zeta_tau * t)
        w = np.cos(2 * np.pi * x) * np.sin(2 * np.pi * z) * np.exp(-4 * np.pi**2 * (self.eta + self.zeta) * t)
        v = -np.sin(2 * np.pi * x) * np.cos(2 * np.pi * z) * np.exp(-4 * np.pi**2 * (self.eta + self.zeta) * t)
        p = 0.25 * np.cos(4 * np.pi * x) * np.exp(-8 * np.pi**2 * (self.eta + self.zeta) * t) + np.cos(2 * np.pi * z) * np.exp(-4 * np.pi**2 * self.zeta_tau * t) / (2 * np.pi)
        
        Q = np.pi * np.cos(2 * np.pi * x) * np.sin(4 * np.pi * z) * np.exp(-4 * np.pi**2 * (self.eta + self.zeta + self.zeta_tau) * t)

        if save:
            solution = np.column_stack((tau, w, v, p, Q))
            np.save(save_path, solution)

        return tau, w, v, p

    def gen_simulate_data(self, num=10000, noise_std=0.1, simulation_data_path = 'Data/Simulation/Train'):
        print('Generating simulation data...')
        
        x = np.random.rand(int(num * 3))
        z = np.random.rand(int(num * 3))
        select = simulation_data_domain_check(x, z)
        x = x[select]
        z = z[select]
        x = x[:num]
        z = z[:num]
        
        t_values = np.array([0.1, 0.25, 0.4, 0.55, 0.7, 0.85, 1])
        t = np.random.choice(t_values, size=num)
        
        x_z_t = np.column_stack((x, z, t))
        np.save(f'{simulation_data_path}/x_z_t.npy', x_z_t)

        tau, w, v, p = self.taylor_green_vortex(x, z, t)

        tau = tau + np.random.normal(loc=0, scale=noise_std, size=num)
        w = w + np.random.normal(loc=0, scale=noise_std, size=num)
        v = v + np.random.normal(loc=0, scale=noise_std, size=num)
        p = p + np.random.normal(loc=0, scale=noise_std, size=num) 
        
        np.save(f'{simulation_data_path}/tau.npy', tau)
        np.save(f'{simulation_data_path}/w.npy', w)
        np.save(f'{simulation_data_path}/v.npy', v)
        np.save(f'{simulation_data_path}/p.npy', p)
        
        print('Done.')

    def simulation_gen_new_x(self, density=101, density_t=101, save_path='Data/simulation_new_x.npy'):
        x_range = np.linspace(0, 1, density, endpoint=True)
        z_range = np.linspace(0, 1, density, endpoint=True)
        t_range = np.linspace(0, 1, density_t, endpoint=True)

        x, z, t = np.meshgrid(x_range, z_range, t_range)

        new_x = np.column_stack((x.ravel(), z.ravel(), t.ravel()))

        np.save(save_path, new_x)

    @staticmethod
    def simulation_read_data(simulation_data_path, folder='Train'):

        tau = np.load(os.path.join(simulation_data_path, folder, 'tau.npy'))
        w = np.load(os.path.join(simulation_data_path, folder, 'w.npy'))
        v = np.load(os.path.join(simulation_data_path, folder, 'v.npy'))
        p = np.load(os.path.join(simulation_data_path, folder, 'p.npy'))
        
        x_z_t = np.load(os.path.join(simulation_data_path, folder, 'x_z_t.npy'))
        
        return tau, w, v, p, x_z_t
    
class RealData():
    def __init__(self, data_path=None):
        self.data_path = data_path
        self.area = data_path.split('/')[-1]

        self.r_min = -2000
        self.r_max = 0
        self.theta_min = -10
        self.theta_max = 10
        self.phi_min = 150
        self.phi_max = -110

        self.T_0 = None
        self.S_0 = None
        self.theta_range = None
        self.phi_range = None

        self.Z = 5000
        self.a = 6.4 * 10 ** 6
        self.U = 10 ** -1
        self.g = 9.8
        self.Omega = 10 ** -4
        self.rho_0 = 10 ** 3
        self.mu = 10 ** 4
        self.mu_TS = 2.5 * 10 ** 3
        self.nu = 1.5 * 10 ** -4

        self.update_parameters()
        
    
    def read_data(self, type='train', datasets=['argo', 'cur', 'wcur'], filter_bool=False):
        data_argo, data_cur, data_wcur = None, None, None

        if 'argo' in datasets:
            data_argo = np.load(os.path.join(self.data_path, 'Argo', f'argo_{type}.npy'))
            # data_argo = proportional_sampling(data_argo, 0.1)
            if filter_bool:
                data_argo = self.data_filter(data_argo, 
                                            self.r_min, self.r_max, 
                                            self.theta_min, self.theta_max, 
                                            self.phi_min, self.phi_max, 
                                            self.t_min, self.t_max)
            data_argo = data_argo.astype(np.float32)

        if 'cur' in datasets:
            data_cur = np.load(os.path.join(self.data_path, 'Currents', f'cur_{type}.npy'))
            if filter_bool:
                data_cur = self.data_filter(data_cur, 
                                            self.r_min, self.r_max, 
                                            self.theta_min, self.theta_max, 
                                            self.phi_min, self.phi_max, 
                                            self.t_min, self.t_max)
            data_cur = data_cur.astype(np.float32)

        if 'wcur' in datasets:
            data_wcur = np.load(os.path.join(self.data_path, 'Currents', f'wcur_{type}.npy'))
            # sampling
            if filter_bool:
                data_wcur = self.data_filter(data_wcur, 
                                            self.r_min, self.r_max, 
                                            self.theta_min, self.theta_max, 
                                            self.phi_min, self.phi_max, 
                                            self.t_min, self.t_max)
            data_wcur = data_wcur.astype(np.float32)
        # (18134958, 6) (439749648, 6) (439749648, 5)


        # data_argo = np.memmap(os.path.join(self.data_path, 'Argo', f'argo_{type}.npy'), dtype=np.float64, mode='r', shape=(18134958, 6))

        # data_cur = np.memmap(os.path.join(self.data_path, 'Currents', f'cur_{type}.npy'), dtype=np.float64, mode='r')
        # data_wcur = np.memmap(os.path.join(self.data_path, 'Currents', f'wcur_{type}.npy'), dtype=np.float64, mode='r')

        # data_argo = data_argo.reshape((18134958, 6))
        # data_cur = data_cur.reshape((439749648, 6))
        # data_wcur = data_wcur.reshape((439749648, 5))

        # exit()

        return data_argo, data_cur, data_wcur
    
    def read_data_seaice(self, type='train'):
        data_seaice = np.load(os.path.join(self.data_path, 'SeaIce', f'seaice_{type}.npy'))
        data_seaice = data_seaice.astype(np.float32)

        data_seaice = np.insert(data_seaice, 0, values=0, axis=1)
        data_seaice = np.insert(data_seaice, 4, values=-1.7, axis=1)

        return data_seaice
    

    def read_glory_tempsal(self, folder_path='Data/RealData/Global/TempSal', time='202101'):
        file_list = get_all_files(folder_path)

        for file in file_list:
            if time in file:
                data = np.load(file)
                
                return data

    def read_depths(self, file_path='Data/RealData/Depths.csv'):
        df = pd.read_csv(file_path, header=None)

        depths = df.to_numpy().flatten()
        
        return depths


    def set_parameters(self, r_min, r_max, theta_min, theta_max, phi_min, phi_max, t_min=None, t_max=None, data_path=None):
        if data_path is not None:
            self.data_path = data_path

        self.r_min = r_min
        self.r_max = r_max
        self.theta_min = theta_min
        self.theta_max = theta_max
        self.phi_min = phi_min
        self.phi_max = phi_max

        if t_min is not None and t_max is not None:
            self.t_min = convert_to_timestamp(t_min)
            self.t_max = convert_to_timestamp(t_max)

        self.theta_range = theta_max - theta_min
        self.phi_range = self.cal_phi_range()

    def get_parameters(self):
        return self.r_min, self.r_max, self.theta_min, self.theta_max, self.phi_min, self.phi_max, self.t_min, self.t_max

    def update_parameters(self):
        self.epsilon = self.Z / self.a
        self.R_o = self.U / (self.a * self.Omega)

        self.Re1_inv = self.mu / (self.a * self.U)
        self.Re2_inv = self.a * self.nu / (self.Z ** 2 * self.U)
        self.Rt1_inv = self.mu_TS / (self.a * self.U)
        self.Rt2_inv = self.a * self.nu / (self.Z ** 2 * self.U)
        self.Rs1_inv = self.mu_TS / (self.a * self.U)
        self.Rs2_inv = self.a * self.nu / (self.Z ** 2 * self.U)

        self.b_bar = self.g * self.Z / self.U ** 2

    def cal_phi_range(self):
        if self.phi_min <= self.phi_max:
            return self.phi_max - self.phi_min
        else:
            return self.phi_max - self.phi_min + 360
        
    def cal_a(self):
        return max(self.theta_range, self.phi_range) * 111.32
    
    # def date_convertor(self, t):
    #     return pd.to_datetime(t).value / 10 ** 9
    
    def set_data_type(self, data_type):
        self.data_type = data_type
    
    def normalization_coordinates(self, data):

        r, theta, phi, t = unpack_x(data)

        r_, theta_, phi_, t_ = self.normalization_coordinates_(r, theta, phi, t, nor_phi=True)

        ob_coordinate_t = np.column_stack((r_, theta_, phi_, t_))

        return ob_coordinate_t

    def normalization_coordinates_(self, r, theta, phi, t, nor_phi=True):
        
        r_ = r / self.Z
        theta_ = np.deg2rad(-theta + 90)

        if nor_phi:
            if self.phi_min > self.phi_max:
                loc = phi >= self.phi_min
                phi[loc] -= self.phi_min
                phi[~loc] += 180 + (180 - self.phi_min)   
            else:
                phi -= self.phi_min
        phi_ = np.deg2rad(phi)

        t_ = normalize_t(t, t_min=self.t_min)

        return r_, theta_, phi_, t_

    def normalization_seaice(self, data_seaice):
        ob_coordinate_t_seaice = self.normalization_coordinates(data_seaice)

        self.load_normalization_stats(2)
        seaice = data_seaice[:, 4:5]
        ob_seaice = (seaice - self.T_mean) / self.T_0

        return ob_coordinate_t_seaice, ob_seaice


    def normalization_argo(self, data_argo, sublearner=None):

        ob_coordinate_t_argo = self.normalization_coordinates(data_argo)

        temp = data_argo[:, 4:5]
        sal = data_argo[:, 5:6]

        if self.data_type == 'train':
            self.T_0 = np.std(temp)
            self.S_0 = np.std(sal)
            self.T_mean = np.mean(temp)
            self.S_mean = np.mean(sal)

            stats_df = pd.DataFrame({
                'T_0': [self.T_0],
                'S_0': [self.S_0],
                'T_mean': [self.T_mean],
                'S_mean': [self.S_mean]
            })

            if sublearner is not None:
                stats_df.to_csv(os.path.join(self.data_path, f'normalization_stats_sublearner_{sublearner}.csv'), index=False)
            else:
                stats_df.to_csv(os.path.join(self.data_path, f'normalization_stats.csv'), index=False)

        ob_temp = (temp - self.T_mean) / self.T_0
        ob_sal = (sal - self.S_mean) / self.S_0

        return ob_coordinate_t_argo, ob_temp, ob_sal
    
    def load_normalization_stats(self, sublearner=None):
        if sublearner is not None:
            stats_df = pd.read_csv(os.path.join(self.data_path, f'normalization_stats_sublearner_{sublearner}.csv'))
        else:
            stats_df = pd.read_csv(os.path.join(self.data_path, 'normalization_stats.csv'))

        self.T_0 = stats_df['T_0'].values[0]
        self.S_0 = stats_df['S_0'].values[0]
        self.T_mean = stats_df['T_mean'].values[0]
        self.S_mean = stats_df['S_mean'].values[0]


    def normalization_currents(self, data_wcur, data_cur):
        ob_coordinate_t_wcur = self.normalization_coordinates(data_wcur)
        ob_coordinate_t_cur = self.normalization_coordinates(data_cur)

        w = data_wcur[:, 4:5]
        v1 = data_cur[:, 4:5]
        v2 = data_cur[:, 5:6]

        ob_w = w / (self.epsilon * self.U)
        ob_v1 = v1 / self.U
        ob_v2 = v2 / self.U

        return ob_coordinate_t_wcur, ob_coordinate_t_cur, ob_w, ob_v1, ob_v2

    def anti_normalization(self, prediction, sublearner=None):
        # if self.T_0 is None or self.S_0 is None or \
        #    self.T_mean is None or self.S_mean is None:
        #     self.load_normalization_stats(sublearner)
        self.load_normalization_stats(sublearner)

        tau, sigma, w, v_theta, v_phi, p = unpack_y(prediction)

        tau = tau * self.T_0 + self.T_mean
        sigma = sigma * self.S_0 + self.S_mean
        w = w * (self.epsilon * self.U)
        v_theta = v_theta * self.U
        v_phi = v_phi * self.U

        return np.column_stack([tau, sigma, w, v_theta, v_phi, p])



    def gen_new_x(self, 
                  density_r=101, 
                  density_theta=1001, 
                  density_phi=1001, 
                  t_list=TIME_LIST, 
                  save_path='Data/RealData/new_x_plot.npy',
                  r_range=None,
                  return_bool=False):
        
        if r_range is None:
            r_range = np.linspace(self.r_min, self.r_max, density_r, endpoint=True)

        theta_range = np.linspace(self.theta_min, self.theta_max, density_theta, endpoint=True)

        if self.phi_max >= self.phi_min:
            phi_range = np.linspace(self.phi_min, self.phi_max, density_phi, endpoint=True)
        else:
            density_phi_1 = int(density_phi * (180 - self.phi_min) / self.phi_range)
            density_phi_2 = density_phi - density_phi_1
            phi_range_1 = np.linspace(self.phi_min, 180, density_phi_1, endpoint=True)
            phi_range_2 = np.linspace(-180, self.phi_max, density_phi_2, endpoint=True)
            phi_range = np.concatenate((phi_range_1, phi_range_2))

        r, theta, phi = np.meshgrid(r_range, theta_range, phi_range)
        data_coor = np.column_stack((r.ravel(), theta.ravel(), phi.ravel()))

        is_ocean = self.is_ocean_3d_vector(data_coor)
        data_coor = data_coor[is_ocean]


        t_list = convert_to_timestamp(t_list)
        t_range = np.array(t_list) 

        print(f'Generating new_x with {len(data_coor)} points and {len(t_range)} timestamps')

        new_x = np.zeros((len(data_coor)*len(t_range), 4))
        new_x[:, :3] = np.repeat(data_coor, len(t_range), axis=0)
        new_x[:, 3] = np.tile(t_range, len(data_coor))

        # new_x = np.column_stack(self.normalization_coordinates_(*unpack_x(new_x)))

        if return_bool:
            return new_x
        np.save(save_path, new_x)


    def gen_new_x_slice(self, save_folder='Data/RealData/Local'):

        gen_folder(save_folder)

        self.gen_new_x(density_theta=401, 
                      density_phi=1001, 
                      t_list=TIME_LIST[1:2], 
                      save_path=os.path.join(save_folder, 'new_x_plot_depth.npy'),
                      r_range=self.read_depths()[:40])
        self.gen_new_x(density_r=201, 
                        density_theta=5, 
                        density_phi=1001, 
                        t_list=TIME_LIST[1:2], 
                        save_path=os.path.join(save_folder, 'new_x_plot_latitude.npy'))
        self.gen_new_x(density_r=201, 
                        density_theta=401, 
                        density_phi=11, 
                        t_list=TIME_LIST[1:2], 
                        save_path=os.path.join(save_folder, 'new_x_plot_longitude.npy'))

    def gen_new_x_slice_global(self, save_folder='Data/RealData/Global'):

        gen_folder(save_folder)

        self.gen_new_x(density_theta=1441, 
                      density_phi=2881, 
                      t_list=TIME_LIST[1:2], 
                      save_path=os.path.join(save_folder, 'new_x_plot_depth.npy'),
                      r_range=self.read_depths()[:40])
        
        self.gen_new_x(density_r=201, 
                        density_theta=19, 
                        density_phi=2881, 
                        t_list=TIME_LIST[1:2], 
                        save_path=os.path.join(save_folder, 'new_x_plot_latitude.npy'))
        
        self.gen_new_x(density_r=201, 
                        density_theta=1441, 
                        density_phi=37, 
                        t_list=TIME_LIST[1:2], 
                        save_path=os.path.join(save_folder, 'new_x_plot_longitude.npy'))

    def is_ocean_3d_vector(self, data):

        file_path = 'Data/RealData/GLO-MFC_001_024_mask_bathy.nc'
        dataset = nc.Dataset(file_path)
        # all_vars = dataset.variables.keys()

        latitude = dataset.variables['latitude'][:]
        longitude = dataset.variables['longitude'][:]
        deptho = dataset.variables['deptho'][:]
        
        lat = data[:, 1]
        lon = data[:, 2]
        loc_lat = find_closest_vector(latitude, lat)
        loc_lon = find_closest_vector(longitude, lon)

        depth = -data[:, 0]

        masked_land = deptho.mask[loc_lat, loc_lon]

        is_ocean = deptho[loc_lat, loc_lon] >= depth
        is_ocean[masked_land] = False
        is_ocean[lat < -80] = False

        return is_ocean

    def load_data_currents(self, currents_data_path, currents_save_path, train_vali_test=[8, 1, 1], ratio=1, convert_and_filter=True):
        self.ratio = ratio

        # ---------------------- Currents data -----------------------

        if convert_and_filter:

            # Convert NC files, filter the data and standardize them
            print('Coverting the Copernicus NC files and filtering the currents data.')

            for folder in ['cur', 'wcur']:
                folder_path = os.path.join(currents_data_path, folder)
                self.nc_folder_convert_and_filter(folder_path)
            

        # Merge data fron different files and split them into training, validation and test sets.
        self.currents_merge_and_split(data_path=currents_data_path,
                                save_path=currents_save_path, 
                                train_vali_test=train_vali_test) 

        
    def currents_merge_and_split(self, data_path, save_path, train_vali_test=[8, 1, 1], seed=42):

        print('Merging and spliting the currents data.')

        np.random.seed(seed)

        for i in tqdm(['cur', 'wcur']):
            npy_folder = f'{data_path}/{i}'

            # Lists to store all parts for each file
            all_part1 = []
            all_part2 = []
            all_part3 = []

            # Iterate over all CSV files in folder
            for file_name in tqdm(os.listdir(npy_folder)):
                if file_name.endswith(".npy"):
                    # Read CSV file
                    file_path = os.path.join(npy_folder, file_name)
                    df = np.load(file_path)
                    
                    # Split into three parts
                    n = df.shape[0]
                    b_1 = round(n / sum(train_vali_test) * train_vali_test[0])
                    b_2 = round(n / sum(train_vali_test) * (train_vali_test[0] + train_vali_test[1]))

                    sample = np.random.choice(df.shape[0], size=df.shape[0], replace=False)
                    part1 = df[sample[:b_1]]
                    part2 = df[sample[b_1:b_2]]
                    part3 = df[sample[b_2:n]]
                    
                    # Append the three parts to the respective lists
                    all_part1.append(part1)
                    all_part2.append(part2)
                    all_part3.append(part3)

            # Concatenate all parts for each file
            merged_part1 = np.concatenate(all_part1, axis=0)
            merged_part2 = np.concatenate(all_part2, axis=0)
            merged_part3 = np.concatenate(all_part3, axis=0)

            ndarray_check(merged_part1)
            ndarray_check(merged_part2)
            ndarray_check(merged_part3)

            # Export as NPY files
            np.save(os.path.join(save_path, f"{i}_train.npy"), merged_part1)
            np.save(os.path.join(save_path, f"{i}_vali.npy"), merged_part2)
            np.save(os.path.join(save_path, f"{i}_test.npy"), merged_part3)
        
    def nc_folder_convert_and_filter(self, folder_path):
        nc_paths = []
        npy_save_paths = []
        for file in os.listdir(folder_path):
            if file.endswith(".nc"):
                nc_paths.append(os.path.join(folder_path, file))
                npy_save_paths.append(os.path.join(folder_path, file[:-3] + '.npy'))

        for i in tqdm(range(len(nc_paths))):
            print('Processing' + nc_paths[i])
            self.nc_convert_and_filter(nc_path=nc_paths[i], 
                                npy_save_path=npy_save_paths[i])

    def nc_convert_and_filter(self, nc_path, npy_save_path):

        start_date = datetime.datetime(1950, 1, 1, 0, 0)
        end_date = datetime.datetime(1970, 1, 1, 0, 0)
        time_difference = end_date - start_date
        seconds_1950_to_1970 = time_difference.total_seconds()


        dataset = nc.Dataset(nc_path)
        all_vars = list(dataset.variables.keys())

        # vo: northward velocity (v_theta)
        # uo: eastward velocity (v_phi)
        # wo: vertical velocity

        depth = dataset.variables['depth'][:]
        latitude = dataset.variables['latitude'][:]
        longitude = dataset.variables['longitude'][:]
        time = dataset.variables['time'][:]

        if len(all_vars) == 6:
            var_1 = dataset.variables[all_vars[4]][:]
            var_2 = dataset.variables[all_vars[5]][:]
            indices = np.where(~var_1.mask)
            var_1 = var_1[indices]
            var_2 = var_2[indices]
            result = np.column_stack((depth[indices[1]], latitude[indices[2]], longitude[indices[3]], time[indices[0]], var_2, var_1))
        elif len(all_vars) == 5:
            var = dataset.variables[all_vars[4]][:]
            indices = np.where(~var.mask)
            var = var[indices]
            result = np.column_stack((depth[indices[1]], latitude[indices[2]], longitude[indices[3]], time[indices[0]], var))

        result[:, 0] *= -1
        result[:, 3] = result[:, 3] * 3600 - seconds_1950_to_1970 # Seconds from 1st Jan 1970

        if any([self.r_min, self.r_max, self.theta_min, self.theta_max, self.phi_min, self.phi_max, self.t_min, self.t_max]):
            result = self.data_filter(result, self.r_min, self.r_max, self.theta_min, self.theta_max, self.phi_min, self.phi_max, self.t_min, self.t_max)
        # print(pd.DataFrame(result), '\n', pd.DataFrame(result).describe())

        result = result.data
        result = np.asarray(result)

        # random sampling
        if self.ratio < 1:
            result = proportional_sampling(result, self.ratio)

        result_float32 = result.astype(np.float32)
        np.save(npy_save_path, result_float32)

    def load_data_seaice(self, data_path, save_path=None):
        self.seaice_nc2npy(data_path, os.path.join(save_path, 'seaice_all.npy'))
        self.seaice_data_split(os.path.join(save_path, 'seaice_all.npy'), save_path)


    def load_data_tempsal(self, data_path, save_path=None):
        self.tempsal_nc2npy(data_path, save_path)

    def load_vphi_plot(self, 
                       data_path='Data/RealData/RawFiles/GLOBAL_MULTIYEAR_PHY_001_030/mercatorglorys12v1_gl12_mean_202107.nc',
                       save_path='Data/RealData/Global/Currents/ForPlot/v_phi_plot.npy'):

        ds = xr.open_dataset(data_path)

        # Convert the dataset to a pandas dataframe and reset the index
        df = ds[['vo']].to_dataframe().reset_index()

        # Select relevant columns and drop rows with missing values
        result_df = df[['depth', 'latitude', 'longitude', 'time', 'vo']].dropna()

        result_df['time'] = self.convert2_unix_timestamp(result_df['time'])
        result_df['depth'] *= -1

        np.save(save_path, result_df)

    def seaice_data_split(
        self,
        input_path: str,
        output_dir: str = "./",
        val_ratio: float = 1 / 9,
        shuffle: bool = True,
        random_seed: int = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Load .npy file and split into train/validation sets with export functionality
        
        Args:
            input_path: Path to the input .npy file
            output_dir: Output directory (default: current directory)
            val_ratio: Ratio for validation set (0.0-1.0)
            shuffle: Whether to shuffle data before splitting (default: True)
            random_seed: Random seed for reproducibility
        
        Returns:
            Tuple containing (train_set, validation_set) as NumPy arrays
        
        Raises:
            FileNotFoundError: If input file doesn't exist
            ValueError: If val_ratio is not in [0, 1]
        """
        # Validate input parameters
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file not found: {input_path}")
        if not 0 <= val_ratio <= 1:
            raise ValueError(f"Invalid val_ratio {val_ratio}. Must be between 0 and 1")

        # Load data from .npy file
        data = np.load(input_path)
        data = data[data[:, 3] > 0]
        data = data[:, :3]
        
        # Handle empty dataset
        if len(data) == 0:
            raise ValueError("Input file contains empty dataset")

        # Shuffle data if required
        if shuffle:
            rng = np.random.default_rng(random_seed)
            indices = rng.permutation(len(data))
            data = data[indices]

        # Calculate split index
        split_idx = int(len(data) * (1 - val_ratio))
        
        # Split dataset
        train_data = data[:split_idx]
        val_data = data[split_idx:]

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Generate output filenames
        train_path = os.path.join(output_dir, f"seaice_train.npy")
        val_path = os.path.join(output_dir, f"seaice_vali.npy")

        # Export split datasets
        np.save(train_path, train_data)
        np.save(val_path, val_data)

        return train_data, val_data

    def collect_nc_files(self, nc_path):
        nc_paths = []
        npy_save_paths = []
        for file in os.listdir(nc_path):
            if file.endswith(".nc"):
                nc_paths.append(os.path.join(nc_path, file))
                npy_save_paths.append(os.path.join(nc_path, file[:-3] + '.npy'))

        return nc_paths, npy_save_paths
    
    def convert2_unix_timestamp(self, t):
        t = pd.to_datetime(t)
        t = t.astype('int64') / 10 ** 9

        return t

    def seaice_nc2npy(self, nc_path, save_path):
        # Collect all .nc file paths in the specified directory
        nc_paths, npy_save_paths = self.collect_nc_files(nc_path)

        # List to store all processed dataframes
        all_results = []

        # Process each .nc file
        for i, nc_path in enumerate(tqdm(nc_paths, desc="Processing files")):
            print('Processing ' + nc_paths[i])

            # Open the .nc file using xarray
            ds = xr.open_dataset(nc_path)

            # Convert the dataset to a pandas dataframe and reset the index
            df = ds[['sithick']].to_dataframe().reset_index()

            # Select relevant columns and drop rows with missing values
            result_df = df[['latitude', 'longitude', 'time', 'sithick']].dropna()

            # Convert the 'time' column to datetime and then to Unix timestamp (seconds since epoch)
            result_df['time'] = self.convert2_unix_timestamp(result_df['time'])

            # Append the processed dataframe values to the list
            all_results.append(result_df.values)

        # Concatenate all processed dataframes into a single NumPy array
        combined_results = np.concatenate(all_results, axis=0)

        # Save the combined results as a .npy file
        np.save(save_path, combined_results)
        print(f"Saved combined results to {save_path}")

    def tempsal_nc2npy(self, nc_path, save_path):
        # Collect all .nc file paths in the specified directory
        nc_paths, _ = self.collect_nc_files(nc_path)

        npy_save_paths = []
        for file in nc_paths:
            npy_save_paths.append(os.path.join(save_path, f'{os.path.basename(file)}.npy'))
        gen_folder(save_path)

        # Process each .nc file
        for i, nc_path in enumerate(tqdm(nc_paths, desc="Processing files")):
            print('Processing ' + nc_paths[i])

            # Open the .nc file using xarray
            ds = xr.open_dataset(nc_path)

            # Convert the dataset to a pandas dataframe and reset the index
            df = ds[['thetao', 'so']].to_dataframe().reset_index()

            # Select relevant columns and drop rows with missing values
            result_df = df[['depth', 'latitude', 'longitude', 'time', 'thetao', 'so']].dropna()

            # Convert the 'time' column to datetime and then to Unix timestamp (seconds since epoch)
            result_df['time'] = self.convert2_unix_timestamp(result_df['time'])

            result_df['depth'] *= -1

            # Save the combined results as a .npy file
            np.save(npy_save_paths[i], result_df)
            print(f"Saved results to {npy_save_paths[i]}")


    def data_filter(self, data, r_min, r_max, theta_min, theta_max, phi_min, phi_max, t_min, t_max):

        # t_min = pd.to_datetime(t_min).value / 10 ** 9
        # t_max = pd.to_datetime(t_max).value / 10 ** 9

        if phi_min <= phi_max:
            data = data[(data[:, 0] >= r_min) & (data[:, 0] <= r_max) &
                        (data[:, 1] >= theta_min) & (data[:, 1] <= theta_max) &
                        (data[:, 2] >= phi_min) & (data[:, 2] <= phi_max) &
                        (data[:, 3] >= t_min) & (data[:, 3] <= t_max)]
            
        else:
            data = data[(data[:, 0] >= r_min) & (data[:, 0] <= r_max) &
                        (data[:, 1] >= theta_min) & (data[:, 1] <= theta_max) &
                        ((data[:, 2] >= phi_min) | (data[:, 2] <= phi_max)) &
                        (data[:, 3] >= t_min) & (data[:, 3] <= t_max)]

        return data
    
    def data_filter_theta(self, data, sublearner=1, normalize=False):

        if not normalize:
            if sublearner == 1:
                data = data[(data[:, 1] < 45) & (data[:, 1] > -45)]
            elif sublearner == 2:
                data = data[(data[:, 1] > 40) | (data[:, 1] < -40)]
        else:
            if sublearner == 1:
                data = data[(data[:, 1] > np.deg2rad(45)) & (data[:, 1] < np.deg2rad(135))]
            elif sublearner == 2:
                data = data[(data[:, 1] < np.deg2rad(50)) | (data[:, 1] > np.deg2rad(130))]

        return data

    # def download_copernicus_data(self, data_path=None):
    #     copernicusmarine.get(
    #         dataset_id='cmems_mod_glo_phy-cur_anfc_0.083deg_P1M-m',
    #         username='zxiong1',
    #         password='Leroy&2253',
    #         filter="*2021/*",
    #         create_file_list="Data/RealData/currents_files_to_download.txt"
    #     )

    def coordinates_rotation(self, data=None, inverse=False, tensor=False):
        # Choose the library (PyTorch or NumPy) based on the tensor flag
        lib = torch if tensor else np

        # Clone the input data to avoid in-place operations
        new_data = data.clone() if tensor else data.copy()

        # Extract theta and phi from the input data
        theta = new_data[:, 1]
        phi = new_data[:, 2]

        # Precompute sin(theta), cos(theta), sin(phi), and cos(phi)
        sin_theta = lib.sin(theta)
        cos_theta = lib.cos(theta)
        cos_phi = lib.cos(phi)
        sin_phi = lib.sin(phi)

        # Convert spherical coordinates to Cartesian coordinates
        x = sin_theta * cos_phi
        y = sin_theta * sin_phi
        z = cos_theta

        # Apply rotation based on the inverse flag
        if not inverse:
            x_2, y_2, z_2 = x, -z, y
        else:
            x_2, y_2, z_2 = x, z, -y

        # Compute the new theta_2 using arccos
        theta_2 = lib.arccos(z_2)

        # Compute the new phi_2 using vectorized conditional operations
        phi_2 = lib.zeros_like(x_2)
        phi_2 = lib.where(x_2 > 0, lib.arctan(y_2 / x_2), phi_2)
        phi_2 = lib.where((y_2 >= 0) & (x_2 < 0), lib.arctan(y_2 / x_2) + lib.pi, phi_2)
        phi_2 = lib.where((y_2 < 0) & (x_2 < 0), lib.arctan(y_2 / x_2) - lib.pi, phi_2)
        phi_2 = lib.where((y_2 > 0) & (x_2 == 0), lib.pi / 2, phi_2)
        phi_2 = lib.where((y_2 < 0) & (x_2 == 0), -lib.pi / 2, phi_2)

        # Adjust phi_2 to be within the range [0, 2*pi] using a mask (non-inplace operation)
        mask = phi_2 < 0
        phi_2 = phi_2 + mask * 2 * lib.pi  # Replace phi_2[phi_2 < 0] += 2 * lib.pi

        # Avoid inplace assignment by constructing a new tensor
        if tensor:
            new_data = torch.cat([
                new_data[:, 0:1],  # Keep the first column unchanged
                (-theta_2 + lib.pi).unsqueeze(1),  # Update theta_2
                (-phi_2 + 2 * lib.pi).unsqueeze(1), # Update phi_2
                new_data[:, 3:4]
            ], dim=1)
        else:
            # For NumPy, direct assignment is safe
            new_data[:, 1] = -theta_2 + lib.pi
            new_data[:, 2] = -phi_2 + 2 * lib.pi

        return new_data


    def read_pred_results(self, pre_path):
        results = []
        for var in VAR_NAMES[:5]:
            file_path = os.path.join(pre_path, f'{var}_pred.npy')
            results.append(np.load(file_path))

        return results



# if __name__ == '__main__':
#     RealData.load_data('../Data/RealData')