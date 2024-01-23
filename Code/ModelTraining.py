import numpy as np
import pandas as pd
import deepxde as dde
import torch
from deepxde.backend import torch
from IPython.display import display
import shutil, os
import time
from tqdm import tqdm

from PrimitiveEquations import primitive_equations


def data_argo_train(data_path):
    data = np.load(os.path.join(data_path, 'Argo/argo_train_scale.npy'))
    zero_rows = np.nonzero((data == 0).all(axis=1))
    data = np.delete(data, zero_rows, axis=0)
    # print('argo_train:\n', data.describe())
    return data[:, 0], data[:, 1], data[:, 2], data[:, 3], data[:, 4], data[:, 5]

def data_argo_validate(data_path):
    data = np.load(os.path.join(data_path, 'Argo/argo_vali_scale.npy'))
    return data[:, 0], data[:, 1], data[:, 2], data[:, 3], data[:, 4], data[:, 5]

def data_cur_train(data_path):
    data = np.load(os.path.join(data_path, 'Currents/cur_train_scale.npy'))
    zero_rows = np.nonzero((data == 0).all(axis=1))
    data = np.delete(data, zero_rows, axis=0)
    # print('cur_train:\n', data.describe())
    return data[:, 0], data[:, 1], data[:, 2], data[:, 3], data[:, 4], data[:, 5]

def data_cur_validate(data_path):
    data = np.load(os.path.join(data_path, 'Currents/cur_vali_scale.npy'))
    return data[:, 0], data[:, 1], data[:, 2], data[:, 3], data[:, 4], data[:, 5]

def data_wcur_train(data_path):
    data = np.load(os.path.join(data_path, 'Currents/wcur_train_scale.npy'))
    zero_rows = np.nonzero((data == 0).all(axis=1))
    data = np.delete(data, zero_rows, axis=0)
    # print('wcur_train:\n', data.describe())
    return data[:, 0], data[:, 1], data[:, 2], data[:, 3], data[:, 4]

def data_wcur_validate(data_path):
    data = np.load(os.path.join(data_path, 'Currents/wcur_vali_scale.npy'))
    return data[:, 0], data[:, 1], data[:, 2], data[:, 3], data[:, 4]


def penn_training(data_path, domain_points_path, batch_size, init_beta_tau, init_beta_sigma, num_domain, num_boundary, input_output_size, n_layers, activation, initializer, model_save_path_1, model_save_path_2, variable_save_path, save_period, resample_period_1, resample_period_2, num_iter_1, num_iter_2, optimizer, learning_rate_1, learning_rate_2, loss_weights_1, loss_weights_2):

    start_time = time.time()
    print("Time:", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

    # pd.set_option('display.width', 1000)
    # pd.options.display.max_columns = 40

    # ------------------------ Get the Data -------------------------

    # training data
    [ob_r_argo, ob_theta_argo, ob_phi_argo, ob_t_argo, ob_temp, ob_sal] = data_argo_train(data_path)
    # ob_temp = np.reshape(ob_temp.to_numpy(), (-1, 1))
    # ob_sal = np.reshape(ob_sal.to_numpy(), (-1, 1))
    ob_temp = ob_temp.reshape(-1, 1)
    ob_sal = ob_sal.reshape(-1, 1)
    # print(type(ob_temp))
    ob_coordinate_t_argo = np.column_stack((ob_r_argo, ob_theta_argo, ob_phi_argo, ob_t_argo))
    ob_temp_sal = np.column_stack((ob_temp, ob_sal))
    bs_argo = 512
    observe_temp = dde.icbc.PointSetBC(ob_coordinate_t_argo, ob_temp, component=0, batch_size=bs_argo, shuffle=True)
    observe_sal = dde.icbc.PointSetBC(ob_coordinate_t_argo, ob_sal, component=1, batch_size=bs_argo, shuffle=True)

    [ob_r_wcur, ob_theta_wcur, ob_phi_wcur, ob_t_wcur, ob_w] = data_wcur_train(data_path)
    # ob_w = np.reshape(ob_w.to_numpy(), (-1, 1))
    ob_w = ob_w.reshape(-1, 1)
    ob_coordinate_t_wcur = np.column_stack((ob_r_wcur, ob_theta_wcur, ob_phi_wcur, ob_t_wcur))
    bs_wcur = 512
    observe_w = dde.icbc.PointSetBC(ob_coordinate_t_wcur, ob_w, component=2, batch_size=bs_wcur, shuffle=True)

    [ob_r_cur, ob_theta_cur, ob_phi_cur, ob_t_cur, ob_v1, ob_v2] = data_cur_train(data_path)
    # ob_v1 = np.reshape(ob_v1.to_numpy(), (-1, 1))
    # ob_v2 = np.reshape(ob_v2.to_numpy(), (-1, 1))
    ob_v1 = ob_v1.reshape(-1, 1)
    ob_v2 = ob_v2.reshape(-1, 1)
    ob_coordinate_t_cur = np.column_stack((ob_r_cur, ob_theta_cur, ob_phi_cur, ob_t_cur))
    ob_v1_v2 = np.column_stack((ob_v1, ob_v2))
    bs_cur = 512
    observe_v1 = dde.icbc.PointSetBC(ob_coordinate_t_cur, ob_v1, component=3, batch_size=bs_cur, shuffle=True)
    observe_v2 = dde.icbc.PointSetBC(ob_coordinate_t_cur, ob_v2, component=4, batch_size=bs_cur, shuffle=True)


    # Get the validation data
    [ob_r_va, ob_theta_va, ob_phi_va, ob_t_va, ob_temp_va, ob_sal_va] = data_argo_validate(data_path)
    ob_coordinate_t_argo_va = np.column_stack((ob_r_va, ob_theta_va, ob_phi_va, ob_t_va))
    ob_temp_sal_va = np.column_stack((ob_temp_va, ob_sal_va))

    [ob_r_wcur_va, ob_theta_wcur_va, ob_phi_wcur_va, ob_t_wcur_va, ob_w_va] = data_wcur_validate(data_path)
    # ob_w_va = np.reshape(ob_w_va.to_numpy(), (-1, 1))
    ob_w_va = ob_w_va.reshape(-1, 1)
    ob_coordinate_t_wcur_va = np.column_stack((ob_r_wcur_va, ob_theta_wcur_va, ob_phi_wcur_va, ob_t_wcur_va))

    [ob_r_cur_va, ob_theta_cur_va, ob_phi_cur_va, ob_t_cur_va, ob_v1_va, ob_v2_va] = data_cur_validate(data_path)
    ob_coordinate_t_cur_va = np.column_stack((ob_r_cur_va, ob_theta_cur_va, ob_phi_cur_va, ob_t_cur_va))
    ob_v1_v2_va = np.column_stack((ob_v1_va, ob_v2_va))


    # ---------------------- Spatial domain: pointcloud ---------------------------
    domain_points = np.load(os.path.join(domain_points_path, 'domain_points.npy'))
    zero_rows = np.nonzero((domain_points == 0).all(axis=1))
    domain_points = np.delete(domain_points, zero_rows, axis=0)
    boundary_points = domain_points[domain_points[:, 1] == 1]

    space_domain = dde.geometry.PointCloud(points=domain_points, boundary_points=boundary_points, boundary_normals=None)
    # Time domain: t
    time_domain = dde.geometry.TimeDomain(0, 1)
    # Spatio-temporal domain
    geomtime = dde.geometry.GeometryXTime(space_domain, time_domain)


    # Boundary conditions
    def bound_upper(x, _):
        return np.isclose(x[0], 1)

    bc_pi = dde.icbc.DirichletBC(geomtime, lambda x: 0, bound_upper, component=5)

    # Unknown parameters
    beta_tau = dde.Variable(init_beta_tau)
    beta_sigma = dde.Variable(init_beta_sigma)


    # Training dataset and Loss
    data = dde.data.PDE(
        geometry=geomtime,
        # pde=None,
        pde=primitive_equations,
        # bcs=[],
        bcs=[bc_pi, observe_temp, observe_sal, observe_w, observe_v1, observe_v2],
        num_boundary=num_boundary,
        num_domain=num_domain,
    )
    data_ = dde.data.PDE(
        geometry=geomtime,
        pde=None,
        bcs=[observe_temp, observe_sal, observe_w, observe_v1, observe_v2],
        num_domain=0,
    )


    # Neural Network setup
    net = dde.nn.FNN(layer_sizes=input_output_size, activation=activation, kernel_initializer=initializer)
    model = dde.Model(data_, net)


    # callbacks for storing results
    checker_1 = dde.callbacks.ModelCheckpoint(
        f"{model_save_path_1}/model", save_better_only=False, period=save_period
    )
    checker_2 = dde.callbacks.ModelCheckpoint(
        f"{model_save_path_2}/model", save_better_only=False, period=save_period
    )
    variable = dde.callbacks.VariableValue([beta_tau, beta_sigma], period=save_period, filename=variable_save_path)
    resampler = dde.callbacks.PDEPointResampler(period=resample_period_1, pde_points=True, bc_points=True)
    resampler_ = dde.callbacks.PDEPointResampler(period=resample_period_2, pde_points=True, bc_points=True)


    # Compile, train and save model
    torch.cuda.empty_cache()

    iter = [num_iter_1, num_iter_2]
    disp = 100


    dde.config.set_random_seed(1921)
    model.compile(optimizer, lr=learning_rate_1, loss_weights=loss_weights_1, external_trainable_variables=[beta_tau, beta_sigma])
    loss_history, train_state = model.train(
        iterations=iter[0], callbacks=[resampler_, checker_1], display_every=disp, disregard_previous_best=False
    )
    # dde.saveplot(loss_history, train_state, issave=True, isplot=True)



    model.change_data(data)


    model.compile(optimizer, lr=learning_rate_2, loss_weights=loss_weights_2, external_trainable_variables=[beta_tau, beta_sigma])
    model.restore(f'{model_save_path_1}/model-{num_iter_1}.pt')
    loss_history, train_state = model.train(
        iterations=iter[0], callbacks=[resampler, checker_2, variable], display_every=disp, disregard_previous_best=False
    )
    # dde.saveplot(loss_history, train_state, issave=True, isplot=True)



    # Calculate validation loss
    model_num = int(sum(iter) / 1000)
    print('Calculate validation loss...')
    loss_va = np.empty([model_num, 6])
    loss = np.empty([model_num, 6])

    train_mean_std = pd.read_csv(data_path + 'train_mean_std.csv', index_col=0)
    ob_temp_sal[:, 0] = ob_temp_sal[:, 0] * train_mean_std.loc['std', 'temp'] + train_mean_std.loc['mean', 'temp']
    ob_temp_sal[:, 1] = ob_temp_sal[:, 1] * train_mean_std.loc['std', 'sal'] + train_mean_std.loc['mean', 'sal']
    ob_v1_v2[:, 0] = ob_v1_v2[:, 0] * train_mean_std.loc['std', 'v_theta'] + train_mean_std.loc['mean', 'v_theta']
    ob_v1_v2[:, 1] = ob_v1_v2[:, 1] * train_mean_std.loc['std', 'v_phi'] + train_mean_std.loc['mean', 'v_phi']
    ob_w = ob_w * train_mean_std.loc['std', 'w'] + train_mean_std.loc['mean', 'w']


    MODEL = np.linspace(1000, sum(iter), model_num, endpoint=True, dtype = 'int').tolist()
    for i in tqdm(range(model_num)):
        model.compile("adam", lr= 1 / (10 ** 4))
        model.restore(f'{model_save_path}/model-' + str(MODEL[i]) + '.pt')
        
        # Train loss
        fit_argo = fit_cur = fit_wcur = np.empty([0, 8])
        for j in tqdm(range(round(len(ob_coordinate_t_argo) // 1e5) + 1)):
            if (j + 1) * 1e5 > len(ob_coordinate_t_argo):
                fit_argo_t = model.predict(ob_coordinate_t_argo[round(j * 1e5):, ])
            else:
                fit_argo_t = model.predict(ob_coordinate_t_argo[round(j * 1e5):round((j + 1) * 1e5), ])
            
            fit_argo = np.vstack((fit_argo, fit_argo_t))
            
        for j in tqdm(range(round(len(ob_coordinate_t_cur) // 1e5) + 1)):
            if (j + 1) * 1e5 > len(ob_coordinate_t_cur):
                fit_cur_t = model.predict(ob_coordinate_t_cur[round(j * 1e5):, ])
                fit_wcur_t = model.predict(ob_coordinate_t_wcur[round(j * 1e5):, ])
            else:
                fit_cur_t = model.predict(ob_coordinate_t_cur[round(j * 1e5):round((j + 1) * 1e5), ])
                fit_wcur_t = model.predict(ob_coordinate_t_wcur[round(j * 1e5):round((j + 1) * 1e5), ])
                
            fit_cur = np.vstack((fit_cur, fit_cur_t))
            fit_wcur = np.vstack((fit_wcur, fit_wcur_t))
            
        fit_argo = Functions.anti_normalization(fit_argo, train_mean_std)
        fit_cur = Functions.anti_normalization(fit_cur, train_mean_std)
        fit_wcur = Functions.anti_normalization(fit_wcur, train_mean_std)
        
        loss[i, 1:3] = np.mean((fit_argo[:, :2] - ob_temp_sal) ** 2, axis=0)
        loss[i, 4:6] = np.mean((fit_cur[:, 3:5] - ob_v1_v2) ** 2, axis=0)
        loss[i, 3] = np.mean((fit_wcur[:, 2] - ob_w.flatten()) ** 2, axis=0)
        
        # Validation loss
        n = 3e5
        fit_argo_va = fit_cur_va = fit_wcur_va = np.empty([0, 8])
        for j in range(round(len(ob_coordinate_t_argo_va) // n) + 1):
            if (j + 1) * n > len(ob_coordinate_t_argo_va):
                fit_argo_t_va = model.predict(ob_coordinate_t_argo_va[round(j * n):, ])
                fit_cur_t_va = model.predict(ob_coordinate_t_cur_va[round(j * n):, ])
                fit_wcur_t_va = model.predict(ob_coordinate_t_wcur_va[round(j * n):, ])
            else:
                fit_argo_t_va = model.predict(ob_coordinate_t_argo_va[round(j * n):round((j + 1) * n), ])
                fit_cur_t_va = model.predict(ob_coordinate_t_cur_va[round(j * n):round((j + 1) * n), ])
                fit_wcur_t_va = model.predict(ob_coordinate_t_wcur_va[round(j * n):round((j + 1) * n), ])
            
            fit_argo_va = np.vstack((fit_argo_va, fit_argo_t_va))            
            fit_cur_va = np.vstack((fit_cur_va, fit_cur_t_va))
            fit_wcur_va = np.vstack((fit_wcur_va, fit_wcur_t_va))

        fit_argo_va = Functions.anti_normalization(fit_argo_va, train_mean_std)
        fit_cur_va = Functions.anti_normalization(fit_cur_va, train_mean_std)
        fit_wcur_va = Functions.anti_normalization(fit_wcur_va, train_mean_std)
        loss_va[i, 1:3] = np.mean((fit_argo_va[:, :2] - ob_temp_sal_va) ** 2, axis=0)
        loss_va[i, 4:6] = np.mean((fit_cur_va[:, 3:5] - ob_v1_v2_va) ** 2, axis=0)
        loss_va[i, 3] = np.mean((fit_wcur_va[:, 2] - ob_w_va.flatten()) ** 2, axis=0)
        
        loss[i, 0] = loss_va[i, 0] = MODEL[i]
    np.savetxt('loss_va.csv', loss_va, delimiter=',')
    np.savetxt('loss.csv', loss, delimiter=',')





    # -------------------- Test loss -------------------
    train_mean_std = pd.read_csv(data_path + 'train_mean_std.csv', index_col=0)
    loss_va = pd.read_csv('loss_va.csv')
    var_names = ['temp', 'sal', 'w', 'v_theta', 'v_phi']
    for var in range(5):
        r_minow_index = np.argmin(loss_va.iloc[:, (var + 1)])
        best_model = round(loss_va.iloc[r_minow_index, 0])
        model.compile("adam", lr= 1 / (10 ** 5))
        print('Best model (' + var_names[var] + '):', best_model)
        model.restore(f'{model_save_path}/model-' + str(best_model) + '.pt', verbose=0)

        for test_type in ['', '_pre']:
            if var_names[var] in ['temp', 'sal']:
                df_test = pd.read_csv(data_path + 'Argo/argo_test' + test_type + '_scale.csv')
            elif var_names[var] == 'w':
                df_test = pd.read_csv(data_path + 'Currents_m/wcur_test' + test_type + '_scale.csv')
            else:
                df_test = pd.read_csv(data_path + 'Currents_m/cur_test' + test_type + '_scale.csv')
            
            ob_coordinate_t_test = df_test[['r', 'theta', 'phi', 't']].values
            ob_test = df_test[[var_names[var]]].values

            
            n = 3e5
            fit_test = np.empty([0, 8])
            for j in tqdm(range(round(len(ob_coordinate_t_test) // n) + 1)):
                if (j + 1) * n > len(ob_coordinate_t_test):
                    fit_test_t = model.predict(ob_coordinate_t_test[round(j * n):, ])
                else:
                    fit_test_t = model.predict(ob_coordinate_t_test[round(j * n):round((j + 1) * n), ])
                
                fit_test = np.vstack((fit_test, fit_test_t))            

            fit_test[:, var] = fit_test[:, var] * train_mean_std.loc['std', var_names[var]] + train_mean_std.loc['mean', var_names[var]]

            RMSE = np.sqrt(np.mean((fit_test[:, var] - ob_test.flatten()) ** 2, axis=0))
            
            print('RMSE (', var_names[var], ', ', test_type, '): ', RMSE, sep='')


    end_time = time.time()
    time_range = end_time - start_time
    print("Time:", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    print("Time taken:", time_range, 'seconds or ', time_range / 3600, 'hours')