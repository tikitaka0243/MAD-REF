import numpy as np
import deepxde as dde
import torch
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib.colors import ListedColormap
import sys
import pandas as pd
import fnmatch

file_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(file_path)

from DeepXDEModification import Model2, MDRF_Net_SIMULATION, saveplot_2
from OtherMethods import gaussian_process_inference, regression_kriging


def unpack_x(X):
    x = X[:, 0:1]
    z = X[:, 1:2]
    t = X[:, 2:3]
    return x, z, t


def unpack_y(Y):
    tau = Y[:, 0:1]
    w = Y[:, 1:2]
    v = Y[:, 2:3]
    p = Y[:, 3:4]
    if Y.shape[1] == 5:
        q = Y[:, 4:5]
        return tau, w, v, p, q
    return tau, w, v, p


def taylor_green_vortex(x, z, t, eta = 0.01, zeta = 0.01, zeta_tau = 0.01, save=False, save_path='Data/simulation_solution.npy'):
    tau = np.sin(2 * np.pi * z) * np.exp(-4 * np.pi**2 * zeta_tau * t)
    w = np.cos(2 * np.pi * x) * np.sin(2 * np.pi * z) * np.exp(-4 * np.pi**2 * (eta + zeta) * t)
    v = -np.sin(2 * np.pi * x) * np.cos(2 * np.pi * z) * np.exp(-4 * np.pi**2 * (eta + zeta) * t)
    p = 0.25 * np.cos(4 * np.pi * x) * np.exp(-8 * np.pi**2 * (eta + zeta) * t) + np.cos(2 * np.pi * z) * np.exp(-4 * np.pi**2 * zeta_tau * t) / (2 * np.pi)
    
    Q = np.pi * np.cos(2 * np.pi * x) * np.sin(4 * np.pi * z) * np.exp(-4 * np.pi**2 * (eta + zeta + zeta_tau) * t)

    if save:
        solution = np.column_stack((tau, w, v, p, Q))
        np.save(save_path, solution)

    return tau, w, v, p


def simulation_data_domain_check(x, z):
    x = (x - 0.5) * 2
    z = (z - 0.5) * 2
    
    eq = np.power(np.abs(x), 3) + np.power(np.abs(z), 3)
    output = eq <= 0.7 ** 3
    
    return output
    

def gen_simulate_data(num=10000, noise_std=0.1, simulation_data_path = 'Data/Simulation/Train', eta = 0.01, zeta = 0.01, zeta_tau = 0.01):
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

    tau, w, v, p = taylor_green_vortex(x, z, t, eta, zeta, zeta_tau)

    tau = tau + np.random.normal(loc=0, scale=noise_std, size=num)
    w = w + np.random.normal(loc=0, scale=noise_std, size=num)
    v = v + np.random.normal(loc=0, scale=noise_std, size=num)
    p = p + np.random.normal(loc=0, scale=noise_std, size=num) 
    
    np.save(f'{simulation_data_path}/tau.npy', tau)
    np.save(f'{simulation_data_path}/w.npy', w)
    np.save(f'{simulation_data_path}/v.npy', v)
    np.save(f'{simulation_data_path}/p.npy', p)
    
    print('Done.')
    
    
def simulation_read_data(simulation_data_path):
    tau = np.load(f'{simulation_data_path}/tau.npy')
    w = np.load(f'{simulation_data_path}/w.npy')
    v = np.load(f'{simulation_data_path}/v.npy')
    p = np.load(f'{simulation_data_path}/p.npy')
    
    x_z_t = np.load(f'{simulation_data_path}/x_z_t.npy')
    
    return tau, w, v, p, x_z_t
    
    
def simulation_data_plot(simulation_data_path = 'Data/Simulation/Train', save_path='Output/Plot/Simulation/Data'):
    print('Plotting simualtion data...')
    
    unique_t = np.linspace(0, 1, 5, endpoint=True)
    simulation_gen_plot_folders(save_path, unique_t)
    
    tau, w, v, p, x_z_t = simulation_read_data(simulation_data_path)
    Y = np.column_stack((tau, w, v, p))
    x, z, t = unpack_x(x_z_t)
    
    for t_i, t_ in tqdm(enumerate(unique_t), total=len(unique_t)):
        plot_indices = find_closest_ts(t_, t.flatten(), n=200)
        x_plot = x[plot_indices]
        z_plot = z[plot_indices]
        
        for var in range(4):
            Y_t = Y[plot_indices]
            simulation_plot_one(var, t_, t_i, x_plot, z_plot, Y_t, save_path, s=10, marker='.')
            
    print('Done.')
    
    
def find_closest_ts(t_point, t_vector, n=100):
    dis = np.abs(t_vector - t_point)
    indices_of_smallest = np.argpartition(dis, n)[:n]
    return indices_of_smallest
    

def simulation_load_data(simulation_data_path = 'Data/Simulation/Train', bc=512):
    ob_coordinate_t = np.load(f'{simulation_data_path}/x_z_t.npy')
    
    ob_tau, ob_w, ob_v, ob_p, x_z_t = simulation_read_data(simulation_data_path)

    ob_tau = ob_tau.reshape(-1, 1)
    ob_w = ob_w.reshape(-1, 1)
    ob_v = ob_v.reshape(-1, 1)
    ob_p = ob_p.reshape(-1, 1)

    observe_tau = dde.icbc.PointSetBC(ob_coordinate_t, ob_tau, component=0)
    observe_w = dde.icbc.PointSetBC(ob_coordinate_t, ob_w, component=1)
    observe_v = dde.icbc.PointSetBC(ob_coordinate_t, ob_v, component=2)
    observe_p = dde.icbc.PointSetBC(ob_coordinate_t, ob_p, component=3)

    return observe_tau, observe_w, observe_v, observe_p


def simulation_geotime():
    space_domain = dde.geometry.Rectangle([0.0, 0.0], [1.0, 1.0])
    time_domain = dde.geometry.TimeDomain(0.0, 1.0)
    return dde.geometry.GeometryXTime(space_domain, time_domain)


def simulation_unknown_parameters():
    zeta = dde.Variable(0.0)
    zeta_tau = dde.Variable(0.0)

    return zeta, zeta_tau


def simulation_para_transform(eta, zeta, eta_tau, zeta_tau):
    output = []
    for para in [eta, zeta, eta_tau, zeta_tau]:
        if isinstance(para, torch.Tensor):
            output.append(torch.sigmoid(para) / 10)
        else:
            output.append(para)
            
    return output


def primitive_equations_2d(X, Y, eta, zeta, eta_tau, zeta_tau, Q=False):
    x, z, t = unpack_x(X)
    if Q:
        tau, w, v, p, q = unpack_y(Y)
    else:
        tau, w, v, p = unpack_y(Y)
    
    # eta, zeta, eta_tau, zeta_tau = simulation_para_transform(eta, zeta, eta_tau, zeta_tau)

    dtau_x = dde.grad.jacobian(Y, X, i=0, j=0)
    dtau_z = dde.grad.jacobian(Y, X, i=0, j=1)
    dtau_t = dde.grad.jacobian(Y, X, i=0, j=2)
    dw_z = dde.grad.jacobian(Y, X, i=1, j=1)
    dv_x = dde.grad.jacobian(Y, X, i=2, j=0)
    dv_z = dde.grad.jacobian(Y, X, i=2, j=1)
    dv_t = dde.grad.jacobian(Y, X, i=2, j=2)
    dp_x = dde.grad.jacobian(Y, X, i=3, j=0)
    dp_z = dde.grad.jacobian(Y, X, i=3, j=1)

    dtau_x_2 = dde.grad.hessian(Y, X, component=0, i=0, j=0)
    dtau_z_2 = dde.grad.hessian(Y, X, component=0, i=1, j=1)
    dv_x_2 = dde.grad.hessian(Y, X, component=2, i=0, j=0)
    dv_z_2 = dde.grad.hessian(Y, X, component=2, i=1, j=1)

    if Q:
        Q = q
    else:
        Q = np.pi * torch.cos(2 * np.pi * x) * torch.sin(4 * np.pi * z) * torch.exp(-4 * np.pi**2 * (eta + zeta + zeta_tau) * t)

    equa_1 = dv_t + v * dv_x + w * dv_z - eta * dv_x_2 - zeta * dv_z_2 + dp_x
    equa_2 = dp_z + tau
    equa_3 = dv_x + dw_z
    equa_4 = dtau_t + v * dtau_x + w * dtau_z - eta_tau * dtau_x_2 - zeta_tau * dtau_z_2 - Q

    return equa_1, equa_2, equa_3, equa_4



def simulation_train_data(geomtime, primitive_equations, observe_tau, observe_w, observe_v, observe_p, num_domain=512, num_boundary=512):
    bc_v_x, bc_v_z, bc_w_x, bc_w_z, bc_w_z_dirichlet, bc_p_x, bc_p_z, bc_tau_x, bc_tau_z = boundary_conditions(geomtime)
    data = dde.data.PDE(
        geometry=geomtime,
        pde=primitive_equations,
        bcs=[bc_v_x, bc_v_z, bc_w_x, bc_w_z, bc_w_z_dirichlet, bc_p_x, bc_p_z, bc_tau_x, bc_tau_z, observe_tau, observe_w, observe_v],
        num_domain=num_domain,
        num_boundary=num_boundary,
    )

    data_ = dde.data.PDE(
        geometry=geomtime,
        pde=None,
        bcs=[observe_tau, observe_w, observe_v],
        num_domain=0,
    )

    return data, data_


def boundary_conditions(geomtime):
    bc_tau_x = dde.icbc.PeriodicBC(geomtime, 0, lambda _, on_boundary: on_boundary, derivative_order=0, component=0)
    bc_tau_z = dde.icbc.PeriodicBC(geomtime, 1, lambda _, on_boundary: on_boundary, derivative_order=0, component=0)
    bc_w_x = dde.icbc.PeriodicBC(geomtime, 0, lambda _, on_boundary: on_boundary, derivative_order=0, component=1)
    bc_w_z = dde.icbc.PeriodicBC(geomtime, 1, lambda _, on_boundary: on_boundary, derivative_order=0, component=1)
    bc_w_z_dirichlet = dde.icbc.DirichletBC(geomtime, lambda x: 0, z_boundary, component=1)
    bc_v_x = dde.icbc.PeriodicBC(geomtime, 0, lambda _, on_boundary: on_boundary, derivative_order=0, component=2)
    bc_v_z = dde.icbc.PeriodicBC(geomtime, 1, lambda _, on_boundary: on_boundary, derivative_order=0, component=2)
    bc_p_x = dde.icbc.PeriodicBC(geomtime, 0, lambda _, on_boundary: on_boundary, derivative_order=0, component=3)
    bc_p_z = dde.icbc.PeriodicBC(geomtime, 1, lambda _, on_boundary: on_boundary, derivative_order=0, component=3)
    
    return [bc_v_x, bc_v_z, bc_w_x, bc_w_z, bc_w_z_dirichlet, bc_p_x, bc_p_z, bc_tau_x, bc_tau_z]


def simulation_model(data_, n_layers=[4] * 4, layer_width=32, layer_sizes=[3] + [4], activation="tanh", kernel_initializer="Glorot normal", Q=False):
    net = MDRF_Net_SIMULATION(n_layers=[4] * 4, layer_width=32, layer_sizes=[3] + [4], activation="tanh", kernel_initializer="Glorot normal", Q=Q)
    model = Model2(data_, net)

    return model


def callbacks(output_dir, save_period=10000, unknown_parameters=None, variable_file_name='variable.dat'):
    checker = dde.callbacks.ModelCheckpoint(os.path.join(output_dir, 'model'), save_better_only=False, period=save_period)
    resampler = dde.callbacks.PDEPointResampler(period=100, pde_points=True, bc_points=True)
    variable = dde.callbacks.VariableValue(unknown_parameters, period=100, filename=os.path.join(output_dir, variable_file_name), precision=6)

    return checker, resampler, variable


def z_boundary(x, on_boundary):
    return on_boundary and (np.isclose(x[1], 0) or np.isclose(x[1], 1))


def simulation_train(simulation_data_path='Data/Simulation/Train', output_dir='Output/Model/Simulation/2Steps', iters=[10000, 20000], display=100, lrs=[1e-4, 1e-5], variable_file_name='variable.dat', Q=False):
    simulation_gen_folder_2steps(output_dir)
    
    observe_tau, observe_w, observe_v, observe_p = simulation_load_data(simulation_data_path, bc=512)

    geomtime = simulation_geotime()

    zeta, zeta_tau = simulation_unknown_parameters()
    eta = eta_tau = 0.01

    data, data_ = simulation_train_data(
        geomtime, 
        lambda X, Y: primitive_equations_2d(X, Y, eta, zeta, eta_tau, zeta_tau, Q), 
        observe_tau, observe_w, observe_v, observe_p, num_domain=2000, 
        num_boundary=500)
    
    simulation_training_step(data_, iters, display, os.path.join(output_dir, 'Step1'), step=1, lrs=lrs, unknown_parameters=[zeta, zeta_tau], Q=Q)
    
    simulation_training_step(data, iters, display, os.path.join(output_dir, 'Step2'), step=2, lrs=lrs, unknown_parameters=[zeta, zeta_tau], variable_file_name=variable_file_name, Q=Q)
    
    
def simulation_train_no_physics(simulation_data_path='Data/Simulation/Train', output_dir='Output/Model/Simulation/NoPhysics', iters=1000, display=100, lrs=1e-4):
    simulation_gen_folder(output_dir)
    
    observe_tau, observe_w, observe_v, observe_p = simulation_load_data(simulation_data_path, bc=512)

    geomtime = simulation_geotime()

    # eta, zeta, eta_tau, zeta_tau = simulation_unknown_parameters()
    eta, zeta, eta_tau, zeta_tau = [0.01, 0.01, 0.01, 0.01]

    data, data_ = simulation_train_data(
        geomtime, 
        lambda X, Y: primitive_equations_2d(X, Y, eta, zeta, eta_tau, zeta_tau), 
        observe_tau, observe_w, observe_v, observe_p, num_domain=20000)
    
    simulation_training_step(data_, [iters], display, output_dir, step=1, lrs=[lrs])
        
    
def simulation_training_step(data, iters, display, output_dir, step=1, lrs=1e-4, unknown_parameters=None, variable_file_name='variable.dat', Q=False):
    model = simulation_model( 
        data, 
        n_layers=[4] * 4, 
        layer_width=32, 
        layer_sizes=[3] + [4], 
        activation="tanh", 
        kernel_initializer="Glorot normal",
        Q=Q)

    model.compile('adam', lr=lrs[step - 1], loss='MSE', external_trainable_variables=unknown_parameters)
    
    if step == 1:
        loss_history, train_state = model.train(
            iterations=iters[step - 1], 
            display_every=display, 
            model_save_path=os.path.join(output_dir, 'model'))
        saveplot_2(loss_history, train_state, issave=True, isplot=True, output_dir=os.path.join(output_dir))
        
    elif step == 2:
        checker, resampler, variable = callbacks(output_dir, save_period=10000, unknown_parameters=unknown_parameters, variable_file_name=variable_file_name)
        
        model.restore(os.path.join(f'{output_dir[:-1]}1', f'model-{iters[0]}.pt'))
    
        loss_history, train_state = model.train(
            iterations=iters[step - 1], 
            display_every=display, 
            callbacks=[checker, variable],
            model_save_path=os.path.join(output_dir, 'model'))
        saveplot_2(loss_history, train_state, issave=True, isplot=True, output_dir=os.path.join(output_dir))


def simulation_gen_new_x(density=101, density_t=101, save_path='Data/simulation_new_x.npy'):
    x_range = np.linspace(0, 1, density, endpoint=True)
    z_range = np.linspace(0, 1, density, endpoint=True)
    t_range = np.linspace(0, 1, density_t, endpoint=True)

    x, z, t = np.meshgrid(x_range, z_range, t_range)

    new_x = np.column_stack((x.ravel(), z.ravel(), t.ravel()))

    np.save(save_path, new_x)


def simulation_inference(checkpoint='Output/Model/Simulation/model.pt', new_x_path='Data/simulation_new_x.npy', save_path='Output/Prediction/Simulation/prediction.npy', para=0, Q=False):
    print('Model inference...')
    
    simulation_gen_folder(os.path.dirname(save_path))
    
    observe_tau, observe_w, observe_v, observe_p = simulation_load_data(simulation_data_path='Data/Simulation/Train')

    geomtime = simulation_geotime()

    zeta, zeta_tau = simulation_unknown_parameters()
    eta = eta_tau = 0.01

    data, data_ = simulation_train_data(
        geomtime, 
        lambda X, Y: primitive_equations_2d(X, Y, eta, zeta, eta_tau, zeta_tau), 
        observe_tau, observe_w, observe_v, observe_p, num_domain=20000)
    
    model = simulation_model( 
        data, 
        n_layers=[4] * 4, 
        layer_width=32, 
        layer_sizes=[3] + [4], 
        activation="tanh", 
        kernel_initializer="Glorot normal",
        Q=Q)
    model.compile('adam', lr=1e-4, loss='MSE', external_trainable_variables=[zeta, zeta_tau][0:para])
    model.restore(checkpoint, verbose=1)

    new_x = np.load(new_x_path)
    prediction = model.predict(new_x)
    prediction = prediction_normalization(prediction, new_x)
    
    np.save(save_path, prediction)
    
    print('Done.')
    
    
def prediction_normalization(prediction, new_x):
    t = new_x[:, 2]
    unique_t = np.unique(t)
    for t_ in unique_t:
        indices = t == t_
        prediction[indices] -= np.mean(prediction[indices], axis=0)
        
    return prediction


def ndarray_check(array):
    print(array, pd.DataFrame(array).describe())
    
    
def simulation_inference_steps(ITERS, checkpoint='Output/Model/Simulation/2Steps', new_x_path='Data/Simulation/Coordinates/simulation_new_x_plot.npy', save_path='Output/Prediction/Simulation/2Steps/prediction_plot', para=2, Q=False):
    simulation_gen_folder(os.path.dirname(save_path))
    
    simulation_inference(
        checkpoint=os.path.join(checkpoint, f'Step1/model-{ITERS[0]}.pt'), 
        new_x_path=new_x_path, 
        save_path=f'{save_path}_step1.npy',
        para=para, 
        Q=Q)

    # Prediction for plot (Step2)
    simulation_inference(
        checkpoint=os.path.join(checkpoint, f'Step2/model-{ITERS[1]}.pt'), 
        new_x_path=new_x_path, 
        save_path=f'{save_path}_step2.npy',
        para=para, 
        Q=Q)
    
    
def simulation_other_methods_inference(simulation_data_path='Data/Simulation/Train', new_x_path='Data/simulation_new_x.npy', save_path='Output/Prediction/Simulation/prediction.npy', model='gpr'):
    print('Other methods inference...')
    
    simulation_gen_folder(os.path.dirname(save_path))
    
    tau, w, v, p, x_z_t = simulation_read_data(simulation_data_path)
    new_x = np.load(new_x_path)
    
    infer = np.zeros((len(new_x), 4))
    if model == 'gpr':
        model_infer = gaussian_process_inference # Super slow
    elif model == 'kriging':
        model_infer = regression_kriging
    for i, y in tqdm(enumerate([tau, w, v, p]), total=4, desc=f"{model.upper()} Inference Progress"):
        infer[:, i] = model_infer(X_train=x_z_t, y_train=y, X_test=new_x)
        
    np.save(save_path, infer)
    
    print('Done!')


def simulation_plot_one(var, t_, t_i, x, z, Y_t, save_path, s=0.5, marker='s', vmins=[-1.1, -1, -1, -0.4, -1], vmaxs=[1.1, 1, 1, 0.4, 1], color_num=[13, 15, 17, 19, 21], cmaps=[plt.cm.plasma, plt.cm.RdYlGn, plt.cm.RdGy, plt.cm.inferno_r, plt.cm.cividis]):
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
        simulation_plot_data_domain()
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
    
    
def simulation_plot_data_domain(l=1):
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


def calc_y(x, constant=0.7, sign=1):
    return sign * abs(constant ** 3 - abs(x) ** 3) ** (1/3)


def data_domain_transform(x):
    return x / 2 + 0.5


def simulation_plot_add_rmse(rmse):
    annotation_text = f"RMSE={rmse:.3f}"
    annotation_position = (0.5, 0.1)
    plt.text(annotation_position[0], annotation_position[1], annotation_text, fontsize=12, color='black')


def simulation_gen_plot_folders(path, unique_t):
    for t_ in unique_t:
        path_ = os.path.join(path, f't={t_:.2f}')
        if not os.path.exists(path_):
            os.mkdir(path_)


def simulation_gen_folder(path):
    if not os.path.exists(path):
        os.mkdir(path)
            
            
def simulation_gen_folder_2steps(path):
    if not os.path.exists(path):
        os.mkdir(path)
        os.mkdir(os.path.join(path, 'Step1'))
        os.mkdir(os.path.join(path, 'Step2'))


def simulation_plot(X_path, Y_path, save_path='Output/Plot/Simulation/Prediction', vmins=[-1.1, -1, -1, -0.4, -3], vmaxs=[1.1, 1, 1, 0.4, 3], color_num=[13, 15, 17, 19, 21], cmaps=[plt.cm.plasma, plt.cm.RdYlGn, plt.cm.RdGy, plt.cm.inferno_r, plt.cm.bone]):
    print('Simulation plot...')
    
    simulation_gen_folder(save_path)
    
    X = np.load(X_path)
    Y = np.load(Y_path)

    t = X[:, 2]
    unique_t = np.unique(t)
    
    simulation_gen_plot_folders(save_path, unique_t)
    
    for t_i, t_ in tqdm(enumerate(unique_t), total=len(unique_t)):
        x, z, _ = unpack_x(X[X[:, 2] == t_])
        Y_t = Y[X[:, 2] == t_]

        for var in range(Y.shape[1]):
            simulation_plot_one(var, t_, t_i, x, z, Y_t, save_path, vmins=vmins, vmaxs=vmaxs, color_num=color_num, cmaps=cmaps)
            
    print('Done.')
    
    
def simulation_plot_res(X_path, pre_path, sol_path, save_path):
    pre = np.load(pre_path)
    sol = np.load(sol_path)
    res = np.abs(pre - sol)
    res_path = f'{pre_path[:-4]}_res.npy'
    np.save(res_path, res)
    
    simulation_plot(X_path, res_path, save_path, vmins = [0, 0, 0, 0], vmaxs = [0.5, 0.5, 0.5, 0.25], color_num = [20, 22, 24, 26, 28], cmaps=[plt.cm.gray_r] * 4)
    

def simulation_plot_2steps(X_path='Data/Simulation/Coordinates/simulation_new_x_plot.npy', Y_path='Output/Prediction/Simulation/2Steps/prediction_plot', save_path='Output/Plot/Simulation/Prediction/2Steps'):
    simulation_gen_folder_2steps(save_path)
    
    # Plot prediction (Step1)
    simulation_plot(
        X_path=X_path, 
        Y_path=f'{Y_path}_step1.npy', 
        save_path=os.path.join(save_path, 'Step1'))

    # Plot prediction (Step2)
    simulation_plot(
        X_path=X_path, 
        Y_path=f'{Y_path}_step2.npy', 
        save_path=os.path.join(save_path, 'Step2'))
    
    
def simulation_plot_parameters(model_dir='Output/Model/Simulation/2Steps2Para', save_dir='Output/Plot/Simulation/variables.png'):
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
    plt.legend()
    plt.savefig(save_dir, bbox_inches='tight')
    plt.close()
    


def find_files_with_prefix(directory, prefix):
    matches = []
    for root, dirs, files in os.walk(directory):
        for filename in fnmatch.filter(files, f"{prefix}*"):
            matches.append(os.path.join(root, filename))
    return matches


def read_dat_to_numpy(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
        data = []
        for line in lines:
            stripped_line = line.strip()
            if not stripped_line:
                continue
            
            parts = stripped_line.replace(" ", "").split("[")
            first_number = float(parts[0])
            numbers_str_inside_brackets = parts[1].replace("]", "").split(",")
            other_numbers = [float(num.strip()) for num in numbers_str_inside_brackets]    
            
            data.append([first_number] + other_numbers)

    numpy_array = np.array(data)
    return numpy_array


def simulation_plot_rmse(
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
        
    MODEL_NAMES = ['MDRF-Net', 'N-MDRF-Net', 'GPR', 'R-Kriging']
    COLORS = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
    LINESTYLES = ['-', '--', ':', '-.']
    
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
    X_path='Data/Simulation/Coordinates/simulation_new_x_plot.npy',
    sol_path='Data/Simulation/Solutions/simulation_solution_plot.npy', 
    mdrf_path='Output/Prediction/Simulation/2Steps2Para/prediction_plot_step2.npy',
    nmdrf_path='Output/Prediction/Simulation/NoPhysics/prediction_plot.npy',
    gpr_path='Output/Prediction/Simulation/GPR/prediction_plot.npy',
    kriging_path='Output/Prediction/Simulation/Kriging/prediction_plot.npy',
    save_path='Output/Plot/Simulation/simualtion_rmse.png'):
    
    simulation_plot_rmse(X_path, sol_path, mdrf_path, nmdrf_path, gpr_path, kriging_path, save_path, data_domain_bool=False)

    simulation_plot_rmse(X_path, sol_path, mdrf_path, nmdrf_path, gpr_path, kriging_path, save_path=save_path[:-4] + '_data_domain.png', data_domain_bool=True)
                    

# if __name__ == '__main__':
#     simulation_train(simulation_data_path='Data/Simulation')

    