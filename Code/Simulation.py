import numpy as np
import deepxde as dde
import torch
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.interpolate import griddata
from matplotlib.colors import ListedColormap


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
    return tau, w, v, p


def taylor_green_vortex(x, z, t, eta = 0.01, zeta = 0.01, eta_tau = 0.01, zeta_tau = 0.01, save=False, save_path='Data/simulation_solution.npy'):
    tau = np.sin(2 * np.pi * z) * np.exp(-4 * np.pi**2 * zeta_tau * t)
    w = np.cos(2 * np.pi * x) * np.sin(2 * np.pi * z) * np.exp(-4 * np.pi * np.pi * (eta + zeta) * t)
    v = -np.sin(2 * np.pi * x) * np.cos(2 * np.pi * z) * np.exp(-4 * np.pi * np.pi * (eta + zeta) * t)
    p = 0.25 * np.cos(4 * np.pi * x) * np.exp(-8 * np.pi**2 * (eta + zeta) * t) + np.cos(2 * np.pi * z) * np.exp(-4 * np.pi**2 * zeta_tau * t) / (2 * np.pi)

    if save:
        solution = np.column_stack((tau, w, v, p))
        np.save(save_path, solution)

    return tau, w, v, p


def gen_simulate_data(num=10000, noise_std=0.1, simulation_data_path = 'Data/Simulation/Train', eta = 0.01, zeta = 0.01, eta_tau = 0.01, zeta_tau = 0.01):
    x = np.random.rand(num)
    z = np.random.rand(num)
    t = np.random.rand(num)

    x_z_t = np.column_stack((x, z, t))
    np.save(f'{simulation_data_path}/x_z_t.npy', x_z_t)

    tau, w, v, p = taylor_green_vortex(x, z, t, eta, zeta, eta_tau, zeta_tau)

    tau = tau + np.random.normal(loc=0, scale=noise_std, size=num)
    w = w + np.random.normal(loc=0, scale=noise_std, size=num)
    v = v + np.random.normal(loc=0, scale=noise_std, size=num)
    p = p + np.random.normal(loc=0, scale=noise_std, size=num) 
    
    np.save(f'{simulation_data_path}/tau.npy', tau)
    np.save(f'{simulation_data_path}/w.npy', w)
    np.save(f'{simulation_data_path}/v.npy', v)
    np.save(f'{simulation_data_path}/p.npy', p)
    

def simulation_load_data(simulation_data_path = 'Data/Simulation/Train', bc=512):
    ob_coordinate_t = np.load(f'{simulation_data_path}/x_z_t.npy')
    
    ob_tau = np.load(f'{simulation_data_path}/tau.npy')
    ob_w = np.load(f'{simulation_data_path}/w.npy')
    ob_v = np.load(f'{simulation_data_path}/v.npy')
    ob_p = np.load(f'{simulation_data_path}/p.npy')

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
    space_domain = dde.geometry.Rectangle([0, 0], [1, 1])
    time_domain = dde.geometry.TimeDomain(0, 1)
    return dde.geometry.GeometryXTime(space_domain, time_domain)


def simulation_unknown_parameters():
    eta = dde.Variable(0.0)
    zeta = dde.Variable(0.0)
    eta_tau = dde.Variable(0.0)
    zeta_tau = dde.Variable(0.0)

    return eta, zeta, eta_tau, zeta_tau


def primitive_equations_2d(X, Y, eta, zeta, eta_tau, zeta_tau):
    x, z, t = unpack_x(X)
    tau, w, v, p = unpack_y(Y)

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
    dv_x_2 = dde.grad.hessian(Y, X, component=3, i=0, j=0)
    dv_z_2 = dde.grad.hessian(Y, X, component=3, i=1, j=1)

    Q = np.pi * torch.cos(2 * np.pi * x) * torch.sin(4 * np.pi * z) * torch.exp(-4 * np.pi**2 * (eta + zeta + zeta_tau) * t)

    equa_1 = dv_t + v * dv_x + w * dv_z - eta * dv_x_2 - zeta * dv_z_2 + dp_x
    equa_2 = dp_z + tau
    equa_3 = dv_x + dw_z
    equa_4 = dtau_t + v * dtau_x + w * dtau_z - eta_tau * dtau_x_2 - zeta_tau * dtau_z_2 - Q

    return equa_1, equa_2, equa_3, equa_4



def simulation_train_data(geomtime, primitive_equations, observe_tau, observe_w, observe_v, observe_p, num_domain=512):
    data = dde.data.PDE(
        geometry=geomtime,
        pde=primitive_equations,
        bcs=[observe_tau, observe_w, observe_v, observe_p],
        num_domain=num_domain,
    )
    data_ = dde.data.PDE(
        geometry=geomtime,
        pde=None,
        bcs=[observe_tau, observe_w, observe_v, observe_p],
        num_domain=0,
    )

    return data, data_


def simulation_model(data_, layer_sizes, activation, kernel_initializer):
    net = dde.nn.PFNN(layer_sizes, activation, kernel_initializer)
    model = dde.Model(data_, net)

    return model


def callbacks(eta, zeta, eta_tau, zeta_tau, output_dir):
    # variable = dde.callbacks.VariableValue([eta, zeta, eta_tau, zeta_tau], period=100, filename=os.path.join(output_dir, 'variable.dat'))
    resampler = dde.callbacks.PDEPointResampler(period=100, pde_points=True, bc_points=True)
    resampler_ = dde.callbacks.PDEPointResampler(period=100, pde_points=True, bc_points=True)

    return resampler, resampler_


def simulation_train(simulation_data_path='Data/Simulation/Train', output_dir='Output/Model/Simulation'):
    observe_tau, observe_w, observe_v, observe_p = simulation_load_data(simulation_data_path, bc=512)

    geomtime = simulation_geotime()

    # eta, zeta, eta_tau, zeta_tau = simulation_unknown_parameters()
    eta, zeta, eta_tau, zeta_tau = [1, 1, 1, 1]

    data, data_ = simulation_train_data(
        geomtime, 
        lambda X, Y: primitive_equations_2d(X, Y, eta, zeta, eta_tau, zeta_tau), 
        observe_tau, observe_w, observe_v, observe_p, num_domain=10000)
    
    model = simulation_model(
        data_, 
        layer_sizes = [3, [32, 32, 32, 32], [32, 32, 32, 32], 4], 
        activation = 'tanh', 
        kernel_initializer = 'Glorot normal')
    
    resampler, resampler_ = callbacks(eta, zeta, eta_tau, zeta_tau, output_dir)

    model.compile('adam', lr=1e-4, loss='MSE')
    loss_history, train_state = model.train(
        iterations=10000, 
        # callbacks=[resampler_],
        display_every=100, 
        model_save_path=os.path.join(output_dir, 'Step1', 'model'))
    dde.saveplot(loss_history, train_state, issave=True, isplot=True, output_dir=os.path.join(output_dir, 'Step1'))

    model.change_data(data)

    model.compile('adam', lr=1e-4, loss='MSE')
    loss_history, train_state = model.train(
        iterations=20000, 
        # callbacks=[resampler],
        display_every=100, 
        model_save_path=os.path.join(output_dir, 'Step2', 'model'))
    dde.saveplot(loss_history, train_state, issave=True, isplot=True, output_dir=os.path.join(output_dir, 'Step2'))


def simulation_gen_new_x(density=101, density_t=101, save_path='Data/simulation_new_x.npy'):
    x_range = np.linspace(0, 1, density, endpoint=True)
    z_range = np.linspace(0, 1, density, endpoint=True)
    t_range = np.linspace(0, 1, density_t, endpoint=True)

    x, z, t = np.meshgrid(x_range, z_range, t_range)

    new_x = np.column_stack((x.ravel(), z.ravel(), t.ravel()))

    np.save(save_path, new_x)


def simulation_inference(checkpoint='Output/Model/Simulation/model.pt', new_x_path='Data/simulation_new_x.npy', save_path='Output/Prediction/Simulation/predcition.npy'):
    observe_tau, observe_w, observe_v, observe_p = simulation_load_data(simulation_data_path='Data/Simulation/Train')

    geomtime = simulation_geotime()

    eta, zeta, eta_tau, zeta_tau = simulation_unknown_parameters()

    data, data_ = simulation_train_data(
        geomtime, 
        lambda X, Y: primitive_equations_2d(X, Y, eta, zeta, eta_tau, zeta_tau), 
        observe_tau, observe_w, observe_v, observe_p, num_domain=10000)
    
    model = simulation_model(
        data, 
        layer_sizes = [3, [32, 32, 32, 32], [32, 32, 32, 32], 4], 
        activation = 'tanh', 
        kernel_initializer = 'Glorot normal')
    model.compile('adam', lr=1e-4, loss='MSE')
    model.restore(checkpoint, verbose=1)

    new_x = np.load(new_x_path)
    prediction = model.predict(new_x)
    np.save(save_path, prediction)


def simulation_plot_one(var, t_, x, z, Y_t, save_path):
    CMAPS = []
    TITLES = ['Temperature', 'Vertical Velocity', 'Horizontal Velocity', 'Pressure']
    SAVE_NAMES = ['tau', 'w', 'v', 'p']
    VMINS = [-1, -1, -1, -0.4]
    VMAXS = [1, 1, 1, 0.4]

    for cmap in [plt.cm.plasma, plt.cm.RdYlGn, plt.cm.RdGy, plt.cm.inferno_r]:
        colors = cmap(np.linspace(0, 1, 13, endpoint=True))
        CMAPS.append(ListedColormap(colors))

    plt.figure(figsize=(4, 3))
    sc = plt.scatter(x, z, c=Y_t[:, var], cmap=CMAPS[var], alpha=1, s=0.5, marker='s', vmin=VMINS[var], vmax=VMAXS[var])
    plt.colorbar(sc)
    plt.title(TITLES[var])
    plt.xlabel('x')
    plt.ylabel('z')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.savefig(
        os.path.join(save_path, f't={t_:.2f}', f'{SAVE_NAMES[var]}.png'), 
        dpi=300, 
        bbox_inches='tight')


def simulation_plot(X_path, Y_path, save_path='Output/Plot/Simulation/Prediction'):
    X = np.load(X_path)
    Y = np.load(Y_path)

    t = X[:, 2]

    for t_ in tqdm(np.unique(t)):
        path = os.path.join(save_path, f't={t_:.2f}')
        if not os.path.exists(path):
            os.mkdir(path)

        x, z, _ = unpack_x(X[X[:, 2] == t_])
        Y_t = Y[X[:, 2] == t_]

        for var in range(4):
            simulation_plot_one(var, t_, x, z, Y_t, save_path)
            

if __name__ == '__main__':
    simulation_train(simulation_data_path='Data/Simulation')

    