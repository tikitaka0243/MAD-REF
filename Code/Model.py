import numpy as np
import deepxde as dde
from sklearn.metrics import root_mean_squared_error
import torch
import torch.nn as nn
import math
import os
from tqdm import tqdm
import sys
import pandas as pd

file_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(file_path)

from Code.Plot import RealDataPlot
from utils import BETA_SIGMA, BETA_TAU, RHO_0, SIGMA_MEAN, SIGMA_STD, TAU_MEAN, TAU_STD, TIME_LIST, VAR_COLORS, VAR_NAMES, VAR_UNITS, gen_folder, gen_folder_2steps, ndarray_check, prediction_normalization, proportional_sampling, replace_group_name, unpack_x, unpack_x_sim, unpack_y, unpack_y_sim
from Code.Backend import MDRF_Net, Model2, MDRF_Net_SIMULATION, saveplot_2
from DataProcess import RealData, SimulationData




def z_boundary(x, on_boundary):
    return on_boundary and (np.isclose(x[1], 0) or np.isclose(x[1], 1))


def feature_transform_simulation(inputs):
    # Periodic BC
    P = 1
    w = 2 * np.pi / P
    x, z, t = w * inputs[:, 0:1], w * inputs[:, 1:2], inputs[:, 2:3]
    
    return torch.cat(
        (
            torch.cos(x),
            torch.sin(x),
            torch.cos(z),
            torch.sin(z),
            t,
        ),
        dim=1
    )

def output_transform_simulation(inputs, outputs):
    x, z, t = unpack_x_sim(inputs)
    tau, w, v, p = unpack_y_sim(outputs)

    # Zero Dirichlet BC
    w_transform = torch.sin(torch.pi * z) * w
    # v_transform = (1 - torch.exp(0 - x)) * (1 - torch.exp(x - 1)) * v

    return torch.cat((tau, w_transform, v, p), dim=1)



class SimulationModel():
    def __init__(self, task_name, dynamic_loss_weight=False, alpha=0.9, Q=False, hard_constraint=True, meta_learning=True, output_dir=None, simulation=True):
        self.dynamic_loss_weight = dynamic_loss_weight
        self.save_period = 1000
        self.alpha = alpha
        self.task_name = task_name
        self.Q = Q
        self.hard_constraint = hard_constraint
        self.layer_sizes = [5] + [4] if self.hard_constraint else [3] + [4]
        self.meta_learning = meta_learning
        self.output_dir = output_dir
        self.simulation = simulation
        self.frozen = False

    def primitive_equations_2d(self, X, Y, eta, zeta, eta_tau, zeta_tau, Q=False):
        x, z, t = unpack_x_sim(X)
        if Q:
            tau, w, v, p, q = unpack_y_sim(Y)
        else:
            tau, w, v, p = unpack_y_sim(Y)
        
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

        if self.hard_constraint:
            w = self.hard_boundary_transform_infer(w, z)
            dw_z = torch.pi * torch.cos(torch.pi * z) * w + torch.sin(torch.pi * z) * dw_z

        if Q:
            Q = q
        else:
            Q = np.pi * torch.cos(2 * np.pi * x) * torch.sin(4 * np.pi * z) * torch.exp(-4 * np.pi**2 * (eta + zeta + zeta_tau) * t)

        equa_1 = dv_t + v * dv_x + w * dv_z - eta * dv_x_2 - zeta * dv_z_2 + dp_x
        equa_2 = dp_z + tau

        equa_3 = dv_x + dw_z
        # equa_3 = dv_x - (-2 * torch.pi * torch.cos(2 * torch.pi * x) * torch.cos(2 * torch.pi * z) * torch.exp(-4 * torch.pi**2 * (eta + zeta) * t))
        # equa_3_1 = dw_z - (torch.cos(2 * torch.pi * x) * 2 * torch.pi * torch.cos(2 * torch.pi * z) * torch.exp(-4 * torch.pi**2 * (eta + zeta) * t))
        # torch.sin(torch.pi * z) * w

        equa_4 = dtau_t + v * dtau_x + w * dtau_z - eta_tau * dtau_x_2 - zeta_tau * dtau_z_2 - Q

        return equa_1, equa_2, equa_3, equa_4

    def simulation_load_data(self, bc=None, folder='Train'):

        ob_tau, ob_w, ob_v, ob_p, x_z_t = SimulationData.simulation_read_data(self.simulation_data_path, folder)

        ob_tau = ob_tau.reshape(-1, 1)
        ob_w = ob_w.reshape(-1, 1)
        ob_v = ob_v.reshape(-1, 1)
        ob_p = ob_p.reshape(-1, 1)

        if self.hard_constraint:
            ob_z = x_z_t[:, 1:2]
            ob_w = self.hard_boundary_transform_obs(ob_w, ob_z)

        observe_tau = dde.icbc.PointSetBC(x_z_t, ob_tau, component=0, batch_size=bc)
        observe_w = dde.icbc.PointSetBC(x_z_t, ob_w, component=1, batch_size=bc)
        observe_v = dde.icbc.PointSetBC(x_z_t, ob_v, component=2, batch_size=bc)
        observe_p = dde.icbc.PointSetBC(x_z_t, ob_p, component=3, batch_size=bc)

        return observe_tau, observe_w, observe_v, observe_p
    
    def hard_boundary_transform_obs(self, ob_w, ob_z):
        return ob_w / np.sin(np.pi * ob_z)
    
    def hard_boundary_transform_infer(self, pre_w, pre_z):
        if isinstance(pre_w, np.ndarray):
            return pre_w * np.sin(np.pi * pre_z)
        else:
            return pre_w * torch.sin(torch.pi * pre_z)

    def simulation_geotime(self):
        space_domain = dde.geometry.Rectangle([0.0, 0.0], [1.0, 1.0])
        time_domain = dde.geometry.TimeDomain(0.0, 1.0)
        return dde.geometry.GeometryXTime(space_domain, time_domain)

    def simulation_unknown_parameters(self):
        zeta = dde.Variable(0.0)
        zeta_tau = dde.Variable(0.0)

        return zeta, zeta_tau

    def simulation_para_transform(self, eta, zeta, eta_tau, zeta_tau):
        output = []
        for para in [eta, zeta, eta_tau, zeta_tau]:
            if isinstance(para, torch.Tensor):
                output.append(torch.sigmoid(para) / 10)
            else:
                output.append(para)
                
        return output

    def simulation_train_data(self, geomtime, primitive_equations, num_domain=512, num_boundary=512):
        
        if self.hard_constraint:
            bcs = self.simulation_load_data(bc=500, folder='Train')[:3]
            bcs_val = self.simulation_load_data(bc=None, folder='Validation')[:3]
        else:
            bcs = self.boundary_conditions(geomtime) + self.simulation_load_data(bc=500, folder='Train')[:3]
            bcs_val = self.boundary_conditions(geomtime) + self.simulation_load_data(bc=None, folder='Validation')[:3]

        data = dde.data.PDE(
            geometry=geomtime,
            pde=primitive_equations,
            bcs=bcs,
            num_domain=num_domain,
            num_boundary=num_boundary,
        )

        data_val = dde.data.PDE(
            geometry=geomtime,
            pde=primitive_equations,
            bcs=bcs_val,
            num_domain=num_domain,
            num_boundary=num_boundary,
        )

        data_ = dde.data.PDE(
            geometry=geomtime,
            pde=None,
            bcs=self.simulation_load_data(bc=500, folder='Train')[:3],
            num_domain=0,
        )

        return data, data_, data_val


    def boundary_conditions(self, geomtime):
        bc_tau_x = dde.icbc.PeriodicBC(geomtime, 0, lambda _, on_boundary: on_boundary, derivative_order=0, component=0)
        bc_tau_z = dde.icbc.PeriodicBC(geomtime, 1, lambda _, on_boundary: on_boundary, derivative_order=0, component=0)
        bc_w_x = dde.icbc.PeriodicBC(geomtime, 0, lambda _, on_boundary: on_boundary, derivative_order=0, component=1)
        bc_w_z = dde.icbc.PeriodicBC(geomtime, 1, lambda _, on_boundary: on_boundary, derivative_order=0, component=1)
        bc_w_z_dirichlet = dde.icbc.DirichletBC(geomtime, lambda x: 0, z_boundary, component=1)
        bc_v_x = dde.icbc.PeriodicBC(geomtime, 0, lambda _, on_boundary: on_boundary, derivative_order=0, component=2)
        bc_v_z = dde.icbc.PeriodicBC(geomtime, 1, lambda _, on_boundary: on_boundary, derivative_order=0, component=2)
        bc_p_x = dde.icbc.PeriodicBC(geomtime, 0, lambda _, on_boundary: on_boundary, derivative_order=0, component=3)
        bc_p_z = dde.icbc.PeriodicBC(geomtime, 1, lambda _, on_boundary: on_boundary, derivative_order=0, component=3)
        
        return bc_v_x, bc_v_z, bc_w_x, bc_w_z, bc_w_z_dirichlet, bc_p_x, bc_p_z, bc_tau_x, bc_tau_z


    def simulation_model(self, data, data_val, n_layers=[4] * 4, layer_width=32, layer_sizes=[5] + [4], activation="tanh", kernel_initializer="Glorot normal"):
        net = MDRF_Net_SIMULATION(n_layers=n_layers, layer_width=layer_width, layer_sizes=layer_sizes, activation=activation, kernel_initializer=kernel_initializer, Q=self.Q)

        if self.hard_constraint:
            net.apply_feature_transform(feature_transform_simulation)
            # net.apply_output_transform(output_transform_simulation)

        model = Model2(data, data_val, net, alpha=self.alpha, task_name=self.task_name, meta_learning=self.meta_learning, output_dir=self.output_dir)

        return model


    def callbacks(self, save_period=10000, unknown_parameters=None, variable_file_name='variable.dat'):
        gen_folder(os.path.join(self.output_dir, 'Step2', 'Checkpoint'))

        checker = dde.callbacks.ModelCheckpoint(os.path.join(self.output_dir, 'Step2', 'Checkpoint', 'model'), save_better_only=False, period=save_period)
        resampler = dde.callbacks.PDEPointResampler(period=100, pde_points=True, bc_points=True)
        variable = dde.callbacks.VariableValue(unknown_parameters, period=100, filename=os.path.join(self.output_dir, 'Step2',variable_file_name), precision=6)

        return checker, resampler, variable


    def simulation_train(self, simulation_data_path='Data/Simulation', iters=[10000, 20000], display=100, save_period=1000, lrs=[1e-4, 1e-5], variable_file_name='variable.dat', loss_weight=None, frozen=False):
        self.save_period = save_period
        self.simulation_data_path = simulation_data_path
        self.loss_weights_ = loss_weight
        self.frozen = frozen

        gen_folder_2steps(self.output_dir)

        geomtime = self.simulation_geotime()

        zeta, zeta_tau = self.simulation_unknown_parameters()
        eta = eta_tau = 0.01

        data, data_, data_val = self.simulation_train_data(
            geomtime, 
            lambda X, Y: self.primitive_equations_2d(X, Y, eta, zeta, eta_tau, zeta_tau, self.Q),
            num_domain=1500, 
            num_boundary=500)
        
        self.simulation_training_step(data_, iters, display, step=1, lrs=lrs, unknown_parameters=[zeta, zeta_tau])
        
        self.simulation_training_step(data, iters, display, step=2, lrs=lrs, unknown_parameters=[zeta, zeta_tau], variable_file_name=variable_file_name, data_val=data_val)
    
    
    # def simulation_train_no_physics(self, simulation_data_path='Data/Simulation/Train', output_dir='Output/Model/Simulation/NoPhysics', iters=1000, display=100, lrs=1e-4):
    #     gen_folder(output_dir)

    #     geomtime = self.simulation_geotime()

    #     # eta, zeta, eta_tau, zeta_tau = simulation_unknown_parameters()
    #     eta, zeta, eta_tau, zeta_tau = [0.01, 0.01, 0.01, 0.01]

    #     data, data_, data_val = self.simulation_train_data(
    #         geomtime, 
    #         lambda X, Y: primitive_equations_2d(X, Y, eta, zeta, eta_tau, zeta_tau), 
    #         num_domain=20000)
        
    #     self.simulation_training_step(data_, [iters], display, output_dir, step=1, lrs=[lrs])
        
    
    def simulation_training_step(self, data, iters, display, step=1, lrs=1e-4, unknown_parameters=None, variable_file_name='variable.dat', data_val=None):
        model = self.simulation_model( 
            data, 
            data_val,
            n_layers=[4] * 4, 
            layer_width=32, 
            layer_sizes=self.layer_sizes, 
            kernel_initializer="Glorot normal")


        checker, resampler, variable = self.callbacks(save_period=self.save_period, unknown_parameters=unknown_parameters, variable_file_name=variable_file_name)
        
        if step == 1:
            model.compile('adam', 
                          lr=lrs[step - 1], 
                          loss='MSE', 
                          decay=['cosine', sum(iters), 0],
                          external_trainable_variables=unknown_parameters, 
                          dynamic_loss_weight=self.dynamic_loss_weight,
                          train_step_num=1)

            gen_folder(os.path.join(self.output_dir, 'Step1', 'Checkpoint'))

            loss_history, train_state = model.train(
                iterations=iters[step - 1], 
                display_every=display, 
                callbacks=[resampler],
                model_save_path=os.path.join(self.output_dir, 'Step1', 'Checkpoint', 'model'))
            saveplot_2(loss_history, train_state, issave=True, isplot=False, output_dir=os.path.join(self.output_dir, 'Step1'))
            
        elif step == 2:
            model.compile('adam', 
                          lr=lrs[step - 1], 
                          loss='MSE', 
                          decay=['cosine', sum(iters), 0], 
                          external_trainable_variables=unknown_parameters, 
                          dynamic_loss_weight=self.dynamic_loss_weight,
                          train_step_num=2,
                          loss_weights=self.loss_weights_,
                          frozen=self.frozen) 
            
            model.restore(os.path.join(self.output_dir, 'Step1', 'Checkpoint', f'model-{iters[0]}.pt'))
        
            loss_history, train_state = model.train(
                iterations=iters[step - 1], 
                display_every=display, 
                callbacks=[checker, resampler, variable],
                model_save_path=os.path.join(self.output_dir, 'Step2', 'Checkpoint', 'model'))
            saveplot_2(loss_history, train_state, issave=True, isplot=False, output_dir=os.path.join(self.output_dir, 'Step1'))


    def simulation_inference(self, checkpoint='Output/Model/Simulation/model.pt', new_x_path='Data/simulation_new_x.npy', save_path='Output/Prediction/Simulation/prediction.npy', para=2, return_results=False):
        print('Model inference...')
        
        gen_folder(os.path.dirname(save_path))

        geomtime = self.simulation_geotime()

        zeta, zeta_tau = self.simulation_unknown_parameters()
        eta = eta_tau = 0.01
        
        model = self.simulation_model( 
            None, 
            None,
            n_layers=[4] * 4, 
            layer_width=32, 
            layer_sizes=self.layer_sizes, 
            activation="tanh", 
            kernel_initializer="Glorot normal")
        model.compile('adam', lr=1e-4, loss='MSE', external_trainable_variables=[zeta, zeta_tau][0:para])
        model.restore(checkpoint, verbose=1)

        new_x = np.load(new_x_path)
        prediction = model.predict(new_x)
        prediction = prediction_normalization(prediction, new_x)

        if self.hard_constraint:
            pre_z = new_x[:, 1:2]
            pre_w = prediction[:, 1:2]
            pre_w = self.hard_boundary_transform_infer(pre_w, pre_z)
            prediction[:, 1:2] = pre_w

        # print('Done.')
        
        if return_results:
            return prediction

        np.save(save_path, prediction)   
        

    def simulation_inference_steps(self, model_num_1, model_num_2, checkpoint='Output/Model/Simulation/2Steps', new_x_path='Data/Simulation/Coordinates/simulation_new_x_plot.npy', save_path='Output/Prediction/Simulation/2Steps/prediction_plot', para=2):
        gen_folder(os.path.dirname(save_path))
        
        self.simulation_inference(
            checkpoint=os.path.join(checkpoint, f'Step1/Checkpoint/model-{model_num_1}.pt'), 
            new_x_path=new_x_path, 
            save_path=f'{save_path}_step1.npy',
            para=para)

        # Prediction for plot (Step2)
        self.simulation_inference(
            checkpoint=os.path.join(checkpoint, f'Step2/Checkpoint/model-{model_num_2}.pt'), 
            new_x_path=new_x_path, 
            save_path=f'{save_path}_step2.npy',
            para=para)

    def rmse_trend(self):
        # Accuracy trend
        all_rmse_results = []
        for i in tqdm(range(1, 101)):
            pre = self.simulation_inference(
                checkpoint=f'Output/Model/Simulation/{self.task_name}/Step2/Checkpoint/model-{int(1000 * i)}.pt', 
                new_x_path='Data/Simulation/Coordinates/simulation_new_x.npy', 
                para=2, 
                return_results=True)

            gt = np.load('Data/Simulation/Solutions/simulation_solution.npy')

            rmse_list = []
            for j in range(pre.shape[1]):
                rmse = root_mean_squared_error(gt[:, j], pre[:, j])
                rmse_list.append(rmse)

            all_rmse_results.append(rmse_list)

            rmse_df = pd.DataFrame(all_rmse_results, columns=['tau', 'w', 'v', 'p'])

            rmse_df.to_csv(f'Output/Prediction/Simulation/RMSETrends/rmse_results_{self.task_name}.csv', index=False)


def primitive_equations(X, Y, D, beta_T, beta_S):
    r, theta, phi, t = unpack_x(X)
    tau, sigma, w, v_theta, v_phi, p = unpack_y(Y)

    theta = theta + 4 / 9 * torch.pi

    tau += D.T_mean / D.T_0
    sigma += D.S_mean / D.S_0
    
    # eta, zeta, eta_tau, zeta_tau = simulation_para_transform(eta, zeta, eta_tau, zeta_tau)

    dtau_z = dde.grad.jacobian(Y, X, i=0, j=0)
    dtau_theta = dde.grad.jacobian(Y, X, i=0, j=1)
    dtau_phi = dde.grad.jacobian(Y, X, i=0, j=2)
    dtau_t = dde.grad.jacobian(Y, X, i=0, j=3)
    dsigma_z = dde.grad.jacobian(Y, X, i=1, j=0)
    dsigma_theta = dde.grad.jacobian(Y, X, i=1, j=1)
    dsigma_phi = dde.grad.jacobian(Y, X, i=1, j=2)
    dsigma_t = dde.grad.jacobian(Y, X, i=1, j=3)
    dw_z = dde.grad.jacobian(Y, X, i=2, j=0)
    # dw_theta = dde.grad.jacobian(Y, X, i=2, j=1)
    # dw_phi = dde.grad.jacobian(Y, X, i=2, j=2)
    # dw_t = dde.grad.jacobian(Y, X, i=2, j=3)
    dvtheta_z = dde.grad.jacobian(Y, X, i=3, j=0)
    dvtheta_theta = dde.grad.jacobian(Y, X, i=3, j=1)
    dvtheta_phi = dde.grad.jacobian(Y, X, i=3, j=2)
    dvtheta_t = dde.grad.jacobian(Y, X, i=3, j=3)
    dvphi_z = dde.grad.jacobian(Y, X, i=4, j=0)
    dvphi_theta = dde.grad.jacobian(Y, X, i=4, j=1)
    dvphi_phi = dde.grad.jacobian(Y, X, i=4, j=2)
    dvphi_t = dde.grad.jacobian(Y, X, i=4, j=3)
    dpi_z = dde.grad.jacobian(Y, X, i=5, j=0)
    dpi_theta = dde.grad.jacobian(Y, X, i=5, j=1)
    dpi_phi = dde.grad.jacobian(Y, X, i=5, j=2)
    # dpi_t = dde.grad.jacobian(Y, X, i=5, j=3)

    dtau_z_2 = dde.grad.hessian(Y, X, component=0, i=0, j=0)
    dtau_theta_2 = dde.grad.hessian(Y, X, component=0, i=1, j=1)
    dtau_phi_2 = dde.grad.hessian(Y, X, component=0, i=2, j=2)
    # dtau_t_2 = dde.grad.hessian(Y, X, component=0, i=3, j=3)
    dsigma_z_2 = dde.grad.hessian(Y, X, component=1, i=0, j=0)
    dsigma_theta_2 = dde.grad.hessian(Y, X, component=1, i=1, j=1)
    dsigma_phi_2 = dde.grad.hessian(Y, X, component=1, i=2, j=2)
    # dsigma_t_2 = dde.grad.hessian(Y, X, component=1, i=3, j=3)
    # dw_z_2 = dde.grad.hessian(Y, X, component=2, i=0, j=0)
    # dw_theta_2 = dde.grad.hessian(Y, X, component=2, i=1, j=1)
    # dw_phi_2 = dde.grad.hessian(Y, X, component=2, i=2, j=2)
    # dw_t_2 = dde.grad.hessian(Y, X, component=2, i=3, j=3)
    dvtheta_z_2 = dde.grad.hessian(Y, X, component=3, i=0, j=0)
    dvtheta_theta_2 = dde.grad.hessian(Y, X, component=3, i=1, j=1)
    dvtheta_phi_2 = dde.grad.hessian(Y, X, component=3, i=2, j=2)
    # dvtheta_t_2 = dde.grad.hessian(Y, X, component=3, i=3, j=3)
    dvphi_z_2 = dde.grad.hessian(Y, X, component=4, i=0, j=0)
    dvphi_theta_2 = dde.grad.hessian(Y, X, component=4, i=1, j=1)
    dvphi_phi_2 = dde.grad.hessian(Y, X, component=4, i=2, j=2)
    # dvphi_t_2 = dde.grad.hessian(Y, X, component=4, i=3, j=3)
    # dpi_z_2 = dde.grad.hessian(Y, X, component=5, i=0, j=0)
    # dpi_theta_2 = dde.grad.hessian(Y, X, component=5, i=1, j=1)
    # dpi_phi_2 = dde.grad.hessian(Y, X, component=5, i=2, j=2)
    # dpi_t_2 = dde.grad.hessian(Y, X, component=5, i=3, j=3)
    # Unnecessary automatic differential calculation will increase the training time


    equa_1 = (
        dvtheta_t
        + (v_theta / 1 * dvtheta_theta
            + v_phi / (1 * torch.sin(theta) + 1e-8) * dvtheta_phi
            - v_phi ** 2 / 1 * (torch.tan(theta) + 1e-8) ** -1)
        + w * dvtheta_z
        + 1 / 1 * dpi_theta
        + 1 / D.R_o * (2 * torch.cos(theta)) * (-v_phi)
        - D.Re1_inv * (1 / (1 ** 2 * torch.sin(theta) + 1e-8)
                * (torch.cos(theta) * dvtheta_theta
                + torch.sin(theta) * dvtheta_theta_2
                + 1 / (torch.sin(theta) + 1e-8) * dvtheta_phi_2)
                - 2 * torch.cos(theta) / (1 ** 2 * torch.sin(theta) ** 2 + 1e-8) * dvphi_phi
                - v_theta / (1 ** 2 * torch.sin(theta) ** 2 + 1e-8))
        - D.Re2_inv * dvtheta_z_2
    )
    
    equa_2 = (
        dvphi_t
        + (v_theta / 1 * dvphi_theta
            + v_phi / (1 * torch.sin(theta) + 1e-8) * dvphi_phi
            + v_theta * v_phi / 1 * (torch.tan(theta) + 1e-8) ** -1)
        + w * dvphi_z
        + 1 / (1 * torch.sin(theta) + 1e-8) * dpi_phi
        + 1 / D.R_o * (2 * torch.cos(theta)) * (v_theta)
        - D.Re1_inv * (1 / (1 ** 2 * torch.sin(theta) + 1e-8)
                * (torch.cos(theta) * dvphi_theta
                + torch.sin(theta) * dvphi_theta_2
                + 1 / (torch.sin(theta) + 1e-8) * dvphi_phi_2)
                + 2 * torch.cos(theta) / (1 ** 2 * torch.sin(theta) ** 2 + 1e-8) * dvtheta_phi
                - v_phi / (1 ** 2 * torch.sin(theta) ** 2 + 1e-8))
        - D.Re2_inv * dvphi_z_2
    )
    
    beta_T_bar = beta_T * D.T_0
    beta_S_bar = beta_S * D.S_0
    rho = 1 - beta_T_bar * (tau - 1) + beta_S_bar * (sigma - 1)
    equa_3 = dpi_z + rho * D.b_bar
    
    equa_4 = 1 / (1 * torch.sin(theta) + 1e-8) * (
        dvtheta_theta * torch.sin(theta) 
        + v_theta * torch.cos(theta) 
        + dvphi_phi
    ) + dw_z
    
    equa_5 = dtau_t + v_theta / 1 * dtau_theta + v_phi / (1 * torch.sin(theta) * dtau_phi + 1e-8) + w * dtau_z - D.Rt1_inv * 1 / (1 ** 2 * torch.sin(theta) + 1e-8) * (
        torch.cos(theta) * dtau_theta 
        + torch.sin(theta) * dtau_theta_2 
        + 1 / (torch.sin(theta) + 1e-8) * dtau_phi_2
    ) - D.Rt2_inv * dtau_z_2

    equa_6 = dsigma_t + v_theta / 1 * dsigma_theta + v_phi / (1 * torch.sin(theta) * dsigma_phi + 1e-8) + w * dsigma_z - D.Rs1_inv * 1 / (1 ** 2 * torch.sin(theta) + 1e-8) * (
        torch.cos(theta) * dsigma_theta 
        + torch.sin(theta) * dsigma_theta_2 
        + 1 / (torch.sin(theta) + 1e-8) * dsigma_phi_2
    ) - D.Rs2_inv * dsigma_z_2


    return equa_1, equa_2, equa_3, equa_4, equa_5, equa_6


    
class RealDataModel:
    def __init__(self, task_name, dataloader, plotter, hard_constraint=True, meta_learning=True, output_dir=None, n_layers=[4] * 2 + [6] * 4, layer_width=256, sub_learner=None, activation_func='sin'):
        self.dataloader = dataloader
        self.plotter = plotter

        self.beta_T = None
        self.beta_S = None

        self.save_period = 10000
        self.task_name = task_name
        self.hard_constraint = hard_constraint
        self.layer_sizes = [4] + [6]
        self.meta_learning = meta_learning
        self.sub_learner = sub_learner
        self.output_dir = os.path.join(output_dir, task_name)

        gen_folder_2steps(self.output_dir)

        self.step_1_plot_pre_path = f'Output/Prediction/RealData/{self.task_name}/prediction_plot_step1.npy'
        self.step_2_plot_pre_path = f'Output/Prediction/RealData/{self.task_name}/prediction_plot_step2.npy'

        self.n_layers = n_layers
        self.layer_width = layer_width
        self.train_var = 'all'
        self.activation_func = activation_func

    def set_restore(self, restore_main, restore_sub):
        self.restore_main = restore_main
        self.restore_sub = restore_sub

    def train(self, iters=[10000, 20000], display=100, save_period=10000, lrs=[1e-4, 1e-5], variable_file_name='variable.dat', num_domain=1500, num_boundary=500, batch_size=500, new_x_path='Data/RealData/new_x_plot.npy', loss_weight=[1, 1, 100, 10, 10], restore_1=None, restore_2=None, train_var='all', reinitialize=False, external_source=False, skip_step_1=False):
    # restore_1: ['TaskName', 'Step1', 1500000]

        self.loss_weight = loss_weight
        self.save_period = save_period
        self.num_domain = num_domain
        self.num_boundary = num_boundary
        self.batch_size = batch_size
        self.restore_1 = restore_1
        self.restore_2 = restore_2
        self.train_var = train_var
        self.reinitialize = reinitialize
        self.external_source = external_source

        data, data_, data_val = self.train_data()
        
        if not skip_step_1:
            self.training_step(data_, 
                            iters, 
                            display, 
                            step=1, 
                            lrs=lrs, 
                            variable_file_name=variable_file_name)
            self.inference_and_plot(new_x_path, 
                                    save_path=self.step_1_plot_pre_path, step_num=1, 
                                    iter_num=iters[0])
        
        self.training_step(data, 
                           iters, 
                           display, 
                           step=2, 
                           lrs=lrs, 
                           variable_file_name=variable_file_name, 
                           data_val=data_val)
        self.inference_and_plot(new_x_path, 
                                save_path=self.step_2_plot_pre_path, step_num=2, 
                                iter_num=iters[1])
        

    def training_step(self, data, iters, display, step=1, lrs=1e-4, variable_file_name='variable.dat', data_val=None):
        model = self.realdata_model( 
            data, 
            data_val,
            kernel_initializer="Glorot normal")

        checker, resampler, variable = self.callbacks(save_period=self.save_period, 
        variable_file_name=variable_file_name, train_step=step)
        
        if step == 1:
            model.compile('adam', 
                          lr=lrs[step - 1], 
                          loss='MSE', 
                          decay=['cosine', sum(iters), 0], 
                          external_trainable_variables=[self.beta_T, self.beta_S], 
                          dynamic_loss_weight=False,
                          train_step_num=1,
                          loss_weights=self.loss_weight)
            
            if self.restore_1 is not None:
                retore_path = f'Output/Model/RealData/{self.restore_1[0]}/{self.restore_1[1]}/Checkpoint/model-{self.restore_1[2]}.pt'
                model.restore(retore_path)

                if self.train_var != 'all' and self.reinitialize:
                    self.reinitialize_param(model.net, reinitialize_group=self.train_var)
            

            gen_folder(os.path.join(self.output_dir, 'Step1', 'Checkpoint'))

            loss_history, train_state = model.train(
                iterations=iters[step - 1], 
                display_every=display, 
                callbacks=[checker, resampler],
                model_save_path=os.path.join(self.output_dir, 'Step1', 'Checkpoint', 'model'))
            saveplot_2(loss_history, train_state, issave=True, isplot=False, output_dir=os.path.join(self.output_dir, 'Step1'))
            
        elif step == 2:
            model.compile('adam', lr=lrs[step - 1], loss='MSE', decay=['cosine', sum(iters), 0], external_trainable_variables=[self.beta_T, self.beta_S], dynamic_loss_weight=False,
            train_step_num=2)

            if self.restore_2 is not None:
                retore_path = f'Output/Model/RealData/{self.restore_2[0]}/{self.restore_2[1]}/Checkpoint/model-{self.restore_2[2]}.pt'
                model.restore(retore_path)

                if self.train_var != 'all' and self.reinitialize:
                    self.reinitialize_param(model.net, reinitialize_group=self.train_var)

            else:
                model.restore(os.path.join(self.output_dir, 'Step1', 'Checkpoint', f'model-{iters[0]}.pt'))

        
            loss_history, train_state = model.train(
                iterations=iters[step - 1], 
                display_every=display, 
                callbacks=[checker, resampler, variable],
                model_save_path=os.path.join(self.output_dir, 'Step2', 'Checkpoint', 'model'))
            saveplot_2(loss_history, train_state, issave=True, isplot=False, output_dir=os.path.join(self.output_dir, 'Step1'))

    def realdata_model(self, data, data_val, kernel_initializer="Glorot normal"):
        net = MDRF_Net(n_layers=self.n_layers, layer_widths=self.layer_width, layer_sizes=self.layer_sizes, activation=self.activation_func, kernel_initializer=kernel_initializer)

        if self.train_var != 'all':
            self.freeze_and_unfreeze(net, unfreeze_group=self.train_var)

        if self.sub_learner == 2:
            net.apply_feature_transform(self.coordinate_rotation_transform)

        if self.hard_constraint:
            net.apply_output_transform(self.output_transform)
            

        model = Model2(data, data_val, net, alpha=None, task_name=self.task_name, meta_learning=self.meta_learning, pde_num=6, output_dir=self.output_dir)

        return model


    def output_transform(self, inputs, outputs):
        r, theta, phi, t = unpack_x(inputs)
        tau, sigma, w, v_theta, v_phi, p = unpack_y(outputs)

        # Zero Dirichlet BC
        w_transform = -r * w * 2.5
    
        return torch.cat((tau, sigma, w_transform, v_theta, v_phi, p), dim=1)
    
    def freeze_and_unfreeze(self, net, unfreeze_group=None):
        """
        Freeze all weights of the model, then unfreeze and optionally reinitialize a specific group of weights
        using SIREN-specific initialization.

        Args:
            net (torch.nn.Module): The neural network model whose weights are to be frozen/unfrozen.
            unfreeze_group (str, optional): The group of weights to unfreeze and optionally reinitialize.
                Expected values: 'temp', 'sal', 'w', 'v_theta', 'v_phi', 'pres'.
                If None, all weights remain frozen. Defaults to None.

        Returns:
            None
        """
        # Validate the unfreeze_group input
        valid_groups = {'temp', 'sal', 'w', 'v_theta', 'v_phi', 'pres'}
        if unfreeze_group is not None and unfreeze_group not in valid_groups:
            raise ValueError(f"Invalid unfreeze_group. Expected one of {valid_groups}, but got '{unfreeze_group}'.")

        # Replace 'v_theta' and 'v_phi' with 'v1' and 'v2' to match parameter names
        if unfreeze_group == 'v_theta':
            unfreeze_group = 'v1'
        elif unfreeze_group == 'v_phi':
            unfreeze_group = 'v2'

        # # Print all weights before freezing
        # print("Network weights before freezing:")
        # for name, param in net.named_parameters():
        #     print(f"{name}: shape={param.shape}")

        # Freeze all weights in the model
        for param in net.parameters():
            param.requires_grad = False

        # Unfreeze the specified group of weights
        if unfreeze_group is not None:
            for name, param in net.named_parameters():
                if f"linears_{unfreeze_group}" in name:  # Unfreeze parameters in the specified group
                    param.requires_grad = True

        # Print the trainable parameters for verification
        print("Model trainable parameters:")
        for name, param in net.named_parameters():
            if param.requires_grad:
                print(f"{name}: {param.shape}")

    def reinitialize_param(self, net, reinitialize_group=None):

        # Validate the unfreeze_group input
        valid_groups = {'temp', 'sal', 'w', 'v_theta', 'v_phi', 'pres'}
        if reinitialize_group is not None and reinitialize_group not in valid_groups:
            raise ValueError(f"Invalid unfreeze_group. Expected one of {valid_groups}, but got '{reinitialize_group}'.")

        reinitialize_group = replace_group_name(reinitialize_group)

        # Reinitialize the parameter
        if reinitialize_group is not None:
            for name, param in net.named_parameters():
                if f"linears_{reinitialize_group}" in name:  # Unfreeze parameters in the specified group
                    if 'weight' in name:
                        n_in = param.size(1)  # Input dimension
                        bound = math.sqrt(6 / n_in)  # Uniform distribution bound
                        nn.init.uniform_(param, -bound, bound)
                    elif 'bias' in name:
                        nn.init.zeros_(param)
        
        print("Reinitialization process completed.")

    
    def coordinate_rotation_transform(self, inputs):
        return self.dataloader.coordinates_rotation(inputs, tensor=True)


    def geotime(self):
        space_domain = dde.geometry.Cuboid([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
        time_domain = dde.geometry.TimeDomain(0.0, 1.0)
        return dde.geometry.GeometryXTime(space_domain, time_domain)
    
    def callbacks(self, save_period=10000, variable_file_name='variable.dat', train_step=2):
        gen_folder(os.path.join(self.output_dir, 'Step2', 'Checkpoint'))

        checker = dde.callbacks.ModelCheckpoint(os.path.join(self.output_dir, f'Step{train_step}', 'Checkpoint', 'model'), save_better_only=False, period=save_period)
        resampler = dde.callbacks.PDEPointResampler(period=100, pde_points=True, bc_points=True)
        variable = dde.callbacks.VariableValue([self.beta_T, self.beta_S], period=100, filename=os.path.join(self.output_dir, 'Step2',variable_file_name), precision=6)

        return checker, resampler, variable

    def load_data_seaice(self, data_type='train'):
        data_seaice = self.dataloader.read_data_seaice(data_type)
        ob_coordinate_t_seaice, ob_seaice = self.dataloader.normalization_seaice(data_seaice)

        print("Data shape:", data_seaice.shape)
        ndarray_check(data_seaice)
        print("Observation shape:", ob_coordinate_t_seaice.shape)
        print("Observation shape:", ob_seaice.shape)
        ndarray_check(np.column_stack((ob_coordinate_t_seaice, ob_seaice)))

        observe_seaice = dde.icbc.PointSetBC(ob_coordinate_t_seaice, ob_seaice, component=0, batch_size=self.batch_size, shuffle=True)

        return [observe_seaice]

    def load_data(self, data_type='train', return_type='all'):

        data_argo, data_cur, data_wcur = self.dataloader.read_data(data_type)

        if self.sub_learner is not None:
            data_argo = self.dataloader.data_filter_theta(data_argo, sublearner=self.sub_learner)
            data_cur = self.dataloader.data_filter_theta(data_cur, sublearner=self.sub_learner)
            data_wcur = self.dataloader.data_filter_theta(data_wcur, sublearner=self.sub_learner)

        self.dataloader.set_data_type(data_type)

        ob_coordinate_t_argo, ob_temp, ob_sal = self.dataloader.normalization_argo(data_argo, sublearner=self.sub_learner)

        ob_coordinate_t_wcur, ob_coordinate_t_cur, ob_w, ob_v1, ob_v2 = self.dataloader.normalization_currents(data_wcur, data_cur)

        observe_temp = dde.icbc.PointSetBC(ob_coordinate_t_argo, ob_temp, component=0, batch_size=self.batch_size, shuffle=True)
        observe_sal = dde.icbc.PointSetBC(ob_coordinate_t_argo, ob_sal, component=1, batch_size=self.batch_size, shuffle=True)
        observe_w = dde.icbc.PointSetBC(ob_coordinate_t_wcur, ob_w, component=2, batch_size=self.batch_size, shuffle=True)
        observe_v1 = dde.icbc.PointSetBC(ob_coordinate_t_cur, ob_v1, component=3, batch_size=self.batch_size, shuffle=True)
        observe_v2 = dde.icbc.PointSetBC(ob_coordinate_t_cur, ob_v2, component=4, batch_size=self.batch_size, shuffle=True)

        if return_type == 'all':
            return observe_temp, observe_sal, observe_w, observe_v1, observe_v2
        elif return_type == 'temp_sal':
            return observe_temp, observe_sal
        elif return_type == 'currents':
            return observe_w, observe_v1, observe_v2
        elif return_type == 'temp':
            return [observe_temp]
        elif return_type == 'sal':
            return [observe_sal]
        elif return_type == 'w':
            return [observe_w]
        elif return_type == 'v_theta':
            return [observe_v1]
        elif return_type == 'v_phi':
            return [observe_v2]
    
    def unknown_parameters(self):
        self.beta_T = dde.Variable(0.0)
        self.beta_S = dde.Variable(0.0)

    def train_data(self):
        
        bcs = self.load_data(data_type='train', return_type=self.train_var)
        # bcs_val = self.load_data(data_type='vali', return_type=self.train_var)

        if self.external_source:
            bc_seaice = self.load_data_seaice(data_type='train')
            bc_seaice_val = self.load_data_seaice(data_type='vali')
            bcs = bc_seaice + bcs
            # bcs_val = bc_seaice_val + bcs_val

        self.unknown_parameters()

        self.primitive_equations = lambda X, Y: primitive_equations(X, Y, self.dataloader, self.beta_T, self.beta_S)

        geot = self.geotime()

        data = dde.data.PDE(
            geometry=geot,
            pde=None,
            bcs=bcs,
            num_domain=self.num_domain,
            num_boundary=self.num_boundary,
        )

        # data_val = dde.data.PDE(
        #     geometry=geot,
        #     pde=self.primitive_equations,
        #     bcs=bcs_val,
        #     num_domain=self.num_domain,
        #     num_boundary=self.num_boundary,
        # )
        data_val = None

        data_ = dde.data.PDE(
            geometry=geot,
            pde=None,
            bcs=self.load_data(data_type='train', return_type=self.train_var),
            num_domain=0,
        )

        return data, data_, data_val

    def inference(self, 
                  new_x=None, 
                  new_x_path=None, 
                  save_path='Output/Prediction/Simulation/prediction.npy', 
                  return_results=False):
        
        print('Model inference...')
        
        gen_folder(os.path.dirname(save_path))

        model = self.realdata_model( 
            None, 
            None,
            kernel_initializer="Glorot normal")
        
        if self.beta_T is None or self.beta_S is None:
            self.unknown_parameters()
        model.compile('adam', lr=1e-4, loss='MSE', external_trainable_variables=[self.beta_T, self.beta_S])

        self.model_restore(model)

        if new_x is None:
            new_x_ = np.load(new_x_path)
        else:
            new_x_ = new_x.copy()

        new_x_ = self.dataloader.normalization_coordinates(new_x_)

        prediction = self.batch_predict(model, new_x_, batch_size=100000)

        prediction = self.dataloader.anti_normalization(prediction, self.sub_learner)
        # print('Done.')

        prediction = self.calculate_dens(prediction)
        
        if return_results:
            return prediction
        
        np.save(save_path, prediction) 

    
    def calculate_dens(self, prediction):
        tau, sigma, w, v_theta, v_phi, p = unpack_y(prediction)
        r_beta = np.sqrt(TAU_STD / SIGMA_STD)
        dens = RHO_0 - (1 / (1 + np.exp(-BETA_TAU))) * (tau - TAU_MEAN) + (1 / (1 + np.exp(-BETA_SIGMA))) * r_beta * (sigma - SIGMA_MEAN)
        prediction = np.column_stack((prediction, dens))

        return prediction

    

    def model_restore(self, 
                      model):
        
        task_main = self.restore_main
        task_sub = self.restore_sub

        checkpoint_main = f'Output/Model/RealData/{task_main[0]}/Step{task_main[1]}/Checkpoint/model-{task_main[2]}.pt'
        model.restore(checkpoint_main, verbose=1)
        
        if task_sub is not None:
            for task in task_sub:
                checkpoint_sub = f'Output/Model/RealData/{task[0]}/Step{task[1]}/Checkpoint/model-{task[2]}.pt'
                model.restore(checkpoint_sub, verbose=1, load_var=task[3])


    def inference_and_plot(self, 
                           new_x_path, 
                           save_path,
                           t_list=TIME_LIST,
                           plot_range=range(7)):
        
        self.inference(new_x_path=new_x_path, save_path=save_path)

        self.plotter.plot(task_name=self.task_name,  
                        fit_path=save_path, 
                        t_list=t_list,
                        new_x_path=new_x_path,
                        area=self.task_name.split('/')[0],
                        sublearner=self.sub_learner,
                        var_range=plot_range)

    def inference_and_plot_slice(self, new_path='Data/RealData/Local'):

        save_path = f'Output/Plot/RealData/{self.task_name}/Slice'
        
        new_x_depth_path = os.path.join(new_path, 'new_x_plot_depth.npy')
        new_x_latitude_path = os.path.join(new_path, 'new_x_plot_depth.npy')
        new_x_longitude_path = os.path.join(new_path, 'new_x_plot_depth.npy')

        new_x_depth = np.load(new_x_depth_path)
        new_x_latitude = np.load(new_x_latitude_path)
        new_x_longitude = np.load(new_x_depth_path)
        
        pre_depth = self.inference(new_x=new_x_depth, return_results=True)[:, :2]
        pre_latitude = self.inference(new_x=new_x_latitude, return_results=True)[:, :2]
        pre_longitude = self.inference(new_x=new_x_longitude, return_results=True)[:, :2]

        df_depth = np.column_stack((new_x_depth, pre_depth))
        df_latitude = np.column_stack((new_x_latitude, pre_latitude))
        df_longitude = np.column_stack((new_x_longitude, pre_longitude))

        self.plotter.plot_data_glory_tempsal_slice_depth(df_depth, save_path, generate_vmin_vmax=False)
        
    
    def test(self, data_type='test', print_and_save=True, return_bool=False, save_pre_bool=False):

        data_argo, data_cur, data_wcur = self.dataloader.read_data(data_type)

        if self.sub_learner is not None:
            data_argo = self.dataloader.data_filter_theta(data_argo, sublearner=self.sub_learner)
            data_cur = self.dataloader.data_filter_theta(data_cur, sublearner=self.sub_learner)
            data_wcur = self.dataloader.data_filter_theta(data_wcur, sublearner=self.sub_learner)

        if save_pre_bool:
            save_pre_path = os.path.join('Output/Prediction/RealData', self.task_name)

        return calculate_and_save_rmse(
            data_argo, data_cur, data_wcur, self.inference, self.output_dir, print_and_save, return_bool, save_pre_path
        )
        

    def validate(self, step_num=2, iter_num=[0, 100000]):
        # Extract start and end iteration from iter_num
        start_iter, end_iter = iter_num
        
        # Define base directory for model checkpoints
        base_dir = os.path.join(self.output_dir, f'Step{step_num}', 'Checkpoint')
        
        # Initialize lists to store RMSE values and corresponding iterations
        iterations = []
        rmse_temp_list = []
        rmse_sal_list = []
        rmse_w_list = []
        rmse_v_theta_list = []
        rmse_v_phi_list = []
        
        # Loop through the specified range of iterations
        for current_iter in tqdm(range(start_iter, end_iter + 1), desc="Processing iterations", unit="iter", mininterval=1):
            # Construct the path to the model checkpoint
            model_path = os.path.join(base_dir, f'model-{current_iter}.pt')
            
            # Check if the checkpoint file exists
            if os.path.exists(model_path):
                # Call the self.test method to get RMSE values
                rmse_temp, rmse_sal, rmse_w, rmse_v_theta, rmse_v_phi = self.test(
                    step_num=step_num,
                    iter_num=current_iter,
                    data_type='vali',
                    print_and_save=False,
                    return_bool=True
                )
                
                # Store the iteration and corresponding RMSE values
                iterations.append(current_iter)
                rmse_temp_list.append(rmse_temp)
                rmse_sal_list.append(rmse_sal)
                rmse_w_list.append(rmse_w)
                rmse_v_theta_list.append(rmse_v_theta)
                rmse_v_phi_list.append(rmse_v_phi)

        save_folder = 'validation_rmse'
        output_path = os.path.join(self.output_dir, f'Step{step_num}', save_folder)
        
        df = pd.DataFrame({
            'Iteration': iterations,
            'RMSE_temp': rmse_temp_list,
            'RMSE_sal': rmse_sal_list,
            'RMSE_w': rmse_w_list,
            'RMSE_v_theta': rmse_v_theta_list,
            'RMSE_v_phi': rmse_v_phi_list
        })
        os.makedirs(output_path, exist_ok=True)
        df.to_csv(os.path.join(output_path, 'validation_results.csv'), index=False)

        # Plot and save RMSE values for each variable
        self.plotter._plot_rmse(iterations, rmse_temp_list, 'RMSE Temperature', os.path.join(output_path, 'rmse_temp.png'))
        self.plotter._plot_rmse(iterations, rmse_sal_list, 'RMSE Salinity', os.path.join(output_path, 'rmse_sal.png'))
        self.plotter._plot_rmse(iterations, rmse_w_list, 'RMSE W', os.path.join(output_path, 'rmse_w.png'))
        self.plotter._plot_rmse(iterations, rmse_v_theta_list, 'RMSE V Theta', os.path.join(output_path, 'rmse_v_theta.png'))
        self.plotter._plot_rmse(iterations, rmse_v_phi_list, 'RMSE V Phi', os.path.join(output_path, 'rmse_v_phi.png'))
        


    def batch_predict(self, model, input_data, batch_size=10000):
        """
        Perform batch prediction on large input data to avoid GPU memory overflow.

        Args:
            model (keras.Model): The trained model used for prediction.
            input_data (np.ndarray): The input data for prediction.
            batch_size (int): The number of samples per batch. Adjust based on GPU memory.

        Returns:
            np.ndarray: The predictions for the entire input data.
        """
        # Get the number of samples in the input data
        num_samples = input_data.shape[0]
        
        # Get the output shape of the model (excluding the batch dimension)
        output_shape = model.predict(input_data[:1]).shape[1:]
        
        # Pre-allocate memory for the predictions array
        predictions = np.zeros((num_samples, *output_shape), dtype=np.float32)
        
        # Perform batch prediction with a progress bar
        for i in tqdm(range(0, num_samples, batch_size), 
                    desc="Batch Prediction Progress", 
                    unit="batch"):
            # Get the current batch
            batch = input_data[i:i + batch_size]
            
            # Predict the current batch
            batch_prediction = model.predict(batch)
            
            # Store the predictions in the pre-allocated array
            predictions[i:i + batch_size] = batch_prediction
        
        return predictions  
    


def calculate_and_save_rmse(data_argo, data_cur, data_wcur, inference_func, output_dir, print_and_save=True, return_bool=False, save_pre_path=None):

    new_x_argo = data_argo[:, :4]
    new_x_cur = data_cur[:, :4]
    new_x_wcur = data_wcur[:, :4]

    pre_argo = inference_func(new_x=new_x_argo, return_results=True)
    pre_cur = inference_func(new_x=new_x_cur, return_results=True)
    pre_wcur = inference_func(new_x=new_x_wcur, return_results=True)

    rmse_temp = root_mean_squared_error(pre_argo[:, 0], data_argo[:, 4])
    rmse_sal = root_mean_squared_error(pre_argo[:, 1], data_argo[:, 5])
    rmse_w = root_mean_squared_error(pre_wcur[:, 2], data_wcur[:, 4])
    rmse_v_theta = root_mean_squared_error(pre_cur[:, 3], data_cur[:, 4])
    rmse_v_phi = root_mean_squared_error(pre_cur[:, 4], data_cur[:, 5])

    if save_pre_path is not None:
        np.save(os.path.join(save_pre_path, 'temp_pred.npy'), pre_argo[:, 0])
        np.save(os.path.join(save_pre_path, 'sal_pred.npy'), pre_argo[:, 1])
        np.save(os.path.join(save_pre_path, 'w_pred.npy'), pre_wcur[:, 2])
        np.save(os.path.join(save_pre_path, 'v_theta_pred.npy'), pre_cur[:, 3])
        np.save(os.path.join(save_pre_path, 'v_phi_pred.npy'), pre_cur[:, 4])

    if print_and_save:
        print(f"RMSE temp: {rmse_temp}")
        print(f"RMSE sal: {rmse_sal}")
        print(f"RMSE w: {rmse_w}")
        print(f"RMSE v_theta: {rmse_v_theta}")
        print(f"RMSE v_phi: {rmse_v_phi}")

        rmse_data = {
            'variable': ['temp', 'sal', 'w', 'v_theta', 'v_phi'],
            'rmse': [rmse_temp, rmse_sal, rmse_w, rmse_v_theta, rmse_v_phi]
        }

        rmse_df = pd.DataFrame(rmse_data)
        save_path = os.path.join(output_dir, 'rmse_results.csv')
        gen_folder(os.path.dirname(save_path))
        rmse_df.to_csv(save_path, index=False)

        print("RMSE results saved to 'rmse_results.csv'")

    if return_bool:
        return rmse_temp, rmse_sal, rmse_w, rmse_v_theta, rmse_v_phi

    
class EnsembleModel:
    def __init__(self, 
                 sublearner_1, 
                 sublearner_2,
                 plotter,
                 dataloader):
        self.sublearner_1 = sublearner_1
        self.sublearner_2 = sublearner_2

        task_name_1 = self.sublearner_1.task_name.partition("/")[2].replace("/", "-")
        task_name_2 = self.sublearner_2.task_name.partition("/")[2].replace("/", "-")
        self.ensemble_task_name = f'Global/Ensemble/{task_name_1}|{task_name_2}'

        self.output_dir = f'Output/Model/RealData/{self.ensemble_task_name}'

        self.plotter = plotter
        self.dataloader = dataloader
    
    def inference(self, new_x=None, new_x_path=None, return_results=False, save_name='prediction_plot.npy', plot_bool=False, t_list=None, plot_range=range(7), fig_save_path=None):

        save_path=f'Output/Prediction/RealData/{self.ensemble_task_name}/{save_name}'

        gen_folder(os.path.dirname(save_path))

        if new_x is None:
            new_x = np.load(new_x_path)

        pre_1 = self.sublearner_1.inference(new_x,  return_results=True)
        pre_2 = self.sublearner_2.inference(new_x, return_results=True)

        theta = new_x[:, 1]
        pre_1 *= self.weight_1(theta)
        pre_2 *= self.weight_2(theta)
        prediction = pre_1 + pre_2

        if return_results:
            return prediction
        
        np.save(save_path, prediction) 

        print(self.ensemble_task_name)

        if plot_bool:
            self.plotter.plot(task_name=self.ensemble_task_name,
                            fit_path=save_path,
                            t_list=t_list,
                            new_x_path=new_x_path,
                            area='Global',
                            var_range=plot_range,
                            save_path=fig_save_path)

    def inference_and_plot_polar(self, 
                                 new_x_path='Data/RealData/Global/new_x_plot_polar.npy', 
                                 save_path='Output/Plot/RealData', 
                                 plot_range=range(2)):
        
        save_path = os.path.join(save_path, self.ensemble_task_name)

        new_x = np.load(new_x_path)       
        pre = self.inference(new_x=new_x, return_results=True)

        for i in tqdm(plot_range, desc="Processing variables"):
            for r_ in tqdm(self.dataloader.read_depths()[:40], desc="Processing depths", leave=False):

                pre_plot = pre[new_x[:, 0] == r_]
                new_x_plot = new_x[new_x[:, 0] == r_]

                values = pre_plot[:, i]
                r, theta, phi, t = unpack_x(new_x_plot)

                for region in ['Arctic', 'Antarctic']:

                    self.plotter.plot_polar_scatter_cartopy(phi, theta, values, 
                                            var_name=VAR_UNITS[i], 
                                            cmap=VAR_COLORS[i],
                                            point_size=2,
                                            output_path=os.path.join(save_path, f'{region}/{VAR_NAMES[i]}_{r_:.2f}m.png'),
                                            vmin_vmax=self.plotter.read_vmin_vmax_polar(region, r_, i),
                                            region=region,
                                            log_transform=False,
                                            face_color=False,
                                            depth=r_)

    def inference_and_plot_slice(self, new_path='Data/RealData/Global'):

        save_path = f'Output/Plot/RealData/{self.ensemble_task_name}/Slice'
        
        new_x_depth_path = os.path.join(new_path, 'new_x_plot_depth.npy')
        new_x_latitude_path = os.path.join(new_path, 'new_x_plot_latitude.npy')
        new_x_longitude_path = os.path.join(new_path, 'new_x_plot_longitude.npy')

        new_x_depth = np.load(new_x_depth_path)
        new_x_latitude = np.load(new_x_latitude_path)
        new_x_longitude = np.load(new_x_longitude_path)
        
        pre_depth = self.inference(new_x=new_x_depth, return_results=True)[:, :2]
        pre_latitude = self.inference(new_x=new_x_latitude, return_results=True)[:, :2]
        pre_longitude = self.inference(new_x=new_x_longitude, return_results=True)[:, :2]

        df_depth = np.column_stack((new_x_depth, pre_depth))
        df_latitude = np.column_stack((new_x_latitude, pre_latitude))
        df_longitude = np.column_stack((new_x_longitude, pre_longitude))

        self.plotter.plot_data_glory_tempsal_slice_depth(df_depth, save_path, generate_vmin_vmax=False)
        self.plotter.plot_data_glory_tempsal_slice_latitude(df_latitude, save_path, generate_vmin_vmax=False)
        self.plotter.plot_data_glory_tempsal_slice_longitude(df_longitude, save_path, generate_vmin_vmax=False)

    def inference_and_plot_slice_pre(self, new_path='Data/RealData/Global', time='202302'):

        save_path = f'Output/Plot/RealData/{self.ensemble_task_name}/Slice{time}'
        
        new_x_depth_path = os.path.join(new_path, 'new_x_plot_depth_pre.npy')

        new_x_depth = np.load(new_x_depth_path)
        
        pre_depth = self.inference(new_x=new_x_depth, return_results=True)[:, :2]

        df_depth = np.column_stack((new_x_depth, pre_depth))

        self.plotter.plot_data_glory_tempsal_slice_depth(df_depth, save_path, generate_vmin_vmax=False, time='202302')

    def inference_and_plot_slice_depth_vtheta(self, new_path='Data/RealData/Global'):
        save_path = f'Output/Plot/RealData/{self.ensemble_task_name}/Slice'
        
        new_x_depth_path = os.path.join(new_path, 'new_x_plot_depth.npy')

        new_x_depth = np.load(new_x_depth_path)
        
        pre_depth = self.inference(new_x=new_x_depth, return_results=True)[:, 3]

        df_depth = np.column_stack((new_x_depth, pre_depth))

        self.plotter.plot_vtheta_slice_depth_(df_depth, save_path, generate_vmin_vmax=False)
    
    def test(self, data_type='test', print_and_save=True, return_bool=False, save_pre_bool=False):

        data_argo, data_cur, data_wcur = self.dataloader.read_data(data_type)

        if save_pre_bool:
            save_pre_path = os.path.join('Output/Prediction/RealData', self.ensemble_task_name)

        return calculate_and_save_rmse(
            data_argo, data_cur, data_wcur, self.inference, self.output_dir, print_and_save, return_bool,
            save_pre_path=save_pre_path
        )
        

    def weight_1(self, theta):
        abs_theta = np.abs(theta)

        result = np.where(abs_theta <= 40, 1, 
                        np.where(abs_theta < 45, (45 - abs_theta) / 5, 0))
        
        return result.reshape(-1, 1)
    
    def weight_2(self, theta):
        abs_theta = np.abs(theta)

        result = np.where(abs_theta >= 45, 1, 
                        np.where(abs_theta > 40, (abs_theta - 40) / 5, 0))
        
        return result.reshape(-1, 1)

        

if __name__ == '__main__':
    pass
    

    