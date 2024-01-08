def Primitive_Equations(x, y):

    z = x[:, 0:1]
    theta = x[:, 1:2]
    phi = x[:, 2:3] 
    t = x[:, 3:4]

    z = z * 2000 - 2000
    
    tau = y[:, 0:1]
    sigma = y[:, 1:2]
    w = y[:, 2:3]
    v_theta = y[:, 3:4]
    v_phi = y[:, 4:5]
    pi = y[:, 5:6]
    
    # train_mean_std = 
    
    tau = tau * train_mean_std.loc['std', 'temp'] + train_mean_std.loc['mean', 'temp']
    sigma = tau * train_mean_std.loc['std', 'temp'] + train_mean_std.loc['mean', 'temp']
    w = w * train_mean_std.loc['std', 'w'] + train_mean_std.loc['mean', 'w']
    v_theta = v_theta * train_mean_std.loc['std', 'v_theta'] + train_mean_std.loc['mean', 'v_theta']
    v_phi = v_phi * train_mean_std.loc['std', 'v_phi'] + train_mean_std.loc['mean', 'v_phi']
    
    pi_E = 10 ** 5
    pi *= pi_E
    
    tau_0 = train_mean_std.loc['mean', 'temp']
    sigma_0 = train_mean_std.loc['mean', 'sal']
    
    # Unknown parameters
    beta_tau = y[:, 6:7]
    beta_sigma = y[:, 7:8]

    dtau_z = dde.grad.jacobian(y, x, i=0, j=0) * train_mean_std.loc['std', 'temp'] / 2000
    dtau_theta = dde.grad.jacobian(y, x, i=0, j=1) * train_mean_std.loc['std', 'temp']
    dtau_phi = dde.grad.jacobian(y, x, i=0, j=2) * train_mean_std.loc['std', 'temp']
    dtau_t = dde.grad.jacobian(y, x, i=0, j=3) * train_mean_std.loc['std', 'temp']
    dsigma_z = dde.grad.jacobian(y, x, i=1, j=0) * train_mean_std.loc['std', 'sal'] / 2000
    dsigma_theta = dde.grad.jacobian(y, x, i=1, j=1) * train_mean_std.loc['std', 'sal']
    dsigma_phi = dde.grad.jacobian(y, x, i=1, j=2) * train_mean_std.loc['std', 'sal']
    dsigma_t = dde.grad.jacobian(y, x, i=1, j=3) * train_mean_std.loc['std', 'sal']
    dw_z = dde.grad.jacobian(y, x, i=2, j=0) * train_mean_std.loc['std', 'w'] / 2000
    # dw_theta = dde.grad.jacobian(y, x, i=2, j=1) * train_mean_std.loc['std', 'w']
    # dw_phi = dde.grad.jacobian(y, x, i=2, j=2) * train_mean_std.loc['std', 'w']
    # dw_t = dde.grad.jacobian(y, x, i=2, j=3) * train_mean_std.loc['std', 'w']
    dvtheta_z = dde.grad.jacobian(y, x, i=3, j=0) * train_mean_std.loc['std', 'v_theta'] / 2000
    dvtheta_theta = dde.grad.jacobian(y, x, i=3, j=1) * train_mean_std.loc['std', 'v_theta']
    dvtheta_phi = dde.grad.jacobian(y, x, i=3, j=2) * train_mean_std.loc['std', 'v_theta']
    dvtheta_t = dde.grad.jacobian(y, x, i=3, j=3) * train_mean_std.loc['std', 'v_theta']
    dvphi_z = dde.grad.jacobian(y, x, i=4, j=0) * train_mean_std.loc['std', 'v_phi'] / 2000
    dvphi_theta = dde.grad.jacobian(y, x, i=4, j=1) * train_mean_std.loc['std', 'v_phi']
    dvphi_phi = dde.grad.jacobian(y, x, i=4, j=2) * train_mean_std.loc['std', 'v_phi']
    dvphi_t = dde.grad.jacobian(y, x, i=4, j=3) * train_mean_std.loc['std', 'v_phi']
    dpi_z = dde.grad.jacobian(y, x, i=5, j=0) * pi_E / 2000
    dpi_theta = dde.grad.jacobian(y, x, i=5, j=1) * pi_E
    dpi_phi = dde.grad.jacobian(y, x, i=5, j=2) * pi_E
    # dpi_t = dde.grad.jacobian(y, x, i=5, j=3) * pi_E

    dtau_z_2 = dde.grad.hessian(y, x, component=0, i=0, j=0) * train_mean_std.loc['std', 'temp'] / 2000 ** 2
    dtau_theta_2 = dde.grad.hessian(y, x, component=0, i=1, j=1) * train_mean_std.loc['std', 'temp']
    dtau_phi_2 = dde.grad.hessian(y, x, component=0, i=2, j=2) * train_mean_std.loc['std', 'temp']
    # dtau_t_2 = dde.grad.hessian(y, x, component=0, i=3, j=3) * train_mean_std.loc['std', 'temp']
    dsigma_z_2 = dde.grad.hessian(y, x, component=1, i=0, j=0) * train_mean_std.loc['std', 'sal'] / 2000 ** 2
    dsigma_theta_2 = dde.grad.hessian(y, x, component=1, i=1, j=1) * train_mean_std.loc['std', 'sal']
    dsigma_phi_2 = dde.grad.hessian(y, x, component=1, i=2, j=2) * train_mean_std.loc['std', 'sal']
    # dsigma_t_2 = dde.grad.hessian(y, x, component=1, i=3, j=3) * train_mean_std.loc['std', 'sal']
    # dw_z_2 = dde.grad.hessian(y, x, component=2, i=0, j=0) * train_mean_std.loc['std', 'w'] / 2000 ** 2
    # dw_theta_2 = dde.grad.hessian(y, x, component=2, i=1, j=1) * train_mean_std.loc['std', 'w']
    # dw_phi_2 = dde.grad.hessian(y, x, component=2, i=2, j=2) * train_mean_std.loc['std', 'w']
    # dw_t_2 = dde.grad.hessian(y, x, component=2, i=3, j=3) * train_mean_std.loc['std', 'w']
    dvtheta_z_2 = dde.grad.hessian(y, x, component=3, i=0, j=0) * train_mean_std.loc['std', 'v_theta'] / 2000 ** 2
    dvtheta_theta_2 = dde.grad.hessian(y, x, component=3, i=1, j=1) * train_mean_std.loc['std', 'v_theta']
    dvtheta_phi_2 = dde.grad.hessian(y, x, component=3, i=2, j=2) * train_mean_std.loc['std', 'v_theta']
    # dvtheta_t_2 = dde.grad.hessian(y, x, component=3, i=3, j=3) * train_mean_std.loc['std', 'v_theta']
    dvphi_z_2 = dde.grad.hessian(y, x, component=4, i=0, j=0) * train_mean_std.loc['std', 'v_phi'] / 2000 ** 2
    dvphi_theta_2 = dde.grad.hessian(y, x, component=4, i=1, j=1) * train_mean_std.loc['std', 'v_phi']
    dvphi_phi_2 = dde.grad.hessian(y, x, component=4, i=2, j=2) * train_mean_std.loc['std', 'v_phi']
    # dvphi_t_2 = dde.grad.hessian(y, x, component=4, i=3, j=3) * train_mean_std.loc['std', 'v_phi']
    # dpi_z_2 = dde.grad.hessian(y, x, component=5, i=0, j=0) * pi_E / 2000 ** 2
    # dpi_theta_2 = dde.grad.hessian(y, x, component=5, i=1, j=1) * pi_E
    # dpi_phi_2 = dde.grad.hessian(y, x, component=5, i=2, j=2) * pi_E
    # dpi_t_2 = dde.grad.hessian(y, x, component=5, i=3, j=3) * pi_E
    # Unnecessary automatic differential calculation will increase the training time

    # equations:
    
    a = 6.4 * 10 ** 6 # radius of the earth
    rho_0 = 10 ** 3 # reference value of density
    Omega = 10 ** (-4) # angular velocity of the earth
    mu = mu_tau = mu_sigma = 10 ** 4 # eddy viscosity coefficients
    nu = nu_tau = nu_sigma = 1.5 * 10 ** (-4)
    g = 9.8 # acceleration due to gravity
    
    rho = rho_0
    
    equa_1 = dvtheta_t + (v_theta / a * dvtheta_theta + v_phi / (a * torch.sin(theta)) * dvtheta_phi - v_phi ** 2 / a * torch.tan(theta) ** (-1)) + w * dvtheta_z + 1 / rho_0 * 1 / a * dpi_theta + 2 * Omega * torch.cos(theta) * (-v_phi) - mu * (1 / (a ** 2 * torch.sin(theta)) * (torch.cos(theta) * dvtheta_theta + torch.sin(theta) * dvtheta_theta_2 + 1 / torch.sin(theta) * dvtheta_phi_2) - 2 * torch.cos(theta) / (a ** 2 * torch.sin(theta) ** 2) * dvphi_phi - v_theta / (a ** 2 * torch.sin(theta) ** 2)) - nu * dvtheta_z_2
    
    equa_2 = dvphi_t + (v_theta / a * dvphi_theta + v_phi / (a * torch.sin(theta)) * dvphi_phi + v_theta * v_phi / a * torch.tan(theta) ** (-1)) + w * dvphi_z + 1 / rho_0 * 1 / (a * torch.sin(theta)) * dpi_phi + 2 * Omega * torch.cos(theta) * v_theta - mu * (1 / (a ** 2 * torch.sin(theta)) * (torch.cos(theta) * dvphi_theta + torch.sin(theta) * dvphi_theta_2 + 1 / torch.sin(theta) * dvphi_phi_2) - 2 * torch.cos(theta) / (a ** 2 * torch.sin(theta) ** 2) * dvtheta_phi - v_phi / (a ** 2 * torch.sin(theta) ** 2)) - nu * dvphi_z_2
    
    rho = rho_0 * (1 - beta_tau * (tau - tau_0) + beta_sigma * (sigma - sigma_0))
    equa_3 = dpi_z + rho * g
    
    equa_4 = 1 / (a * torch.sin(theta)) * (dvtheta_theta * torch.sin(theta) + v_theta * torch.cos(theta) + dvphi_phi) + dw_z
    
    equa_5 = dtau_t + v_theta / a * dtau_theta + v_phi / (a * torch.sin(theta) * dtau_phi) + w * dtau_z - mu_tau * 1 / (a ** 2 * torch.sin(theta)) * (torch.cos(theta) * dtau_theta + torch.sin(theta) * dtau_theta_2 + 1 / torch.sin(theta) * dtau_phi_2) - nu_tau * dtau_z_2
    
    equa_6 = dsigma_t + v_theta / a * dsigma_theta + v_phi / (a * torch.sin(theta) * dsigma_phi) + w * dsigma_z - mu_sigma * 1 / (a ** 2 * torch.sin(theta)) * (torch.cos(theta) * dsigma_theta + torch.sin(theta) * dsigma_theta_2 + 1 / torch.sin(theta) * dsigma_phi_2) - nu_sigma * dsigma_z_2
    
    # print(type(equa_5), equa_5)
    
    return list(map(torch.nan_to_num, [equa_1, equa_2, equa_3, equa_4, equa_5, equa_6]))
    # return equa_6
