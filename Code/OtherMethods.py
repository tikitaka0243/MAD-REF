from datetime import datetime
import os
import re
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from pykrige.rk import RegressionKriging
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.metrics import root_mean_squared_error
import torch
import torch.nn as nn
from tqdm import tqdm
import lightgbm as lgb
from scipy.spatial import cKDTree
from sklearn.preprocessing import StandardScaler
from scipy.optimize import curve_fit
from sklearn.preprocessing import StandardScaler
import gpytorch
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy
from gpytorch.kernels import ScaleKernel, RBFKernel, PeriodicKernel
from gpytorch.means import ConstantMean
from torch.utils.data import TensorDataset, DataLoader
import logging

from Code.Plot import RealDataPlot
from Code.utils import VAR_COLORS, VAR_NAMES, VAR_UNITS, GP_INDUCING_POINTS, GP_INDUCING_SAMPLING, data_split, find_index_in_list, gen_folder, ndarray_check, proportional_sampling, unpack_x
from Code.DataProcess import RealData, SimulationData


def gaussian_process_inference(X_train, y_train, X_test):
    kernel = RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e3))
    gpr = GaussianProcessRegressor(kernel=kernel, alpha=0.1, n_restarts_optimizer=10)
    gpr.fit(X_train, y_train)
    
    batch_size = 10000
    batch_num = int(np.ceil(len(X_test) / batch_size))
    y_pred = np.zeros(len(X_test))
    for i in tqdm(range(batch_num), desc='Gaussian Process', mininterval=8):
        if i < batch_num - 1:
            X_batch = X_test[(i * batch_size):((i + 1) * batch_size)]
            y_pred[(i * batch_size):((i + 1) * batch_size)] = gpr.predict(X_batch)
        elif i == batch_num - 1:
            X_batch = X_test[(i * batch_size):]
            y_pred[(i * batch_size):] = gpr.predict(X_batch)
        
    return y_pred


def regression_kriging(X_train, y_train, X_test):
    x_train = X_train[:, 0:2]
    p_train = X_train[:, 2:3]
    x_test = X_test[:, 0:2]
    p_test = X_test[:, 2:3]
    target_train = y_train
    
    rf_model = RandomForestRegressor(n_estimators=100)
    lr_model = LinearRegression(copy_X=True, fit_intercept=False)
    rk = RegressionKriging(regression_model=lr_model, n_closest_points=10)
    rk.fit(p_train, x_train, target_train)
    test_pre = rk.predict(p_test, x_test)
    
    return test_pre


def compute_covariance_matrix(X_neighbors, params):
    """
    Compute the covariance matrix for a local neighborhood.
    
    Args:
        X_neighbors: Neighboring points (n_neighbors, 4)
        params: Dictionary of covariance parameters
        
    Returns:
        cov_matrix: Covariance matrix (n_neighbors, n_neighbors)
    """
    a = np.array([params['a_d'], params['a_lat'], params['a_lon'], params['a_t']])  # Range parameters for each dimension
    sigma_sq = params['sigma_sq']  # Sill parameter
    nugget = params['nugget']  # Nugget effect
    
    # Compute normalized distances between all pairs of neighbors
    delta = (X_neighbors[:, None, :] - X_neighbors[None, :, :]) / a
    h = np.sqrt(np.sum(delta**2, axis=2))  # Euclidean distance in scaled space
    
    # Exponential covariance model
    cov_matrix = sigma_sq * np.exp(-3 * h)
    np.fill_diagonal(cov_matrix, cov_matrix.diagonal() + nugget)  # Add nugget effect to diagonal
    return cov_matrix

def compute_covariance_vector(x_test, X_neighbors, params):
    """
    Compute covariance vector between test point and neighbors.
    
    Args:
        x_test: Test point (4,)
        X_neighbors: Neighboring points (n_neighbors, 4)
        params: Dictionary of covariance parameters
        
    Returns:
        cov_vector: Covariance vector (n_neighbors,)
    """
    a = np.array([params['a_d'], params['a_lat'], params['a_lon'], params['a_t']])  # Range parameters
    sigma_sq = params['sigma_sq']  # Sill parameter
    
    # Compute normalized distances between test point and neighbors
    delta = (X_neighbors - x_test) / a
    h = np.sqrt(np.sum(delta**2, axis=1))  # Euclidean distance in scaled space
    
    # Exponential covariance model
    return sigma_sq * np.exp(-3 * h)

def estimate_parameters(X, y, m=1000):
    """
    Estimate variogram parameters from data.
    
    Args:
        X: Input features (n_samples, 4)
        y: Target values (n_samples,)
        m: Number of sample pairs to use for estimation
        
    Returns:
        params: Dictionary of estimated parameters
    """
    n = X.shape[0]
    indices = np.random.choice(n, size=(m, 2), replace=True)  # Random sample pairs
    diffs = X[indices[:, 0]] - X[indices[:, 1]]  # Differences between pairs
    gammas = 0.5 * (y[indices[:, 0]] - y[indices[:, 1]])**2  # Semivariances

    params = {}
    a_params = []
    for dim in range(4):  # Estimate range parameter for each dimension
        h_dim = np.abs(diffs[:, dim])  # Distances along current dimension
        valid = h_dim > 1e-8  # Filter out very small distances
        if np.sum(valid) < 10:  # Skip if insufficient data
            a_params.append(1.0)
            continue
            
        # Bin distances and compute mean semivariances
        bins = np.linspace(0, np.percentile(h_dim, 95), 20)
        bin_means, bin_edges = np.histogram(h_dim, bins, weights=gammas)
        bin_counts = np.histogram(h_dim, bins)[0]
        with np.errstate(divide='ignore', invalid='ignore'):
            bin_means = bin_means / bin_counts
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        try:
            # Fit exponential variogram model
            sigma_sq = np.nanmax(bin_means)
            popt, _ = curve_fit(lambda h, a: sigma_sq*(1 - np.exp(-3*h/a)),
                              bin_centers[bin_counts>0], 
                              bin_means[bin_counts>0],
                              p0=[np.nanmedian(bin_centers)])
            a_params.append(popt[0])
        except:
            a_params.append(1.0)

    # Store estimated parameters
    params.update({
        'a_d': max(a_params[0], 0.1),  # Depth range
        'a_lat': max(a_params[1], 0.1),  # Latitude range
        'a_lon': max(a_params[2], 0.1),  # Longitude range
        'a_t': max(a_params[3], 0.1),  # Time range
        'sigma_sq': np.var(y),  # Sill
        'nugget': np.percentile(gammas[np.linalg.norm(diffs, axis=1) < 1e-4], 25)  # Nugget
                  if np.any(np.linalg.norm(diffs, axis=1) < 1e-4) else 0.1
    })
    return params

def spatiotemporal_kriging(X_train, y_train, X_test, k_neighbors=50, sample_pairs=1000):
    """
    Perform spatiotemporal kriging interpolation.
    
    Args:
        X_train: Training features (n_samples, 4) [depth, lat, lon, time]
        y_train: Training target values (n_samples, 1)
        X_test: Test features (m_samples, 4)
        k_neighbors: Number of nearest neighbors to use
        sample_pairs: Number of sample pairs for parameter estimation
        
    Returns:
        y_test: Predicted values (m_samples, 1)
    """
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    y_train = y_train.flatten()
    
    # Build spatial index for fast neighbor search
    tree = cKDTree(X_train_scaled)
    
    # Estimate model parameters
    params = estimate_parameters(X_train_scaled, y_train, m=sample_pairs)
    
    # Predict each test point
    y_pred = []
    for x_test in tqdm(X_test_scaled, desc="Predicting", unit="point", mininterval=5):
        # Find nearest neighbors
        distances, indices = tree.query(x_test, k=k_neighbors)
        if distances.size == 0:  # Fallback if no neighbors found
            y_pred.append(np.mean(y_train))
            continue
            
        X_neighbors = X_train_scaled[indices]
        y_neighbors = y_train[indices]
        
        try:
            # Compute covariance matrix and vector
            C = compute_covariance_matrix(X_neighbors, params)
            c0 = compute_covariance_vector(x_test, X_neighbors, params)
            
            # Build extended matrix for kriging system
            n = C.shape[0]
            C_ext = np.vstack([np.hstack([C, np.ones((n, 1))]), 
                             np.hstack([np.ones((1, n)), [[0]]])])
            c0_ext = np.concatenate([c0, [1]])
            
            # Solve kriging system for weights
            weights = np.linalg.solve(C_ext, c0_ext)[:n]
            y_pred.append(np.dot(weights, y_neighbors))
        except:  # Fallback if matrix is singular
            y_pred.append(np.mean(y_neighbors))
    
    return np.array(y_pred).reshape(-1, 1)


def lightgbm_inference(X_train, y_train, X_test, batch_size=10000, log_file='loss_log.txt', **lgb_params):
    """
    LightGBM regression inference with batched prediction and loss logging
    
    Parameters:
        X_train (np.ndarray): Training features of shape (n_samples_train, n_features)
        y_train (np.ndarray): Training target values of shape (n_samples_train,)
        X_test (np.ndarray): Test features of shape (n_samples_test, n_features)
        batch_size (int): Batch size for prediction (default: 10000)
        log_file (str): File path to save the loss values (default: 'loss_log.txt')
        **lgb_params: Custom LightGBM parameters to override defaults
        
    Returns:
        y_pred (np.ndarray): Predicted values for test set of shape (n_samples_test,)
    """
    
    # Merge default parameters with user-defined parameters
    params = {
        'boosting_type': 'gbdt',       # Gradient Boosting Decision Tree
        'objective': 'regression',     # Regression task
        'metric': 'rmse',             # Evaluation metric
        'num_leaves': 63,              # Controls model complexity
        'learning_rate': 0.05,         # Step size shrinkage
        'feature_fraction': 0.8,       # Random column subsampling
        'bagging_fraction': 0.8,       # Random row subsampling
        'bagging_freq': 5,             # Frequency for bagging
        'verbose': -1,                 # Disable verbose output
        'seed': 42,                    # Random seed
        "device_type": "cuda"             # Enable GPU acceleration (requires GPU support)
    }
    params.update(lgb_params)  # Override with custom parameters
    
    # Convert data to memory-efficient format
    X_train = np.ascontiguousarray(X_train, dtype=np.float32)
    y_train = np.ascontiguousarray(y_train, dtype=np.float32)
    X_test = np.ascontiguousarray(X_test, dtype=np.float32)
    
    # Train LightGBM model with early stopping
    print("Training LightGBM model...")
    train_data = lgb.Dataset(X_train, label=y_train, free_raw_data=True)  # Release memory after conversion
    
    # Define a callback to log the loss
    gen_folder(os.path.dirname(log_file))
    def log_loss(env):
        with open(log_file, 'a') as f:
            current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            f.write(f"Iteration {env.iteration}: {env.evaluation_result_list} - {current_time}\n")
    
    model = lgb.train(
        params,
        train_data,
        num_boost_round=4000,          # Maximum number of boosting rounds
        valid_sets=[train_data],       # Use training set for validation
        callbacks=[
            lgb.early_stopping(
                stopping_rounds=50,    # Stop if no improvement for 50 rounds
                verbose=False          # Disable early stopping messages
            ),
            lgb.log_evaluation(period=10),
            log_loss                  # Custom callback to log loss
        ]
    )
    
    # Batch prediction for memory efficiency
    batch_num = int(np.ceil(len(X_test) / batch_size))
    y_pred = np.zeros(len(X_test), dtype=np.float32)
    
    # Process test data in batches
    for i in tqdm(range(batch_num), desc='LightGBM Prediction', mininterval=2):
        start_idx = i * batch_size
        end_idx = (i + 1) * batch_size
        
        # Handle last batch
        if i == batch_num - 1:
            end_idx = len(X_test)
            
        # Get current batch and predict
        X_batch = X_test[start_idx:end_idx]
        y_pred[start_idx:end_idx] = model.predict(
            X_batch,
            num_iteration=model.best_iteration  # Use best iteration from early stopping
        )
        
    return y_pred




class GPModel(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points):
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=True
        )
        super().__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        # self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(nu=1.5, ard_num_dims=2, active_dims=[1,2]) * 
            (gpytorch.kernels.PeriodicKernel(active_dims=[3]) * 
            gpytorch.kernels.RBFKernel(active_dims=[3])) + 
            gpytorch.kernels.RBFKernel(ard_num_dims=1, active_dims=[0])
        )
    
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

def gp_regression(X_train, y_train, X_test, num_epochs=95, learning_rate=0.01, n_inducing=500, best_model_path=None, induicng_sampling=0.01):
    """
    Perform Gaussian process regression.
    
    Args:
        X_train: Training features (n_samples, 4) [depth, lat, lon, time]
        y_train: Training target values (n_samples, 1)
        X_test: Test features (m_samples, 4)
        
    Returns:
        y_test: Predicted values (m_samples, 1)
    """
    gen_folder(os.path.dirname(best_model_path))

    # Convert to PyTorch tensors and ensure correct dimensions
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).squeeze()  # Ensure 1D target
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    
    # Feature scaling using training data statistics
    X_min = torch.min(X_train_tensor, dim=0).values
    X_max = torch.max(X_train_tensor, dim=0).values
    denominator = X_max - X_min
    denominator[denominator == 0] = 1e-8  # Avoid division by zero
    X_train_scaled = 2 * (X_train_tensor - X_min) / denominator - 1
    X_test_scaled = 2 * (X_test_tensor - X_min) / denominator - 1
    
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_train_scaled = X_train_scaled.to(device)
    y_train_tensor = y_train_tensor.to(device)
    X_test_scaled = X_test_scaled.to(device)
    
    # Create DataLoader for mini-batch training
    train_dataset = TensorDataset(X_train_scaled, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=100000, shuffle=True)
    
    # Initialize inducing points
    n_inducing = min(n_inducing, X_train_scaled.size(0))
    if n_inducing <= 0:
        raise ValueError("Insufficient training samples for inducing points.")
    X_train_np = X_train_scaled.cpu().numpy()
    X_train_np_sampled = proportional_sampling(X_train_np, induicng_sampling)
    kmeans = KMeans(
        n_clusters=n_inducing,
        init='k-means++',  
        max_iter=1000,      
        random_state=0,
        n_init=10,
        verbose=1,     
    )
    kmeans.fit(X_train_np_sampled)
    inducing_points = torch.tensor(
        kmeans.cluster_centers_, 
        dtype=torch.float32
    ).to(device)
    
    # Initialize model and likelihood
    model = GPModel(inducing_points=inducing_points)
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = model.to(device)
    likelihood = likelihood.to(device)
    
    # Configure optimizer
    optimizer = torch.optim.Adam([
        {'params': model.parameters()},
        {'params': likelihood.parameters()}
    ], lr=learning_rate)
    
    # Training loop
    model.train()
    likelihood.train()
    mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=y_train_tensor.size(0))
    
    # Initialize variables for tracking best model
    best_loss = float('inf')
    best_epoch = 0

    # Training loop with tqdm progress monitoring
    training_progress = tqdm(
        range(num_epochs), 
        desc="Total Training", 
        position=0,
        leave=True
    )

    for epoch in training_progress:
        # Inner progress bar for each epoch
        epoch_progress = tqdm(
            train_loader, 
            desc=f"Epoch {epoch+1}/{num_epochs}", 
            leave=False,                 
            position=1 
        )
        
        total_loss = 0  # Accumulator for total loss in the current epoch
        for batch_idx, (x_batch, y_batch) in enumerate(epoch_progress):
            optimizer.zero_grad()
            output = model(x_batch)
            loss = -mll(output, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            avg_loss = total_loss / (batch_idx + 1) 

            epoch_progress.set_postfix({
                'Batch Loss': f"{loss.item():.4f}",
                'Avg Loss': f"{avg_loss:.4f}" 
            })

        # Calculate epoch average loss
        epoch_avg_loss = total_loss / len(train_loader)
        training_progress.set_postfix({
            'Epoch Avg Loss': f"{epoch_avg_loss:.4f}"
        })

        # Save best model based on training loss
        if best_model_path is not None and epoch_avg_loss < best_loss:
            best_loss = epoch_avg_loss
            best_epoch = epoch + 1
            
            # Save model components to CPU
            model_state_dict = model.state_dict()
            model_state_dict_cpu = {k: v.cpu() for k, v in model_state_dict.items()}
            likelihood_state_dict = likelihood.state_dict()
            likelihood_state_dict_cpu = {k: v.cpu() for k, v in likelihood_state_dict.items()}
            inducing_points_cpu = inducing_points.cpu()
            
            torch.save({
                'model_state_dict': model_state_dict_cpu,
                'likelihood_state_dict': likelihood_state_dict_cpu,
                'inducing_points': inducing_points_cpu,
                'epoch': best_epoch,
                'loss': best_loss,
                'X_min': X_min,
                'X_max': X_max,
            }, best_model_path)

        epoch_progress.close()
    
    # Prediction phase
    model.eval()
    likelihood.eval()
    test_dataset = TensorDataset(X_test_scaled)
    test_loader = DataLoader(test_dataset, batch_size=100000, shuffle=False)
    
    predictions = []
    with torch.no_grad():
        for x_batch, in tqdm(test_loader, desc='Testing'):
            pred = likelihood(model(x_batch))
            predictions.append(pred.mean.cpu())
    
    # Combine predictions and format output
    y_test_pred = torch.cat(predictions, dim=0).numpy()
    return y_test_pred.reshape(-1, 1)


def predict_with_saved_model_gp(model_path, X_test_raw, device=None):
    """
    Make predictions using a saved GP model with automatic input scaling.
    
    Args:
        model_path: Path to saved model checkpoint
        X_test_raw: Raw test features (numpy array or DataFrame)
        device: Target device for computation (default: auto-detect)
        
    Returns:
        predictions: Numpy array of predicted values (shape: n_samples, 1)
    """
    # Load the complete model checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Device configuration
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Extract required components from checkpoint
    X_min = checkpoint['X_min'].to(device)
    X_max = checkpoint['X_max'].to(device)
    inducing_points = checkpoint['inducing_points'].to(device)
    
    # Initialize model and likelihood
    model = GPModel(inducing_points=inducing_points)
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    
    # Load trained parameters
    model.load_state_dict(checkpoint['model_state_dict'])
    likelihood.load_state_dict(checkpoint['likelihood_state_dict'])
    model = model.to(device)
    likelihood = likelihood.to(device)
    
    # Convert input data to tensor
    X_test_tensor = torch.tensor(X_test_raw, dtype=torch.float32).to(device)
    
    # Apply feature scaling using original training data parameters
    denominator = X_max - X_min
    denominator[denominator == 0] = 1e-8  # Prevent division by zero
    X_test_scaled = 2 * (X_test_tensor - X_min) / denominator - 1
    X_test_scaled = X_test_scaled.to(device)
    
    # Make predictions
    model.eval()
    likelihood.eval()
    with torch.no_grad():
        # Inside the prediction block:
        batch_size = 100000
        predictions = []
        for i in tqdm(range(0, len(X_test_scaled), batch_size), desc='Predicting'):
            batch = X_test_scaled[i:i+batch_size]
            pred = likelihood(model(batch))
            predictions.append(pred.mean.cpu().numpy())
        predictions = np.concatenate(predictions)
    
    return predictions.reshape(-1, 1)


def simulation_other_methods_inference(simulation_data_path='Data/Simulation/Train', new_x_path='Data/simulation_new_x.npy', save_path='Output/Prediction/Simulation/prediction.npy', model='gpr'):
    print('Other methods inference...')
    
    gen_folder(os.path.dirname(save_path))
    
    tau, w, v, p, x_z_t = SimulationData.simulation_read_data(simulation_data_path)
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



class OtherMethods():
    def __init__(self, data_loader=None):
        self.data_loader = data_loader
        self.plotter = RealDataPlot()


    def load_prediction_tasks(self, datasets=['argo', 'cur', 'wcur']):
        # Load and split data
        (x_argo, y_temp, y_sal, x_cur, y_v_theta, y_v_phi, 
        x_wcur, y_w), (x_argo_test, y_temp_test, y_sal_test,
                        x_cur_test, y_v_theta_test, y_v_phi_test,
                        x_wcur_test, y_w_test) = self._prepare_data(datasets)
        
        # Define all possible prediction tasks with simplified names
        prediction_tasks_all = [
            ('temp', x_argo, y_temp, x_argo_test, y_temp_test),      # Temperature prediction
            ('sal', x_argo, y_sal, x_argo_test, y_sal_test),         # Salinity prediction
            ('v_theta', x_cur, y_v_theta, x_cur_test, y_v_theta_test),  # Theta velocity
            ('v_phi', x_cur, y_v_phi, x_cur_test, y_v_phi_test),     # Phi velocity
            ('w', x_wcur, y_w, x_wcur_test, y_w_test)                # Vertical velocity
        ]

        return prediction_tasks_all

    def test(self, method='lightgbm', save_path='Output/Prediction/RealData/OtherMethods', tasks_to_run=None):
        """
        Test function compatible with LightGBM, Kriging and Sparse Gaussian Process methods.
        
        Args:
            method: Prediction method, either 'lightgbm', 'kriging' or 'sparse_gp'
            save_path: Path to save the results
            tasks_to_run: List of task names to run. If None, run all tasks
                        Valid tasks: ['temp', 'sal', 'v_theta', 'v_phi', 'w']
        """

        if tasks_to_run == ['temp', 'sal']:
            datasets = ['argo']
        elif tasks_to_run == ['w']:
            datasets = ['wcur']
        else:
            datasets = ['argo', 'cur', 'wcur']
        
        prediction_tasks_all = self.load_prediction_tasks(datasets)

        # Extract valid task names (first element of each tuple)
        valid_task_names = {task[0] for task in prediction_tasks_all}  # Now contains just strings

        # Filter tasks based on tasks_to_run
        if tasks_to_run is not None:
            # Validate input task names
            invalid_tasks = set(tasks_to_run) - valid_task_names
            if invalid_tasks:
                raise ValueError(f"Invalid task names: {invalid_tasks}. Valid tasks are: {list(valid_task_names)}")
            
            # Filter tasks using task names
            prediction_tasks = [task for task in prediction_tasks_all if task[0] in tasks_to_run]
        else:
            prediction_tasks = prediction_tasks_all

        # Check for empty task list
        if not prediction_tasks:
            print("Warning: No valid tasks selected. Available tasks:", valid_task_names)
            return

        # Ensure output directory exists
        gen_folder(save_path)

        # Unified prediction pipeline
        for task_name, X_train, y_train, X_test, y_test in prediction_tasks:
            # Method dispatch
            if method == 'lightgbm':
                y_pred = lightgbm_inference(
                    X_train, y_train, X_test,
                    log_file=f'Output/Model/RealData/OtherMethods/LightGBM/{task_name}_loss.log'
                )
            elif method == 'kriging':
                y_pred = spatiotemporal_kriging(X_train, y_train, X_test)
            elif method == 'sparse_gp':
                # Sparse GP with optimized parameters for ocean data
                y_pred = gp_regression(X_train, y_train, X_test, 
                                       best_model_path=f'Output/Model/RealData/OtherMethods/{method}/{task_name}_best.pth',
                                       n_inducing=GP_INDUCING_POINTS[task_name],
                                       induicng_sampling=GP_INDUCING_SAMPLING[task_name])
            else:
                raise ValueError(f"Unsupported method: {method}. Choose from ['lightgbm', 'kriging', 'sparse_gp']")
            
            # Calculate and display metrics
            rmse = root_mean_squared_error(y_test, y_pred)
            print(f'{task_name.ljust(8)} RMSE: {rmse:.4f}')

            self.plot_results(y_pred, X_test, var=task_name, method=method)
            
            # Save predictions
            np.save(os.path.join(save_path, f'{task_name}_pred.npy'), y_pred)

    
    def load_and_inference(self, method='SparseGP'):
        prediction_tasks_all = self.load_prediction_tasks()

        for task_name, X_train, y_train, X_test, y_test in prediction_tasks_all:

            y_pre = predict_with_saved_model_gp(model_path=f'Output/Model/RealData/OtherMethods/{method}/{task_name}_best.pth', 
                                    X_test_raw=X_test)
            print('task_name:', root_mean_squared_error(y_pre, y_test))

            gen_folder(f'Output/Prediction/RealData/OtherMethods/{method}')
            np.save(f'Output/Prediction/RealData/OtherMethods/{method}/{task_name}_pred.npy', np.squeeze(y_pre))


    def plot_results(self, y_pred, x_test, var='temp', method='lightgbm'):

        var_index = find_index_in_list(var, lst=VAR_NAMES)

        timestamp_1 = datetime(2021, 1, 1, 0, 0).timestamp()
        timestamp_2 = datetime(2021, 2, 1, 0, 0).timestamp()
        y_pred = y_pred[(x_test[:, 3] > timestamp_1) & (x_test[:, 3] < timestamp_2)]

        x_test = x_test[(x_test[:, 3] > timestamp_1) & (x_test[:, 3] < timestamp_2)]
        
        x_test = self.data_loader.normalization_coordinates(x_test)
        r, theta, phi, t = unpack_x(x_test)

        self.plotter.plot_global_(r, theta, phi, 
                            var=y_pred, 
                            vmin=None,
                            vmax=None,
                            cmap=VAR_COLORS[var_index], 
                            label=VAR_UNITS[var_index], 
                            extend=None, 
                            save_path=f'Output/Plot/RealData/OtherMethods/{method}/{var}.png', 
                            var_name=var)

    def _prepare_data(self, datasets=['argo', 'cur', 'wcur']):
        """return:
        (x_argo_train, y_temp_train, y_sal_train, 
        x_cur_train, y_v_theta_train, y_v_phi_train,
        x_wcur_train, y_w_train),
        (x_argo_test, y_temp_test, y_sal_test,
        x_cur_test, y_v_theta_test, y_v_phi_test,
        x_wcur_test, y_w_test)
        """
        # Load data
        train_data = self.data_loader.read_data(type='train', datasets=datasets, filter_bool=True)
        test_data = self.data_loader.read_data(type='test', datasets=datasets, filter_bool=True)
        
        # Split datasets
        train_splits = data_split(*train_data, datasets=datasets)
        test_splits = data_split(*test_data, datasets=datasets)
        
        return train_splits, test_splits
