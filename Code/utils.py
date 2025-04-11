import fnmatch
import os
from typing import Union, List
from datetime import datetime, timezone
from dateutil.relativedelta import relativedelta

import numpy as np
import pandas as pd


LINESTYLES = ['-', '--', ':', '-.']
VARS = ['tau', 'w', 'v', 'p']

TASK_COLORS = [
    'tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 
    'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan'
]
METHOD_COLORS = {'MAD-REF Net': 'tab:blue', 'Neural Field': 'tab:orange', 'LightGBM': 'orchid', 'SparseGP': 'lightseagreen'}
METHOD_MARKERS = {'MAD-REF Net': 'o', 'Neural Field': 's', 'LightGBM': 'D', 'SparseGP': 'p'}
WARM_COLORS = [
    '#EF476F', '#FF6F61', '#FFA177', '#FFD166', '#FF9F1C',
    '#E76F51', '#F4A261', '#E9C46A', '#D62828', '#F77F00'
]
PURPLE_COLORS = [
    (78 / 255, 101 / 255, 155 / 255),  # (0.3059, 0.3961, 0.6078)
    (138 / 255, 140 / 255, 191 / 255),  # (0.5412, 0.5490, 0.7490)
    (184 / 255, 168 / 255, 207 / 255),  # (0.7216, 0.6588, 0.8118)
    (231 / 255, 188 / 255, 198 / 255)   # (0.9059, 0.7373, 0.7765)
]

VAR_COLORS = ['plasma', 'viridis', 'RdYlGn', 'RdGy', 'RdBu', 'inferno_r', 'magma']
VAR_UNITS = ['Temperature (°C)', 'Salinity (psu)', 'Vertical Velocity (m/s)', 'Northward Velocity (m/s)', 'Eastward Velocity (m/s)', 'Pressure (N/m^2)', 'Density (kg/m^3)']
VAR_NAMES = ['temp', 'sal', 'w', 'v_theta', 'v_phi', 'pres', 'dens']
VMIN = [None, None, -3e-5, -0.4, -1.4, None, None]
VMAX = [None, None, 3e-5, 0.4, 1.4, None, None]
VMIN_GLOBAL = [-2, 31.730, -2e-5, -0.4, -0.8, 0, 1016]
VMAX_GLOBAL = [32.437, 37.965, 2e-5, 0.4, 0.8, 20540000, 1032]
EXTEND = [None, None, 'both', 'both', 'both', None, None]
EXTEND_GLOBAL = ['both', 'both', 'both', 'both', 'both', None, None]
VMIN_VMAX_SEAICE = [0.0007629627361893654, 6.401257356628776]
EXTEND_SEAICE = None

SLICE_FIGSIZE = [(10, 4), (10, 2), (4, 2)]
SLICE_COOR_NAMES = ['depth', 'latitude', 'longitude']
SLICE_LAT_UNIT = ['20°S', '10°S', '0°', '10°N', '20°N']
SLICE_LONG_UNIT = ['150°E', '160°E', '170°E', '180°', '170°W', '160°W', '150°W', '140°W', '130°W', '120°W', '110°W']
SLICE_LAT_UNIT_GLOBAL = [f"{abs(lat)}°S" if lat < 0 else f"{lat}°N" if lat > 0 else "0°" for lat in range(-70, 81, 10)]
SLICE_LONG_UNIT_GLOBAL = ['180°', '170°W', '160°W', '150°W', '140°W', '130°W', '120°W', '110°W', '100°W', '90°W', '80°W', '70°W', '60°W', '50°W', '40°W', '30°W', '20°W', '10°W', '0°', '10°E', '20°E', '30°E', '40°E', '50°E', '60°E', '70°E', '80°E', '90°E', '100°E', '110°E', '120°E', '130°E', '140°E', '150°E', '160°E', '170°E']

DATA_PATH = ['Argo/argo', 'Currents/wcur', 'Currents/cur']
TIME_LIST = ["2021-01-16T12:00:00Z", 
             "2021-07-16T12:00:00Z", 
             "2022-01-16T12:00:00Z", 
             "2022-07-16T12:00:00Z", 
             "2023-01-16T12:00:00Z"]

RHO_0 = 1027
TAU_MEAN = 8.833255471380031
SIGMA_MEAN = 34.81696024350364
TAU_STD = 7.453051337599042
SIGMA_STD = 0.6527534642839026
BETA_TAU = -0.9988
BETA_SIGMA = 1.0349E-01

GP_INDUCING_POINTS = {'temp': 500, 'sal': 500, 'w': 1000, 'v_theta': 1000, 'v_phi': 1000}
GP_INDUCING_SAMPLING = {'temp': 0.01, 'sal': 0.01, 'w': 0.001, 'v_theta': 0.001, 'v_phi': 0.001}

def unpack_x_sim(X):
    x = X[:, 0:1]
    z = X[:, 1:2]
    t = X[:, 2:3]
    return x, z, t

def unpack_x(X):
    r = X[:, 0:1]
    theta = X[:, 1:2]
    phi = X[:, 2:3]
    t = X[:, 3:4]
    return r, theta, phi, t


def unpack_y_sim(Y):
    tau = Y[:, 0:1]
    w = Y[:, 1:2]
    v = Y[:, 2:3]
    p = Y[:, 3:4]
    if Y.shape[1] == 5:
        q = Y[:, 4:5]
        return tau, w, v, p, q
    return tau, w, v, p

def unpack_y(Y):
    tau = Y[:, 0:1]
    sigma = Y[:, 1:2]
    w = Y[:, 2:3]
    v_theta = Y[:, 3:4]
    v_phi = Y[:, 4:5]
    p = Y[:, 5:6]
    
    return tau, sigma, w, v_theta, v_phi, p


def gen_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)
            
            
def gen_folder_2steps(path):
    if not os.path.exists(path):
        os.makedirs(path)
        os.mkdir(os.path.join(path, 'Step1'))
        os.mkdir(os.path.join(path, 'Step2'))


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


def simulation_data_domain_check(x, z):
    x = (x - 0.5) * 2
    z = (z - 0.5) * 2
    
    eq = np.power(np.abs(x), 3) + np.power(np.abs(z), 3)
    output = eq <= 0.7 ** 3
    
    return output


def ndarray_check(array):
    pd.set_option('display.max_columns', None)
    print(array, '\n', pd.DataFrame(array).describe())


def prediction_normalization(prediction, new_x):
    t = new_x[:, 2]
    unique_t = np.unique(t)
    for t_ in unique_t:
        indices = t == t_
        prediction[indices] -= np.mean(prediction[indices], axis=0)
        
    return prediction


import torch

def linear_normalize(x: torch.Tensor, 
                    dim: int = None, 
                    keepdim: bool = False,
                    eps: float = 1e-8) -> torch.Tensor:
    """
    Perform linear normalization (sum-to-one scaling) on a tensor.
    
    Args:
        x (torch.Tensor): Input tensor to be normalized
        dim (int, optional): Dimension along which to normalize. 
            If None, normalize over all elements. Default: None
        keepdim (bool): Whether to retain the reduced dimension in the output. 
            Default: False
        eps (float): Small epsilon value to prevent division by zero. 
            Default: 1e-8

    Returns:
        torch.Tensor: Normalized tensor with same shape as input 
            (dimension sizes may differ if keepdim=False)

    Features:
        - Gradient-friendly: Maintains computational graph for backpropagation
        - Numerically stable: Automatic protection against zero-division
        - Dimension-aware: Supports arbitrary normalization dimensions
        - Broadcast-compatible: Preserves shape consistency for element-wise ops
    """
    # Compute sum along specified dimension (keep dim for broadcasting)
    sum_x = torch.sum(x, dim=dim, keepdim=True)
    
    # Clamp to prevent division by zero (maintains gradient flow)
    sum_x = sum_x.clamp(min=eps)
    
    # Perform normalization
    normalized = x / sum_x
    
    # Squeeze dimension if keepdim=False
    if not keepdim and dim is not None:
        normalized = normalized.squeeze(dim)
    
    return normalized

def convert_to_timestamp(date_input: Union[str, List[str]]) -> Union[int, List[int]]:
    """
    Convert a date string or a list of date strings in the format YYYY-MM-DDTHH:MI:SSZ
    to Unix timestamps (seconds since Jan 1, 1970).

    Args:
        date_input (Union[str, List[str]]): A single date string or a list of date strings.

    Returns:
        Union[int, List[int]]: A single timestamp or a list of timestamps.
    """
    def _convert_single(date_str: str) -> int:
        """Convert a single date string to a Unix timestamp."""
        # Parse the string into a datetime object and set the timezone to UTC
        dt = datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        # Convert the datetime object to a Unix timestamp (seconds)
        return float(dt.timestamp())

    # Handle single string input
    if isinstance(date_input, str):
        return _convert_single(date_input)
    # Handle list of strings input
    elif isinstance(date_input, list):
        return [_convert_single(date_str) for date_str in date_input]
    # Raise an error for invalid input types
    else:
        raise TypeError("Input must be a string or a list of strings")
    
def normalize_t(t, t_min):
    
    a = 6.4 * 10 ** 6
    U = 10 ** -1

    is_list = isinstance(t, list)
    t = np.asarray(t)
    t -= t_min
    result = t * U / a

    return result.tolist() if is_list else result

def data_old2new(old_path, new_path):

    df_old = pd.read_csv(old_path).values


    df_old = old2new_(df_old)
    
    ndarray_check(df_old)

    np.save(new_path, df_old)

def data_old2new_global_argo(old_path, new_path):
    df_old = np.load(old_path)

    df_old = old2new_(df_old)

    # convert to float32
    df_old = df_old.astype(np.float32)
    np.save(new_path, df_old)


def old2new_(df_old):
    for j in range(1, 3):
        df_old[:, j] = np.rad2deg(df_old[:, j])

    time_range = convert_to_timestamp("2023-01-01T00:00:00Z") - convert_to_timestamp("2021-01-01T00:00:00Z")
    df_old[:, 3] = df_old[:, 3] * time_range + convert_to_timestamp("2021-01-01T00:00:00Z")

    return df_old




def data_old2new_2():

    gen_folder('Data/RealData/Local/Argo')

    for data_type in ['all', 'train', 'test', 'vali']:
        print(data_type)
        data_old2new(old_path=f'/public/home/tianting/XiongZhixi/PINN_equator/Data/Equator_all/Argo/argo_{data_type}.csv', 
                     new_path=f'Data/RealData/Local/Argo/argo_{data_type}.npy')
        
    # gen_folder('Data/RealData/Local/Currents')

    # for data_type in ['vali']:
    #     print(data_type)
    #     data_old2new(old_path=f'/public/home/tianting/XiongZhixi/PINN_equator/Data/Equator_all/Currents_m/wcur_{data_type}.csv', 
    #                  new_path=f'Data/RealData/Local/Currents/wcur_{data_type}.npy')
        

def data_old_split(data_path='Data/RealData/Local/Argo/argo_all.npy', train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, random_seed=42):
    """
    Splits the dataset into training, validation, and test sets according to the specified ratios,
    and exports the split datasets to files.

    Args:
        data_path (str): Path to the data file.
        train_ratio (float): Ratio of the training set, default is 0.8.
        val_ratio (float): Ratio of the validation set, default is 0.1.
        test_ratio (float): Ratio of the test set, default is 0.1.
        random_seed (int): Random seed for reproducibility.

    Returns:
        None
    """
    # Check if the ratios sum to 1
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("The sum of train_ratio, val_ratio, and test_ratio must be 1.")

    # Load the data
    data = np.load(data_path)

    # Shuffle the data randomly
    np.random.seed(random_seed)
    np.random.shuffle(data)

    # Calculate the split points
    num_samples = len(data)
    train_end = int(num_samples * train_ratio)
    val_end = train_end + int(num_samples * val_ratio)

    # Split the dataset
    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]

    # Export the datasets
    data_dir = os.path.dirname(data_path)
    train_file = os.path.join(data_dir, f'argo_train.npy')
    val_file = os.path.join(data_dir, f'argo_vali.npy')
    test_file = os.path.join(data_dir, f'argo_test.npy')

    # Convert data to float32 before saving
    np.save(train_file, train_data.astype(np.float32))
    np.save(val_file, val_data.astype(np.float32))
    np.save(test_file, test_data.astype(np.float32))

    # ndarray_check(train_data)
    # ndarray_check(val_data)
    # ndarray_check(test_data)

    print(f"Dataset split completed:")
    print(f"Training set size: {len(train_data)}")
    print(f"Validation set size: {len(val_data)}")
    print(f"Test set size: {len(test_data)}")

def proportional_sampling(data, proportion, replace=False, random_state=42):

    if random_state is not None:
        np.random.seed(random_state)

    sample_size = int(data.shape[0] * proportion)
    indices = np.random.choice(data.shape[0], size=sample_size, replace=replace)
    sampled_data = data[indices]

    return sampled_data

def find_closest_vector(list, value):
    min_value = min(list)
    loc = np.copy(value)
    value_180 = loc == 180
    loc[value_180] = 0
    loc[~value_180] = np.round((loc[~value_180] - min_value) / (1 / 12))

    return loc.astype(np.int32)


def get_all_files(folder_path):
    """
    Returns a list of full paths for all files in the specified folder, including files in subdirectories.
    Args:
        folder_path (str): The path to the folder to search.
    Returns:
        list: A list of full file paths.
    """
    file_list = []
    # Use os.walk to recursively traverse the directory tree
    for root, _, files in os.walk(folder_path):
        # Iterate through each file in the current directory
        for file in files:
            # Construct the full file path and add it to the list
            file_path = os.path.join(root, file)
            file_list.append(file_path)
    return file_list

import re

def replace_group_name(reinitialize_group):
    """
    Replace 'v_theta' with 'v1' and 'v_phi' with 'v2' to match parameter names.
    
    Args:
        reinitialize_group (str): The group name to be replaced.
    
    Returns:
        str: The replaced group name.
    """
    if reinitialize_group == 'v_theta':
        return 'v1'
    elif reinitialize_group == 'v_phi':
        return 'v2'
    else:
        return reinitialize_group  # Return the original value if no match is found


def find_index_in_list(target, lst=VAR_NAMES):

    try:
        return lst.index(target)
    except ValueError:
        return -1

def data_split(data_argo, data_cur, data_wcur, datasets=['argo', 'cur', 'wcur']):
    x_argo, y_temp, y_sal, x_cur, y_v_theta, y_v_phi, x_wcur, y_w = None, None, None, None, None, None, None, None

    if 'argo' in datasets:
        x_argo = data_argo[:, 0:4]
        y_temp = data_argo[:, 4]
        y_sal = data_argo[:, 5]
    if 'cur' in datasets:
        x_cur = data_cur[:, 0:4]
        y_v_theta = data_cur[:, 4]
        y_v_phi = data_cur[:, 5]
    if 'wcur' in datasets:
        x_wcur = data_wcur[:, 0:4]
        y_w = data_wcur[:, 4]

    return x_argo, y_temp, y_sal, x_cur, y_v_theta, y_v_phi, x_wcur, y_w

def generate_monthly_dates(start_year, start_month, end_year, end_month):
    dates = []
    current_date = datetime(start_year, start_month, 1, 0, 0, 0)
    end_date = datetime(end_year, end_month, 1, 0, 0, 0)
    
    while current_date <= end_date:
        dates.append(current_date.strftime("%Y-%m-%dT%H:%M:%SZ"))
        current_date += relativedelta(months=1)
    
    return dates


TIME_LIST_SHINY = generate_monthly_dates(2021, 1, 2023, 1)


def create_image_grid(image_paths, cols=None, max_width=None):
    """
    Combine multiple images into a grid layout without spacing between them.
    
    Args:
        image_paths: List[str], paths to input image files
        cols: Optional[int], number of columns in the grid. If None, calculated automatically
        max_width: Optional[int], maximum width of output image. Used to auto-calculate columns if cols=None
    
    Returns:
        PIL.Image object containing the grid arrangement of all input images
    
    Raises:
        ValueError: if image_paths is empty
    """
    if not image_paths:
        raise ValueError("Image list cannot be empty")
    
    # Open all images and get their dimensions
    images = [Image.open(path) for path in image_paths]
    widths, heights = zip(*(img.size for img in images))
    
    # Determine grid layout
    num_images = len(images)
    if cols is None:
        if max_width is not None:
            # Calculate columns based on desired maximum width
            avg_width = sum(widths) / num_images
            cols = max(1, math.floor(max_width / avg_width))
        else:
            # Default to square-like layout
            cols = math.ceil(math.sqrt(num_images))
    
    rows = math.ceil(num_images / cols)
    
    # Calculate cell dimensions (maximum width/height in each grid cell)
    cell_width = max(widths)
    cell_height = max(heights)
    
    # Calculate output image dimensions
    output_width = cell_width * cols
    output_height = cell_height * rows
    
    # Create blank output image
    output_img = Image.new('RGB', (output_width, output_height))
    
    # Paste each image into its grid position
    for i, img in enumerate(images):
        x = (i % cols) * cell_width  # column position
        y = (i // cols) * cell_height  # row position
        output_img.paste(img, (x, y))
    
    return output_img

    