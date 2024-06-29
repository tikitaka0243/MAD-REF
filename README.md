<!-- <span style="color: grey;">_It is better to read in light appearance._</span> -->

<div align="right">
  
<span style="font-size: 12px; font-style: italic; opacity: 0.5;">
  
  _It is better to read in light appearance._
  
</span>

</div>


# Reconstructing and forecasting marine dynamic variable fields across space and time globally and gaplessly



![](/Image/Results.jpg)



# Abstract

Spatiotemporal projections in marine are crucial for science and society, significantly enhancing the understanding of marine systems, protecting the marine environment, and guiding human activities in the ocean. Previous artificial intelligence (AI) and statistics-based inversion methods face limitations in leveraging ocean data, generating continuous inversion outputs, and incorporating informative physical constraints. In response to these challenges, we propose the Marine Dynamic Reconstruction and Forecast Neural Networks (MDRF-Net), which seamlessly integrates marine physical mechanisms from the primitive equations and multi-source observed data to reconstruct and forecast continuous global ocean temperature-salinity and dynamic fields. This mesh-free and easily implementable model enhances interpretability and enables the inversion of variable fields not directly from observed data, as well as exploring challenging-to-observe marine areas like the Arctic zone and forecasting oceanic variations for different future timeframe. Methodologically, MDRF-Net employs the ensemble method with multiple rotations of the Earth's coordinate system, a parallel neural network sharing the first layer, and a two-step training strategy to improve its performance in polar regions, accuracy, and training efficiency. MDRF-Net has undergone thorough spatiotemporal validation and outperforms other inversion methods in accuracy when compared to reanalysis data. It achieves notably low overall errors: 0.455 °C, 0.0714 psu, 4.254×10<sup>-6</sup>m/s, 0.0777 m/s, and 0.0825 m/s for vertical, northward, and eastward velocities, respectively. Overall, MDRF-Net is proficient at learning the ocean dynamics system utilizing physical mechanisms, and is helping create remarkable effects on Earth's climate and human use of the ocean.


## Keywords

Fields inversion | Global ocean dynamics | Primitive equations | Uncollected marine variables


# Marine Dynamic Reconstruction and Forecast Neural Networks (MDRF-Net)
![MDRF-Net](/Image/MDRF-NetStructure.jpg)


# Simulation Study

We first explore the capabilities of MDRF-Net in a simulated system by considering a 2D simplified version of the primitive equations which has only one dimension in the horizontal direction ($x$) and does not include the diffusion equation for salinity as well as the equation of state. The simplified equations involve four variables: temperature ($\tau$), horizontal velocity ($v$), vertical velocity ($w$), and pressure ($p$). The domain is based on a Cartesian coordinate system and is dimensionless.

The simplified 2D primitive equations are given by:
$$
\begin{aligned}
\frac{\partial v}{\partial t} + v \frac{\partial v}{\partial x} + w \frac{\partial v}{\partial z} - \eta \frac{\partial^2 v}{\partial x^2} - \zeta \frac{\partial^2 v}{\partial z^2} + \frac{\partial p}{\partial x} &= 0, \\
\frac{\partial p}{\partial z} &= -\tau, \\
\frac{\partial v}{\partial x} + \frac{\partial w}{\partial z} &= 0, \\
\frac{\partial \tau}{\partial t} + v \frac{\partial \tau}{\partial x} + w \frac{\partial \tau}{\partial z} - \eta_{\tau} \frac{\partial^2 \tau}{\partial x^2} - \zeta_{\tau} \frac{\partial^2 \tau}{\partial z^2} &= Q,
\end{aligned}
$$
where this system admits a specific Taylor-Green vortex solution corresponding to a periodic source term $Q$.

The Taylor-Green vortex solution is as follows:
$$
\begin{aligned}
v &= -\sin(2\pi x)\cos(2\pi z)\exp\left[-4\pi^2(\eta+\zeta)t\right],\\
w &= \cos(2\pi x)\sin(2\pi z)\exp\left[-4\pi^2(\eta+\zeta)t\right], \\
p &= \frac{1}{4}\cos(4\pi x)\exp\left[-8\pi^2(\eta+\zeta)t\right] + \frac{1}{2\pi}\cos(2\pi z)\exp(-4\pi^2\zeta_{\tau}t),\\
\tau &= \sin(2\pi z)\exp(-4\pi^2\zeta_{\tau}t), \\
Q &= \pi\cos(2\pi x)\sin(4\pi z)\exp\left[-4\pi^2(\eta+\zeta+\zeta_{\tau})t\right].
\end{aligned}
$$

The results are presented below, with further details available in our paper.

![Simulation study](/Image/simulation_comparison.jpg)

# Work with Real Data

## Data source and processing

We utilized ocean temperature and salinity data from the Argo project (https://argo.ucsd.edu/) and current reanalysis data from the EU Copernicus ocean service (https://www.copernicus.eu/en).

We found that a convenient way to download Argo data is to use Argo's data visualization tool, EuroArgo Selection Tool (https://dataselection.euro-argo.eu/), and select the csv format. For the EU Copernicus marine service, it is better to use FileZilla to connect to their server and download the relevant nc data files (see Copernicus website for details), which is faster. The dataset we use is 'GLOBAL_ANALYSISFORECAST_PHY_001_024', and the code 'cur' and 'wcur' in it represent the 3D ocean current reanalysis data.

After downloading the data, run the following function in the `Code/LoadData.py` file to process them.

```python
load_data(argo_data_path, argo_save_path, currents_data_path, ...)
```

Parameters explained:
* `argo_data_path` and `argo_data_path`: Path of the raw Argo data and path to save the processed Argo data.
* `currents_data_path` and `currents_data_path`: Path of the raw currents data and path to save the processed currents data.
* `r_min` and `r_max`: The minimum and maximum values of the water depth of the ocean variable fields in meters, e.g., -2000 and 0.
* `theta_min` and `theta_max`: The minimum and maximum values of the latitude of the ocean variable field in degrees, e.g. -20 and 60.
* `phi_min` and `phi_max`: The minimum and maximum values of the longitude of the ocean variable field in degrees, e.g. -120 and 170.
* `t_min` and `t_max`: Minimum and maximum values of the ocean variable field time in the following format 'YYYY-MM-DDTHH:MM:SSZ', e.g. '2021-05-18T05:42:37Z'.
* `train_vali_test`: The ratio of the training set, validation set, and test set, e.g. [8, 1, 1].

## Train MDRF-Net

Above is the structure of MDRF-Net, run the following function to train the model.

```python
mdrf_net(data_path, r_min, r_max, ...)
```

Parameters explained:
* `data_path`: Path to data storage.
* `r_min` and `r_max`: The minimum and maximum values of the water depth of the ocean variable fields in meters, e.g., -2000 and 0.
* `theta_min` and `theta_max`: The minimum and maximum values of the latitude of the ocean variable field in degrees, e.g. -20 and 60.
* `phi_min` and `phi_max`: The minimum and maximum values of the longitude of the ocean variable field in degrees, e.g. -120 and 170.
* `t_min` and `t_max`: Minimum and maximum values of the ocean variable field time in the following format 'YYYY-MM-DDTHH:MM:SSZ', e.g. '2021-05-18T05:42:37Z'.
* `r_num`, `theta_num` and `phi_num`: The density of sampling points used for calculating equation errors across water depth, latitude, and longitude.
* `batch_size`: The batch size for each variable's observed data points.
* `init_beta_tau` and `init_beta_sigma`: The initial values of the parameters $\beta_\tau$ and $\beta_\sigma$ in the primitive equations.
* `num_domain` and `num_boundary`: The number of sampling points on the domain and along the boundary used for computing the equation error.
* `input_output_size` The number of inputs and outputs of the network.
* `n_layers`: Each parallel subnet's number of layers.
* `activation`: Activation function.
* `initializer`: Parameters initialize method.
* `model_save_path_1` and `model_save_path_2`: The storage paths for models in the two training phases.
* `variable_save_path`: The storage path for the model's unknown parameters.
* `save_period`: The storage interval for the model and its parameters, measured in iterations.
* `resample_period_1` and `resample_period_2`: The resampling interval for observational data and sampling points, measured in iterations.
* `num_iter_1` and `num_iter_2`: The total number of iterations for each stage in the two-stage training separately.
* `optimizer`: Optimization algorithm.
* `learning_rate_1` and `learning_rate_2`: The initial learning rates for the two training stages.
* `loss_weights_1` and `loss_weights_2`: The internal weights of the loss function for the two training stages.


# R Shiny platform

We have integrated all the variable prediction performances across different time and space into an interactive R Shiny platform (https://tikitakatikitaka.shinyapps.io/mdrf-net-shiny/). This platform facilitates users to access information about marine variables of interest across different time and spatial dimensions.

<img src="Image/Shiny.png" alt="Shiny Image" width="400"/>