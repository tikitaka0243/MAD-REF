<!-- <span style="color: grey;">_It is better to read in light appearance._</span> -->

<div align="right">
  
<span style="font-size: 12px; font-style: italic; opacity: 0.5;">
  
  _It is better to read in light appearance._
  
</span>

</div>


# Reconstructing and forecasting marine dynamic variable fields across space and time globally and gaplessly



![](/Plot/ResultEquator.jpg)



# Abstract

Spatiotemporal projections in marine are crucial for science and society, significantly enhancing the understanding of marine systems, protecting the marine environment, and guiding human activities in the ocean. Previous artificial intelligence (AI) and statistics-based inversion methods face limitations in leveraging ocean data, generating continuous inversion outputs, and incorporating informative physical constraints. In response to these challenges, we propose the Marine Dynamic Reconstruction and Forecast Neural Networks (MDRF-Net), which seamlessly integrates marine physical mechanisms from the primitive equations and multi-source observed data to reconstruct and forecast continuous global ocean temperature-salinity and dynamic fields. This mesh-free and easily implementable model enhances interpretability and enables the inversion of variable fields not directly from observed data, as well as exploring challenging-to-observe marine areas like the Arctic zone and forecasting oceanic variations for different future timeframe. Methodologically, MDRF-Net employs the ensemble method with multiple rotations of the Earth's coordinate system, a parallel neural network sharing the first layer, and a two-step training strategy to improve its performance in polar regions, accuracy, and training efficiency. MDRF-Net has undergone thorough spatiotemporal validation and outperforms other inversion methods in accuracy when compared to reanalysis data. It achieves notably low overall errors: 0.455 °C, 0.0714 psu, 4.254×10<sup>-6</sup>m/s, 0.0777 m/s, and 0.0825 m/s for vertical, northward, and eastward velocities, respectively. Overall, MDRF-Net is proficient at learning the ocean dynamics system utilizing physical mechanisms, and is helping create remarkable effects on Earth's climate and human use of the ocean.


## Keywords

Fields inversion | Global ocean dynamics | Primitive equations | Uncollected marine variables


# Data source and processing

We utilized ocean temperature and salinity data from the Argo project (https://argo.ucsd.edu/) and current reanalysis data from the EU Copernicus ocean service (https://www.copernicus.eu/en).

We found that a convenient way to download Argo data is to use Argo's data visualization tool, EuroArgo Selection Tool (https://dataselection.euro-argo.eu/), and select the csv format. For the EU Copernicus marine service, it is better to use FileZilla to connect to their server and download the relevant nc data files (see Copernicus website for details), which is faster. The dataset we use is 'GLOBAL_ANALYSISFORECAST_PHY_001_024', and the code 'cur' and 'wcur' in it represent the 3D ocean current reanalysis data.

After downloading the data, run the following function in the `Code/LoadData.py` file to process them.

```python
load_data(argo_data_path, argo_save_path, currents_data_path, currents_save_path, r_min, r_max, theta_min, theta_max, phi_min, phi_max, t_min, t_max, trian_vali_test=[8, 1, 1], ratio=1)
```

# Primitive equations neural networks (PENN)

![](/Plot/MDRF-NetStructure.pdf)

Above is the structure of PENN, run the following function to train the model.

```python
penn(data_path, r_min, r_max, ...)
```


# R Shiny platform

We have integrated all the variable prediction performances across different time and space into an interactive R Shiny platform (https://tikitakatikitaka.shinyapps.io/penn-shiny/). This platform facilitates users to access information about marine variables of interest across different time and spatial dimensions.

<img src="Plot/Shiny.png" alt="图像" width="400"/>