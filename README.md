


# From Sparse Spatio-temporal Data to Dynamic Ocean Fields: A Physics-Guided Neural Framework for Global Marine Predictions
Zhixi Xiong, Yukang Jiang, Wenfang Lu, Xueqin Wang, and Ting Tian





## Abstract

The reconstruction of continuous ocean temperature-salinity and dynamic fields from sparse, heterogeneous observations is a critical challenge in marine science, with profound implications for climate prediction, resource management, and biodiversity conservation. 
Despite recent advances, existing methods struggle to reconcile sparse observations with physical constraints, particularly in underobserved regions like the deep ocean and polar zones.
To address these problems, we propose the Marine Dynamic Reconstruction and Forecast Neural Networks (MAD-REF Net), which integrates marine physical mechanisms and observed data to reconstruct and forecast continuous ocean temperature-salinity and dynamic fields.
Notably, MAD-REF Net   is capable of reconstructing variables in regions lacking observational data, such as polar areas, and can infer unobserved variables, thereby providing a comprehensive understanding of marine dynamics across both data-rich and data-sparse regions.
MAD-REF Net   leverages statistical theories and techniques to eliminate polar singularities, ensure global continuity, model multi-scale features, and enhance training efficiency while balancing data-physics alignment.
Overall, MAD-REF Net   effectively learns the ocean dynamics system using physical mechanisms and statistical insights, contributing to a deeper understanding of marine systems and their impact on the environment and human use of the ocean.


### Keywords

Fields inversion | Global ocean dynamics | Primitive equations | Uncollected marine variables


## Marine Dynamic Reconstruction and Forecast Neural Networks (MAD-REF Net)
![MDRF-Net](/Image/MAD-REF_structure.jpg)


## Real Data Source

We utilized ocean temperature and salinity data from the Argo project (https://argo.ucsd.edu/) and current reanalysis data from the EU Copernicus ocean service (https://www.copernicus.eu/en).

We found that a convenient way to download Argo data is to use Argo's data visualization tool, EuroArgo Selection Tool (https://dataselection.euro-argo.eu/), and select the CSV format. For the EU Copernicus marine service, it is better to use FileZilla to connect to their server and download the relevant NC data files (see Copernicus website for details), which is faster. The dataset we use is 'GLOBAL_ANALYSISFORECAST_PHY_001_024', and the codes 'cur' and 'wcur' in them represent the 3D ocean current reanalysis data.

## Installation
```
conda env create -f environment.yml
```

## Train the Model
```
python Simulation.py
```

## R Shiny platform

We have integrated all the variable prediction performances across different times and spaces into an interactive R Shiny platform (https://tikitakatikitaka.shinyapps.io/mdrf-net-shiny/). This platform facilitates users to access information about marine variables of interest across different time and spatial dimensions.

<img src="Image/Shiny.png" alt="Shiny Image" width="400"/>