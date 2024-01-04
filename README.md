# PENN

**Primitive Equations Neural Networks**


This model is from the paper 'Inversion of 3D marine variable fields with primitive equations neural network'.

## Abstract

Marine variable fields inversion contributes to the study of marine ecosystems and the guidance of human marine activities, whereas previous AI- or statistics-based methods have many limitations in data usage and inversion result output, as well as lacking physical constraints. Therefore, we proposed a mesh-free and easy-to-implement model for the inversion of ocean temperature-salinity and dynamic fields. It embeds the primitive equations that can describe the ocean current motion as well as the diffusion of temperature and salinity, which brings high interpretability and enables the inversion of variable fields without observed data, and it adopts a parallel neural network sharing the first layer and a two-step training strategy to improve efficiency and accuracy. We validated our model in both local and global scenario and compared the inversion results with reanalysis data and other inversion methods, and it demonstrated favourable performance consistently.

## Keywords

Ocean dynamics, Fields inversion, Primitive equations, Deep learning.


# Data source and processing

We utilized ocean temperature and salinity data from the Argo project (https://argo.ucsd.edu/) and current reanalysis data from the EU Copernicus ocean service (https://www.copernicus.eu/en).

We found that a convenient way to download Argo data is to use Argo's data visualization tool, EuroArgo Selection Tool (https://dataselection.euro-argo.eu/), and select the csv format. For the EU Copernicus marine service, it is better to use FileZilla to connect to their server and download the relevant nc data files (see Copernicus website for details), which is faster. The dataset we use is 'GLOBAL_ANALYSISFORECAST_PHY_001_024', and the code 'cur' and 'wcur' in it represent the 3D ocean current reanalysis data. You can download the global ocean-wide data for the whole year of 2021 to 2022, and our code will process them.

Once the download is finished, you can run the file LoadData.py in the data folder.