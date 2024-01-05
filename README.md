# Primitive Equations Neural Networks

**Primitive Equations Neural Networks**


This model is from the paper 'Local and global dynamic marine variable fields inversion with primitive equations neural networks'.

## Abstract

Marine variable fields inversion contributes to the study of marine systems and guides human activities in the marine domain. Previous artificial intelligence (AI)- and statistics-based methods have limitations in utilizing ocean data, producing inversion outputs, and incorporating physical constraints. Therefore, we propose the Primitive Equations Neural Networks (PENN), a mesh-free and easy-to-implement model for the inversion of ocean temperature-salinity and dynamic fields. This approach enhances interpretability and enables the inversion of variable fields for unobserved data. We validate PENN in both local (equatorial Pacific) and global scenarios, comparing the inversion outcomes with competing reanalysis data and other inversion methods. Throughout the evaluations, PENN consistently demonstrates superior performance, where the overall errors for temperature, salinity, vertical velocity, northward velocity, and eastward velocity fields are 0.455 °C, 0.0714 psu, 4.254 $\times 10^{-6}$ m/$\text{s}^2$, 0.0777 m/$\text{s}^2$, and 0.0825 m/$\text{s}^2$ respectively for the global ocean. Therefore, by providing continuous global inversion marine fields across different time and locations, PENN can accurately identify phenomena such as the Mediterranean Salinity Crisis and the North Atlantic Warm Current. Furthermore, by precisely learning ocean change patterns, PENN can effectively predict future ocean changes and challenging-to-observe marine areas, including the Arctic zone.

$$
\theta
$$

## Keywords

Ocean dynamics, Fields inversion, Primitive equations, Deep learning.


# Data source and processing

We utilized ocean temperature and salinity data from the Argo project (https://argo.ucsd.edu/) and current reanalysis data from the EU Copernicus ocean service (https://www.copernicus.eu/en).

We found that a convenient way to download Argo data is to use Argo's data visualization tool, EuroArgo Selection Tool (https://dataselection.euro-argo.eu/), and select the csv format. For the EU Copernicus marine service, it is better to use FileZilla to connect to their server and download the relevant nc data files (see Copernicus website for details), which is faster. The dataset we use is 'GLOBAL_ANALYSISFORECAST_PHY_001_024', and the code 'cur' and 'wcur' in it represent the 3D ocean current reanalysis data. You can download the global ocean-wide data for the whole year of 2021 to 2022, and our code will process them.

Once the download is finished, you can run the file LoadData.py in the data folder.