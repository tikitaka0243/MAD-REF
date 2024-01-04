import math
import pandas as pd
import numpy as np
# import os, platform, subprocess, re
# # import random


# pd.set_option('display.width', 1000)
# pd.options.display.max_columns = 40


raw_argo_data = pd.read_csv('Argo_pre/raw_argo_data.csv')
argo_all = raw_argo_data[['PRES_ADJUSTED (decibar)', 'LATITUDE (degree_north)', 'LONGITUDE (degree_east)', 'DATE (YYYY-MM-DDTHH:MI:SSZ)',  'TEMP_ADJUSTED (degree_Celsius)', 'PSAL_ADJUSTED (psu)']].dropna()
argo_all.columns = ['r', 'theta', 'phi', 't', 'temp', 'sal']

timestamp = [1514764800, 1546300800, 1577836800, 1609459200, 1640995200, 1672531200]
argo_all['t'] = pd.to_datetime(argo_all['t']).dt.tz_convert(None)
argo_all['t'] = argo_all['t'].values.astype(np.int64) / 10 ** 9

argo_all['r'] = - argo_all['r']
argo_all['theta'] = argo_all['theta'] / 180 * math.pi
argo_all['phi'] = argo_all['phi'] / 180 * math.pi

argo_all.to_csv('Argo_pre/argo_all.csv', index=False)


