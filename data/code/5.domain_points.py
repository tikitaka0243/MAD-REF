import numpy as np
import pandas as pd


print('############## LoadData4-Generate domain points ###############')

# Define the bounds and step sizes for the spherical coordinate grid
r_min, r_max, r_step = 0, 1, 0.05
theta_min, theta_max, theta_step = -20 / 180 * np.pi, 20 / 180 * np.pi, 0.5 / 180 * np.pi
phi_min, phi_max, phi_step = -50 / 180 * np.pi, 50 / 180 * np.pi, 0.5 / 180 * np.pi

# Generate the spherical coordinate grid
r, theta, phi = np.meshgrid(np.arange(r_min, r_max + r_step, r_step),
                            np.arange(theta_min, theta_max + theta_step, theta_step),
                            np.arange(phi_min, phi_max + phi_step, phi_step))

# Convert the spherical coordinates to a flattened array and export to CSV
df = pd.DataFrame({'r': r.ravel(), 'theta': theta.ravel(), 'phi': phi.ravel()})
df = df[~(df == 0).any(axis=1)]
df.to_csv('domain_points.csv', index=False)
