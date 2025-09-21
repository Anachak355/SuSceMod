# -*- coding: utf-8 -*-
"""
@author: Anasua Chakraborty
"""

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score
import rasterio
from tabulate import tabulate

def load_raster(file_path):
    
    with rasterio.open(file_path) as src:
        raster_data = src.read(1)  # Read the first band
        nodata_value = src.nodata  # Get the NoData value
    return raster_data, nodata_value

def calculate_kappa(map1, map2, nodata_values):

    # Flatten the maps to 1D arrays
    map1_flat = map1.flatten()
    map2_flat = map2.flatten()

    # Create a mask to ignore NoData values and filter NaN values
    valid_mask = (
        (map1_flat != nodata_values[0]) &
        (map2_flat != nodata_values[1]) &
        ~np.isnan(map1_flat) &
        ~np.isnan(map2_flat)
    )
    
    # Filter out NoData and NaN values
    map1_valid = map1_flat[valid_mask]
    map2_valid = map2_flat[valid_mask]

    # Calculate the Kappa coefficient
    kappa = cohen_kappa_score(map1_valid, map2_valid)
    
    return kappa

def save_results_table(results_df, output_path='kappa_results.txt'):
    
    table = tabulate(results_df, headers='keys', tablefmt='pretty')
    with open(output_path, 'w') as f:
        f.write(table)
    print(f"Results saved to {output_path}")

######################################################################################

# Set the parent folder path
parent_folder = '' # Please input here
results_file = parent_folder + 'kappa_results.txt'

zoning_types = ['none', 'hab', 'eco', 'aut']

results = []
for zoning_type in zoning_types:
    # Change the paths according to zoning. This code is just an example
    
    simulated_map_path = '' # Please input here
    observed_map_path = '' # Please input here

    simulated_map, sim_nodata = load_raster(simulated_map_path)
    observed_map, obs_nodata = load_raster(observed_map_path)
    kappa_value = calculate_kappa(simulated_map, observed_map, (sim_nodata, obs_nodata))
    results.append({
        'Zoning': zoning_type,
        'Kappa Coefficient': kappa_value
    })
results_df = pd.DataFrame(results)

# Print the results table
print(tabulate(results_df, headers='keys', tablefmt='pretty'))

# Save the results table to a file
save_results_table(results_df, results_file)