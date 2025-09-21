# -*- coding: utf-8 -*-
"""
@author: Anasua Chakraborty
"""

import numpy as np
import rasterio

def create_matrix_map(simulated_map_path, observed_map_path, output_map_path):

    with rasterio.open(observed_map_path) as obs_src:
        observed_map = obs_src.read(1)  # Read the first band
        obs_meta = obs_src.meta
    
    with rasterio.open(simulated_map_path) as sim_src:
        simulated_map = sim_src.read(1)  # Read the first band
    
    # Ensure both maps have the same shape
    assert observed_map.shape == simulated_map.shape, "Error: The maps do not have the same shape!"
    
    comparison_map = np.full_like(observed_map, np.nan, dtype=np.float32)
    valid_mask = ~np.isnan(observed_map) & ~np.isnan(simulated_map)
    
    # True Negatives
    comparison_map[(observed_map == 0) & (simulated_map == 0) & valid_mask] = 10
    
    # True Positives
    comparison_map[(observed_map == simulated_map) & (observed_map > 0) & valid_mask] = 40
    
    # False Positives
    comparison_map[(observed_map == 0) & (simulated_map > 0) & valid_mask] = 20
    
    # False Negatives
    comparison_map[(observed_map > 0) & (simulated_map == 0) & valid_mask] = 30
    
    obs_meta.update(dtype=rasterio.float32, count=1, nodata=np.nan)
    
    # Save the comparison map to a new file
    with rasterio.open(output_map_path, 'w', **obs_meta) as dst:
        dst.write(comparison_map, 1)
    
    print(f"Comparison map saved successfully at: {output_map_path}")

#####################################################################

parent_folder = '' # Please input here
zoning_type = 'hab' # 'hab' 'eco' 'aut' 'none'

# Change the paths according to zoning. This code is just an example
simulated_map_path = '' # Please input here
observed_map_path = '' # Please input here
output_map_path = '' # Please input here

create_matrix_map(simulated_map_path, observed_map_path, output_map_path)