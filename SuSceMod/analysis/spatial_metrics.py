# -*- coding: utf-8 -*-
"""
@author: Anasua Chakraborty
"""

import rasterio
import numpy as np
import pandas as pd
from skimage.measure import label
from math import log2


def read_raster(file_path):

    with rasterio.open(file_path) as src:
        raster = src.read(1)
        # Mask the NaN values to ignore them during the calculations
        valid_raster = np.ma.masked_invalid(raster)  # Mask NaN values
        return valid_raster, src.transform
    
def calculate_urban_sprawled_index(raster):

    non_nan_cells = np.count_nonzero(~raster.mask)
    unique, counts = np.unique(raster.compressed(), return_counts=True)
    proportions = counts / non_nan_cells
    usi = -np.sum(proportions * np.log(proportions))
    return usi

def calculate_shannon_entropy(raster, num_classes=4):

    total_cells = raster.size
    entropy = 0
    for class_value in range(num_classes):

        mask = raster == class_value
        valid_cells = np.ma.masked_array(raster, ~mask)
        
        count = np.count_nonzero(valid_cells.mask == False)
        if count > 0:
            p = count / total_cells
            entropy -= p * log2(p)
    
    return entropy

def calculate_average_contiguity(raster, class_value):

    mask = (raster == class_value).astype(np.uint8)
    labeled_mask, num_labels = label(mask, connectivity=1, return_num=True)
    total_contiguity = 0
    valid_patches = 0

    for label_id in range(1, num_labels + 1):
        patch = (labeled_mask == label_id).astype(np.uint8)
        neighbors = np.sum(patch & np.roll(patch, 1, axis=0)) + np.sum(patch & np.roll(patch, 1, axis=1))
        total_contiguity += neighbors
        valid_patches += 1

    if valid_patches > 0:
        return total_contiguity / valid_patches
    return np.nan

def calculate_patch_density(raster, class_value, pixel_size=100):

    mask = (raster == class_value).astype(np.uint8)
    labeled_mask, num_labels = label(mask, connectivity=1, return_num=True)
    raster_area = raster.size * (pixel_size ** 2)
    if raster_area > 0:
        return num_labels / raster_area
    return np.nan

def process_raster(file_path, classes=[0, 1, 2, 3], pixel_size=100):

    raster, _ = read_raster(file_path)
    results = []
    entropy = calculate_shannon_entropy(raster)
    usi = calculate_urban_sprawled_index(raster)

    for class_value in classes:
        avg_contiguity = calculate_average_contiguity(raster, class_value)
        patch_density = calculate_patch_density(raster, class_value)
        
        results.append({
            'Class': class_value,
            'USI': usi,
            'Shannon Entropy': entropy,
            'Average Contiguity': avg_contiguity,
            'Patch Density': patch_density
        })

    return results

def consolidate_results(raster_files, pixel_size=100):

    consolidated_data = []
    for file_path in raster_files:
        raster_name = file_path.split('/')[-2] + "/" + file_path.split('/')[-1]
        metrics = process_raster(file_path, pixel_size=pixel_size)
        for metric in metrics:
            metric['Raster'] = raster_name
            consolidated_data.append(metric)

    return pd.DataFrame(consolidated_data)


############################################

raster_files = [
    "", # Please input here - Initial file
    "", # Please input here - Scenario 1
    "" # Please input here - Scenario 2
]

# Consolidate results into a table
results_table = consolidate_results(raster_files)
print(results_table)

results_table.to_csv("spatial_metrics_results.csv", index=False)
