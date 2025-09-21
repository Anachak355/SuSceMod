# -*- coding: utf-8 -*-
"""
@author: Anasua Chakraborty
"""

import rasterio
import numpy as np
import pandas as pd
import os
import glob

def compute_cross_tab(raster1_path, raster2_path, nodata_value):
    with rasterio.open(raster1_path) as src1, rasterio.open(raster2_path) as src2:
        data1 = src1.read(1)
        data2 = src2.read(1)
    
    # Mask out nodata values
    valid_mask = (data1 != nodata_value) & (data2 != nodata_value)
    data1 = data1[valid_mask]
    data2 = data2[valid_mask]
    
    # Compute the cross-tabulation
    cross_tab = pd.crosstab(data1, data2)
    
    return cross_tab

def get_unique_filename(base_name):
    counter = 1
    file_name = f"{base_name}.csv"
    while os.path.exists(file_name):
        file_name = f"{base_name}({counter}).csv"
        counter += 1
    return file_name

def print_to_csv(df_list,output_filename):
    csv_output = []
    
    for df in df_list:
        df_temp = df.iloc[:, :]
        csv_output.append(df_temp.to_csv(index=False, header=False))
        csv_output.append("\n")
    
    output_filename = get_unique_filename(output_filename)
    
    with open(output_filename, 'w', newline='') as f:
        f.writelines(csv_output)
        
def combine_csvs(output_folder, output_filename):
    with pd.ExcelWriter(output_filename) as writer:
        for csv_file in glob.glob(os.path.join(output_folder, '*.csv')):
            sheet_name = os.path.splitext(os.path.basename(csv_file))[0]
            df = pd.read_csv(csv_file)
            df.to_excel(writer, sheet_name=sheet_name, index=False)
        
###################################################################

parent_raster_folder = '' # Please input here
output_path = '' # Please input here
sim_types = ['_BAU'] # '_BAU' '_DO' '_centralities'
zoning_types = ['BU'] # 'hab' 'eco' 'aut' 'BU'
years = [(1990, 2000), (2000, 2010), (2010, 2020), (2020, 2030), (2030, 2040), (2040, 2050)]

nodata_value = np.nan  # or 0 for zero

###################################################################

values_dict = {}
for sim_type in sim_types:
    zonings = {}
    for zoning_type in zoning_types:
        tabs = []
        for year1, year2 in years:
            # Base the paths to the maps on the scenario being modelled and the zoning type. This code is just an example
            map1 = '' # Please input here
            map2 = '' # Please input here
            crosstab_overall = compute_cross_tab(map1, map2, nodata_value)
            tabs.append(crosstab_overall)
        
        output_filename = output_path + f'{sim_type[1:]}_{zoning_type}.csv'
        print_to_csv(tabs, output_filename)
        
        zonings[zoning_type] = tabs
                
    values_dict[sim_type] = zonings

###################################################################

combine_csvs(output_path, output_path + 'Combined.xlsx')