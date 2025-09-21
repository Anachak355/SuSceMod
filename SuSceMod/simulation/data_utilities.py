# -*- coding: utf-8 -*-
"""
Created on Sun Aug  4 11:05:58 2024

@author: Administrator
"""

import numpy as np
import pandas as pd
import rasterio
from osgeo import gdal
import csv

def load_raster_data(raster_path):
    with rasterio.open(raster_path) as src:
        # Read as float so NaN is representable, and apply the dataset mask/NoData
        arr = src.read(1, out_dtype="float32", masked=True)
        # Convert masked values (incl. NoData) to NaN
        data = np.ma.filled(arr, np.nan)
    return data


def writeraster(template_raster_path, output_raster_path, lu_org):
    
    gdal.UseExceptions()
    gdal.PushErrorHandler('CPLQuietErrorHandler')
    
    template_raster = gdal.Open(template_raster_path)
    output_raster = gdal.GetDriverByName('GTiff').Create(
        output_raster_path,
        template_raster.RasterXSize,
        template_raster.RasterYSize,
        1,
        gdal.GDT_Float32,
    )
    output_raster.SetGeoTransform(template_raster.GetGeoTransform())
    output_raster.SetProjection(template_raster.GetProjection())
    
    output_band = output_raster.GetRasterBand(1)
    output_band.WriteArray(lu_org.astype('float32'))
    output_band.SetNoDataValue(np.nan)
    output_raster = None
    template_raster = None
    
def save_counts_to_csv(class_counts, filename):  
    counts = [class_counts[key][1] for key in class_counts.keys()]
    keys = list(class_counts.keys())
    result = [list(pair) for pair in zip(*counts)]
    
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([''] + keys)
        for i, row in enumerate(result):
            row = list(map(str, row))
            row.insert(0,str(class_counts[keys[0]][0][i]))
            writer.writerow(row)
            
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