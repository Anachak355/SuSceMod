import numpy as np
import pandas as pd
import rasterio
from osgeo import gdal
import csv
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

def load_raster_data(raster_path):
    with rasterio.open(raster_path) as src:
        arr = src.read(1, out_dtype="float32", masked=True)
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

def counts_to_df(counts_dict, exclude_nan=True):
    rows = []
    for year, (cats, counts) in counts_dict.items():
        for c, v in zip(cats, counts):
            if(c != 'nan'):
                rows.append({"year": year, "category": c, "count": v})
    
    counts_df = pd.DataFrame(rows)
    return counts_df
            
def compute_cross_tab(raster1_path, raster2_path, nodata_value):
    with rasterio.open(raster1_path) as src1, rasterio.open(raster2_path) as src2:
        data1 = src1.read(1)
        data2 = src2.read(1)
    
    valid_mask = (data1 != nodata_value) & (data2 != nodata_value)
    data1 = data1[valid_mask]
    data2 = data2[valid_mask]
    
    cross_tab = pd.crosstab(data1, data2)
    
    return cross_tab

def plot_raster_classes(raster, classes=None, colors=None, title="Raster Preview"):
    """
    Plot a classified raster with discrete colors.

    Parameters
    ----------
    raster : np.ndarray
        2D array with integer class values and NaNs.
    classes : list, optional
        List of class values to display. If None, inferred from raster.
    colors : list, optional
        List of colors (hex or names) matching the number of classes.
        If None, defaults to Set2 palette.
    title : str
        Title of the plot.
    """
    # Infer classes if not provided
    if classes is None:
        classes = sorted([c for c in np.unique(raster) if not np.isnan(c)])

    n_classes = len(classes)

    # Choose colors
    if colors is None:
        base_colors = plt.cm.Set2.colors  # soft categorical palette
        if n_classes <= len(base_colors):
            colors = base_colors[:n_classes]
        else:
            colors = plt.cm.tab20.colors[:n_classes]  # fallback
    cmap = ListedColormap(colors)

    # Define boundaries
    bounds = np.arange(min(classes) - 0.5, max(classes) + 1.5, 1)
    norm = BoundaryNorm(bounds, cmap.N)

    # Plot
    plt.figure(figsize=(14, 6))
    im = plt.imshow(raster, cmap=cmap, norm=norm)

    # Add discrete colorbar
    cbar = plt.colorbar(im, ticks=classes)
    cbar.ax.set_yticklabels([f"Class {c}" for c in classes])
    cbar.set_label("Density Class")

    # Clean up axes
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    plt.show()