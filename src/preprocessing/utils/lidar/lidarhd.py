import datetime
import json

import geopandas as gpd
import laspy
import numpy as np
import pandas as pd
import rasterio
from affine import Affine
from rasterio.transform import from_bounds
from scipy.stats import binned_statistic_2d


def load_lidar_grid(lidar_grid_folder_path):
    lidar_grid = gpd.read_file(lidar_grid_folder_path)
    # Extract Lambert 93 coordinates
    # Regular expression pattern
    pattern = r"_FXX_(\d{4})_(\d{4})_"
    # Extract and create new columns
    lidar_grid[["X", "Y"]] = lidar_grid["nom_pkk"].str.extract(pattern)
    return lidar_grid


def get_lidar_gps_time(las_file, point_index):
    if isinstance(las_file, str):
        # Load the LAS file
        las = laspy.read(las_file)
    else:
        las = las_file

    # Extract the GPS time for the specified point
    gps_time = las.gps_time[point_index]

    # According to IGN documentation, absolute time is 14/09/2011 00:00:00 UTC
    # cf. https://geoservices.ign.fr/sites/default/files/2023-10/DC_LiDAR_HD_1-0_PTS.pdf
    start_date = datetime.datetime(2011, 9, 14, 0, 0, 0)

    # Calculate the date and time from GPS time
    # Week number and seconds within the week
    gps_weeks = int(gps_time / (60 * 60 * 24 * 7))
    gps_seconds = gps_time - gps_weeks * 60 * 60 * 24 * 7

    # Calculate final date and time
    final_date_time = start_date + datetime.timedelta(
        weeks=gps_weeks, seconds=gps_seconds
    )

    return final_date_time


def get_classification_raster(
    las, width=1000, height=1000, save_path=None, bounds=None, resolution=None
):
    # Define the grid size (x, y in meter coordinates)
    if bounds is not None:
        x_min, y_min, x_max, y_max = bounds
    else:
        x_min, x_max = np.around(np.min(las.x), 0), np.around(np.max(las.x), 0)
        y_min, y_max = np.around(np.min(las.y), 0), np.around(np.max(las.y), 0)
    grid_x, grid_y = np.linspace(x_min, x_max, width + 1), np.linspace(
        y_min, y_max, height + 1
    )

    # Calculate the mode for each grid cell
    def compute_mode(array):
        # for int non negative, much faster to use the following rather than scipy.stats.mode
        most_frequent_value = np.argmax(np.bincount(array))
        return most_frequent_value

    # Bin data into the grid
    classif, x_edge, y_edge, _ = binned_statistic_2d(
        las.x,
        las.y,
        las.classification,
        statistic=compute_mode,
        bins=[grid_x, grid_y],
    )
    classif = np.flipud(classif.T)

    # Save as tif raster
    if save_path is not None:
        # Construct affine transform
        transform = from_bounds(
            x_min, y_min, x_max, y_max, classif.shape[1], classif.shape[0]
        )
        if resolution is not None:
            # Fix resolution
            transform = Affine(
                resolution,
                transform.b,
                transform.c,
                transform.d,
                -resolution,
                transform.f,
            )

        # Define metadata
        profile = {
            "driver": "GTiff",  # GeoTIFF
            "dtype": np.uint8,
            "nodata": None,  # Specify if you have a nodata value
            "width": classif.shape[1],
            "height": classif.shape[0],
            "count": 1,  # Number of bands; assuming a single band for 2D array
            "crs": "EPSG:2154",  # Coordinate Reference System; adjust as needed
            "transform": transform,
        }

        # Save the array as a TIFF
        with rasterio.open(save_path, "w", **profile) as dst:
            dst.write(classif, 1)  # Write array to the first band

    return classif, profile


# Extract stats on classified points
def get_las_meta_data(las, save_path=None):
    meta = {}
    meta["counts"] = pd.Series(las.classification).value_counts().to_dict()
    meta["n_points"] = len(las.points)
    meta["acquisition_date"] = get_lidar_gps_time(las, 0).strftime("%Y%m%d")
    if save_path is not None:
        with open(save_path, "w") as f:
            json.dump(meta, f)
    return meta
