import os

import numpy as np
import rasterio
import requests
from affine import Affine
from osgeo import gdal, osr
from rasterio.enums import Resampling
from rasterio.mask import mask
from rasterio.windows import from_bounds
from tqdm import tqdm


# Function to download a file from a URL
def download_file(url, filename, folder_path, verbose=True):
    # Full path to save the file
    file_path = os.path.join(folder_path, filename)
    # Download the file and save it
    response = requests.get(url)
    if response.status_code == 200:
        with open(file_path, "wb") as file:
            file.write(response.content)
        if verbose:
            print(f"File downloaded successfully: {file_path}")
    else:
        if verbose:
            print(
                f"Failed to download file: status code {response.status_code}"
            )


# Create vrt for spot and display it
def create_virtual_dataset(
    files_list, output_path, crs=2154, noDataValue=None
):
    # Create a VRT
    vrt = gdal.BuildVRT(output_path, files_list)
    # Save the VRT to a file
    vrt.FlushCache()  # Ensure all data is written
    vrt = None  # Close the dataset

    # Open the VRT for editing
    vrt = gdal.Open(output_path, gdal.GA_Update)

    if noDataValue is not None:
        # Set the nodata value for each band in the VRT
        for i in range(1, vrt.RasterCount + 1):
            band = vrt.GetRasterBand(i)
            band.SetNoDataValue(noDataValue)
    # Set CRS
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(crs)
    vrt.SetProjection(srs.ExportToWkt())
    vrt = None  # Close the dataset
    print(f"Virtual dataset successfully created at {output_path}")


def create_tif_from_npy(
    npy,
    tif_path,
    save_path,
    n_band=1,
    height=None,
    width=None,
    dtype=np.float32,
):
    if isinstance(npy, str):
        data = np.load(npy)
    else:
        data = npy
    with rasterio.open(tif_path) as src:
        profile = src.profile  # Get the metadata/profile of the VRT
        transform = src.transform
    add_dim = 1 if len(data.shape) == 3 else 0
    if height is None:
        height = data.shape[0 + add_dim]
    if width is None:
        width = data.shape[1 + add_dim]
    scaling_w = profile["width"] / width
    scaling_h = profile["height"] / height
    new_transform = transform * transform.scale(scaling_h, scaling_w)
    profile.update(
        {
            "driver": "GTiff",
            "count": n_band,
            "dtype": dtype,
            "height": height,
            "width": width,
            "transform": new_transform,
        }
    )

    # Update transform to reflect new resolution
    # Create a new GeoTIFF file with the same profile as the VRT
    with rasterio.open(save_path, "w", **profile) as dst:
        if len(data.shape) == 2:
            dst.write(data, 1)  # Write the data to the new GeoTIFF
        else:
            for band in range(1, n_band + 1):
                dst.write(
                    data[band - 1], band
                )  # Write the data to the new GeoTIFF


def check_for_corrupted_files(files_list):
    """Return files than can be opened with rasterio."""
    res = []
    for file_path in tqdm(files_list, desc="Checking for corrupted files"):
        try:
            with rasterio.open(file_path) as src:
                data = src.read()
            res.append(file_path)
        except:
            print(f"File {file_path} cannot be opened with rasterio.")
            os.rename(
                file_path, file_path.replace(".tif", "_FORMAT_ISSUE.tif")
            )
    return res


def extract_and_mask_data(vrt_path, geometry, output_tiff_path):
    # Open the VRT file with rasterio
    with rasterio.open(vrt_path) as src:
        # Mask the data with the geometry in the GeoDataFrame
        out_image, out_transform = mask(
            src, [geometry], crop=True, all_touched=True
        )

        # Define the metadata for the new TIFF file
        out_meta = src.meta.copy()
        out_meta.update(
            {
                "driver": "GTiff",
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform,
                "compress": "lzw",
                "tiled": True,
                "blockxsize": 256,
                "blockysize": 256,
                "BIGTIFF": "YES",  # Tile size: width  # Tile size: height
            }
        )

        # Write the masked data to a new TIFF file
        with rasterio.open(output_tiff_path, "w", **out_meta) as dest:
            dest.write(out_image)


def get_window(
    image_path,
    geometry=None,
    bounds=None,
    resolution=None,
    return_profile=False,
):
    """Retrieve a window from an image, within given bounds or within the bounds of a geometry."""
    with rasterio.open(image_path) as src:
        profile = src.profile
        if bounds is None:
            if geometry is not None:
                bounds = geometry.bounds
            else:
                bounds = src.bounds

        window = from_bounds(*bounds, transform=src.transform)
        transform = src.window_transform(window)

        init_resolution = profile["transform"].a

        if (resolution is not None) and (init_resolution != resolution):
            scale_factor = init_resolution / resolution
            data = src.read(
                out_shape=(
                    src.count,
                    int(np.round(window.height * scale_factor, 0)),
                    int(np.round(window.width * scale_factor, 0)),
                ),
                resampling=Resampling.bilinear,
                window=window,
            )
            # Update the transform for the new resolution
            transform = Affine(
                resolution,
                transform.b,
                transform.c,
                transform.d,
                -resolution,
                transform.f,
            )
        else:
            data = src.read(window=window)

    if return_profile:
        new_profile = profile.copy()
        new_profile.update({"transform": transform})
        return data, profile
    else:
        return data
