import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import shapely
from attr import ib
from rasterio.enums import Resampling

from src.utils import mkdir, pylogger
from src.utils.data import read_tif_downscaled

log = pylogger.RankedLogger(__name__, rank_zero_only=True)


def prepare_downsample_data(
    gf_aois: gpd.GeoDataFrame,
    downsample_factor: int,
    fold_evalref: Path,
    layers: List[str] = ["avg"],  # ["avg", "std", "max", "min"]
) -> Dict[str, Dict]:
    """
    Prepare raster targets to help evaluation
    - Extract to disk, return paths and profiles
    """

    downsample_path = {}
    downsample_folder = mkdir(fold_evalref / "downsample_target")
    if not os.path.exists(downsample_folder):
        os.makedirs(downsample_folder)

    for aoi_name, row_aoi in gf_aois.iterrows():
        poly_aoi = row_aoi["geometry"]
        with rasterio.open(str(row_aoi["rasterpath"])) as src:
            # downsample raster
            downsample_profile = src.profile.copy()
            height, width = int(src.height // downsample_factor), int(
                src.width // downsample_factor
            )
            layer_count = src.count * len(layers)
            data = np.zeros((layer_count, height, width), dtype=src.dtypes[0])
            for h in range(height):
                for w in range(width):
                    window = rasterio.windows.Window(
                        w * downsample_factor,
                        h * downsample_factor,
                        downsample_factor,
                        downsample_factor,
                    )
                    original = src.read(window=window)
                    i = 0
                    if "avg" in layers:
                        data[
                            i * src.count : (i + 1) * src.count, h, w
                        ] = np.mean(original, axis=(1, 2))
                        i += 1
                    if "std" in layers:
                        data[
                            i * src.count : (i + 1) * src.count, h, w
                        ] = np.std(original, axis=(1, 2))
                        i += 1
                    if "max" in layers:
                        data[
                            i * src.count : (i + 1) * src.count, h, w
                        ] = np.max(original, axis=(1, 2))
                        i += 1
                    if "min" in layers:
                        data[
                            i * src.count : (i + 1) * src.count, h, w
                        ] = np.min(original, axis=(1, 2))
                        i += 1

            downsample_profile[
                "transform"
            ] = src.transform * src.transform.scale(
                (src.width / data.shape[-1]), (src.height / data.shape[-2])
            )

            downsample_profile["height"] = data.shape[-2]
            downsample_profile["width"] = data.shape[-1]
            downsample_profile["count"] = layer_count
            downsample_profile["dtype"] = data.dtype

            # save downsampled raster
            downsample_path[aoi_name] = os.path.join(
                downsample_folder,
                f"{aoi_name}_downsample_x{downsample_factor}.tif",
            )
            with rasterio.open(
                downsample_path[aoi_name], "w", **downsample_profile
            ) as dst:
                dst.write(data)
    return downsample_path


def prepare_target_profiles(
    gf_aois: gpd.GeoDataFrame,
    layer_name: str,
) -> Dict[str, Dict]:
    """
    Prepare raster targets to help evaluation
    - Extract to disk, return paths and profiles
    """
    targets_profile = {}
    for aoi_name, row_aoi in gf_aois.iterrows():
        poly_aoi = row_aoi["geometry"]
        with rasterio.open(str(row_aoi[layer_name])) as src:
            raster_profile = src.profile
        raster_profile["count"] = 1
        raster_profile["dtype"] = "uint8"
        targets_profile[aoi_name] = {"profile": raster_profile, "path": None}
    return targets_profile


def apply_split(
    gf_aois: gpd.GeoDataFrame,
    subset_train: List[str],
    subset_val: List[str],
    subset_test: List[str],
    subset_pred: List[str],
):
    gf_aois["split"] = ""
    gf_aois.loc[subset_train, "split"] = "train"
    gf_aois.loc[subset_val, "split"] = "val"
    gf_aois.loc[subset_test, "split"] = "test"
    gf_aois.loc[subset_pred, "split"] = "pred"
    # Keep only aois belonging to a split
    gf_aois = gf_aois.query("split in ['train', 'val', 'test', 'pred']").copy()
    return gf_aois


def set_data_path(
    gf_aois: gpd.GeoDataFrame,
    data_path: Path,
    layers_names: List[str],
):
    for layer in layers_names:
        gf_aois[layer] = ""
        for idx, row in gf_aois.iterrows():
            gf_aois.loc[idx, layer] = data_path / layer / f"{idx}.tif"

            assert gf_aois.loc[
                idx, layer
            ].exists(), f"File {gf_aois.loc[idx, layer]} does not exist"
    return gf_aois


def compute_data_stat(gf_aois, layers_names, scale=4):
    mean = []
    std = []
    for layer in layers_names:
        stats = {}
        for aoi_name, row in gf_aois.iterrows():
            X = read_tif_downscaled(row[layer], scale=scale, crop=False)
            stats[aoi_name] = dict(
                mean=X.mean(axis=(1, 2)), var=X.var(axis=(1, 2))
            )
        # Weighted mean and std with respect to the number of samples (~area)
        stats = pd.DataFrame(stats).T.assign(n=gf_aois["train_samples"])
        log.info(f"Estimating from per-aoi stats:\n{stats}")
        mean.extend((stats["mean"] * stats["n"]).sum() / stats["n"].sum())
        std.extend(
            np.sqrt((stats["var"] * stats["n"]).sum() / stats["n"].sum())
        )
    return mean, std


def separate_geometry(
    multipolygon: shapely.MultiPolygon, distance: float = 1000
):
    polygones = list(multipolygon.geoms)
    areas = []
    while len(polygones) > 0:
        poly = polygones.pop()
        last_removed = 0
        while last_removed > -1:
            last_removed = -1
            for i, p in enumerate(polygones):
                if shapely.dwithin(poly, p, distance):
                    poly = poly.union(p)
                    last_removed = i
                    break
            if last_removed > -1:
                polygones.pop(last_removed)
        areas.append(poly)
    return areas
