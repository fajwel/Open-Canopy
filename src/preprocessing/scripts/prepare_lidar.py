import json
import os
import tempfile
from datetime import datetime

import geopandas as gpd
import hydra
import laspy
import numpy as np
import rasterio
import rootutils
from omegaconf import DictConfig, OmegaConf
from shapely.geometry import Polygon

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.preprocessing.utils.lidar.chm import (
    get_elevation_relative_to_ground,
    interpolate_missing_data,
)
from src.preprocessing.utils.lidar.lidarhd import (
    get_classification_raster,
    get_las_meta_data,
    load_lidar_grid,
)
from src.preprocessing.utils.utils import create_tif_from_npy, download_file


@hydra.main(
    version_base=None,
    config_path="../config",
    config_name="prepare_lidar_config",
)
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    # Start by sampling tiles

    if not cfg["data_version"]:
        data_version = datetime.now().strftime("%Y%m%d%H%M")
    else:
        data_version = cfg["data_version"]

    save_dir = os.path.join(
        os.path.expanduser(cfg["save_dir_path"]), data_version
    )

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        os.makedirs(os.path.join(save_dir, "lidar"))
        os.makedirs(os.path.join(save_dir, "spot"))
        os.makedirs(os.path.join(save_dir, "rgealti"))

    # save the config next to the data
    OmegaConf.save(cfg, os.path.join(save_dir, "data_config.yaml"))

    if os.path.exists(os.path.join(save_dir, "sampled_geometries.geojson")):
        print("Geometries already sampled, loading from file")
        gdf = gpd.read_file(
            os.path.join(save_dir, "sampled_geometries.geojson")
        )
        if not cfg.overwrite:
            # Filter lidar geometries that have already been processed
            ix_processed = np.array(
                [
                    x.split(".")[0]
                    for x in os.listdir(os.path.join(save_dir, "lidar"))
                    if x.endswith(cfg["lidar"]["format"])
                    and ("classif" not in x)
                    and ("ISSUE" not in x)
                ]
            ).astype(int)
            ix_missing = np.array(list(set(gdf.index) - set(ix_processed)))
            gdf = gdf.loc[ix_missing]
            print(
                f"{len(ix_processed)} lidar geometries have already been processed, skipping them."
            )
    else:
        print("Sampling geometries")
        # load lidar grid
        lidar_grid = load_lidar_grid(cfg["lidar_grid_path"])
        crs = 2154
        if cfg["coordinates"] is not None:
            # AOI
            aoi = Polygon(cfg["coordinates"])
            # Filter lidar_grid to only include geometries within 'aoi'
            gdf = lidar_grid[lidar_grid.to_crs(cfg.crs).within(aoi)]
        else:
            gdf = lidar_grid
        # Sample n geometries
        if cfg["max_n"]:
            gdf = gdf.sample(cfg["max_n"], random_state=1)
        # Save the GeoDataFrame
        gdf.to_file(
            os.path.join(save_dir, "sampled_geometries.geojson"),
            driver="GeoJSON",
            crs=crs,
        )
        print(f"Sampled geometries successfully saved in {save_dir}")
    print(f"There are {gdf.shape[0]} patches to process")

    if cfg["n_images_to_process"] * cfg["i"] > gdf.shape[0]:
        raise ValueError(
            f"Index ({cfg['i']}) x {cfg['n_images_to_process']} is out of bounds for sampled geometries"
        )
    else:
        # ix refers here to row number, not index
        ix = np.arange(
            cfg["n_images_to_process"] * cfg["i"],
            min(cfg["n_images_to_process"] * (cfg["i"] + 1), gdf.shape[0]),
        )
    gdf = gdf.iloc[ix]
    gdf["image_name"] = gdf.index.astype(str)
    for _, row in gdf.iterrows():
        print(f"Processing {row['image_name']}")
        if cfg["lidar"]["process"]:
            # Compute CHM (NB could add option to compute it using the dtm, which is also faster,
            # but maybe not as precise)
            # Download lidar file in tmp dir to save space
            if cfg["lidar"]["format"] == "npy":
                output_path = os.path.join(
                    save_dir, "lidar", row["image_name"] + ".npy"
                )
            elif cfg["lidar"]["format"] == "tif":
                output_path = os.path.join(
                    save_dir, "lidar", row["image_name"] + ".tif"
                )
            classif_path = os.path.join(
                save_dir,
                "lidar",
                row["image_name"] + "_classification_mask.tif",
            )
            metadata_path = os.path.join(
                save_dir, "lidar", row["image_name"] + "_metadata.json"
            )
            if (
                not cfg["overwrite"]
                and os.path.exists(output_path)
                and os.path.exists(classif_path)
                and os.path.exists(metadata_path)
            ):
                print(f"{output_path} already exists, skipping")
            else:
                with tempfile.TemporaryDirectory(
                    dir=os.path.expanduser(cfg.temp_dir_path)
                ) as temp_dir:
                    print(f"Processing lidar {row['nom_pkk']}")
                    download_file(row["url_telech"], row["nom_pkk"], temp_dir)
                    las = laspy.read(os.path.join(temp_dir, row["nom_pkk"]))
                # One tile is 1000m x 1000m
                width = int(np.around(1000 / cfg["lidar"]["resolution"], 0))
                # Get metadata and classification raster
                metadata = get_las_meta_data(las, save_path=None)
                bounds = row.geometry.bounds
                # Save each image and metadata in a separate file
                _, profile = get_classification_raster(
                    las,
                    width=width,
                    height=width,
                    save_path=classif_path,
                    bounds=bounds,
                    resolution=cfg["lidar"]["resolution"],
                )

                # Get lidar image
                lidar_image = get_elevation_relative_to_ground(
                    las,
                    width,
                    width,
                    method=cfg["lidar"]["method"],
                    quantile=cfg["lidar"]["quantile"],
                    classes_to_filter=[1, 65, 66],
                    dtm=dtm,
                )
                lidar_image = interpolate_missing_data(lidar_image)
                lidar_image[lidar_image < 0] = 0
                # Add histogram of heights to the metadata
                metadata["bin_edges"] = [
                    0,
                    5,
                    10,
                    15,
                    20,
                    25,
                    30,
                    35,
                    40,
                    45,
                    50,
                    55,
                    60,
                    100,
                ]
                hist, _ = np.histogram(
                    lidar_image.flatten(), bins=metadata["bin_edges"]
                )
                metadata["hist"] = hist.tolist()
                with open(metadata_path, "w") as f:
                    json.dump(metadata, f)
                # multiply by 10 and save to int16 to save disk usage
                # NB could save 2 digits but not necessary
                lidar_image = np.round(10 * lidar_image, 0).astype(np.uint16)
                if cfg["lidar"]["format"] == "npy":
                    np.save(output_path, lidar_image)
                elif cfg["lidar"]["format"] == "tif":
                    create_tif_from_npy(
                        lidar_image,
                        classif_path,
                        output_path,
                        n_band=1,
                        dtype=rasterio.uint16,
                    )
                else:
                    raise ValueError(
                        f"Format {cfg['lidar']['format']} not supported to save LiDAR tiles"
                    )


if __name__ == "__main__":
    main()
