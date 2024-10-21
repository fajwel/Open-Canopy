import os
from datetime import datetime
import hydra
import numpy as np
import pandas as pd
import rasterio
from omegaconf import DictConfig, OmegaConf
from osgeo_utils.gdal_pansharpen import gdal_pansharpen
import warnings


@hydra.main(version_base=None, config_path="../config", config_name="pansharpen_config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    if not cfg["version"]:
        version = datetime.now().strftime("%Y%m%d%H%M")
    else:
        version = str(cfg["version"])

    save_dir = os.path.join(cfg["save_dir"], version, str(cfg.year))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # save the config next to the data
    OmegaConf.save(cfg, os.path.join(save_dir, "pansharpen_config.yaml"))

    # Get paths to images (multispectral and panchromatic)
    spot_dir = os.path.expanduser(cfg.spot_dir)
    spot_ms_dir = os.path.join(spot_dir, "spot_" + str(cfg.year) + "_ms")
    spot_pan_dir = os.path.join(spot_dir, "spot_" + str(cfg.year) + "_pan")
    ms_files = np.array(
        [
            os.path.join(spot_ms_dir, x)
            for x in os.listdir(spot_ms_dir)
            if not x.startswith(".") and x.endswith(".TIF")
        ]
    )
    pan_files = np.array(
        [
            os.path.join(spot_pan_dir, x)
            for x in os.listdir(spot_pan_dir)
            if not x.startswith(".") and x.endswith(".TIF")
        ]
    )
    spot_df = pd.DataFrame(
        np.sort(np.column_stack([ms_files, pan_files]), axis=0),
        columns=["ms", "pan"],
    )

    # Check all images correspond (have same date + first numbers)
    def check_images_match(row):
        return (
            row["ms"].split("/")[-1].split("_")[2][0:13]
            == row["pan"].split("/")[-1].split("_")[2][0:13]
        )

    spot_df["match"] = spot_df.apply(check_images_match, axis=1)

    if spot_df.shape[0] != spot_df["match"].sum():
        warnings.warn(
            f"There are {spot_df.shape[0] - spot_df['match'].sum()} images that do not match."
        )

    spot_df["short_name_is_duplicated"] = (
        spot_df["pan"]
        .apply(lambda x: x.split("/")[-1].split("_")[2][0:14])
        .duplicated()
    )

    # Pansharpen images
    for index, row in spot_df.iterrows():
        image_ms_path = row["ms"]
        image_pan_path = row["pan"]

        image_name = (
            "pansharpened_" + image_ms_path.split("/")[-1].split("_")[2][0:14] + ".tif"
        )

        # DEPRECATION WARNING: for future runs, keep only the long name
        # check if short name is duplicated, if yes, keep longer name:
        if row["short_name_is_duplicated"]:
            image_name = (
                "pansharpened_" + image_ms_path.split("/")[-1].split("_")[2] + ".tif"
            )

        out_path = os.path.join(save_dir, image_name)
        final_out_path = os.path.join(save_dir, "compressed_" + image_name)
        if not os.path.isfile(final_out_path):
            print(f"Pansharpening {image_ms_path} and {image_pan_path}")
            gdal_pansharpen(
                [
                    "",
                    "-spat_adjust",
                    "intersection",
                    image_pan_path,
                    image_ms_path,
                    out_path,
                ]
            )
            # gdal_pansharpen(image_ms_path, image_pan_path, out_path)
            with rasterio.open(os.path.join(save_dir, image_name), "r") as src:
                data = src.read()
                profile = src.profile.copy()

            if cfg.scale_to_eight_bits:
                print(f"Clipping max values to {cfg.max_value}")
                max_value = np.array(cfg.max_value)
                min_value = np.zeros_like(max_value)

                for i in range(data.shape[0]):
                    data[i, :, :] = np.clip(data[i, :, :], min_value[i], max_value[i])
                    data[i, :, :] = (
                        (data[i, :, :] - min_value[i])
                        / (max_value[i] - min_value[i])
                        * 255
                    )
                data = data.astype(np.uint8)
                profile.update(dtype=rasterio.uint8, BIGTIFF="YES")

            if cfg.compress:
                # Update the profile to include compression and tiling
                profile.update(compress="zstd", BIGTIFF="YES")
            if cfg.tiled:
                profile.update(
                    tiled=True,
                    blockxsize=256,
                    blockysize=256,
                    BIGTIFF="YES",  # Tile size: width  # Tile size: height
                )

            # Write to a new TIFF with the updated profile
            print("Compressing")
            with rasterio.open(
                os.path.join(save_dir, "compressed_" + image_name), "w", **profile
            ) as dst:
                dst.write(data)

            if cfg.remove_uncompressed:
                os.remove(os.path.join(save_dir, image_name))


if __name__ == "__main__":
    main()
