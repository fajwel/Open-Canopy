import os
import subprocess
from datetime import datetime

import hydra
import numpy as np
import pandas as pd
import rasterio
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


def gdal_pansharpen(ms_path, pan_path, out_path):
    # simple pansharpening with Brovey transform
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Define the command to run
    command = [
        "gdal_pansharpen.py",
        pan_path,
        ms_path,
        out_path,
    ]

    # Run the command
    subprocess.run(command, check=True, capture_output=True)


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
    OmegaConf.save(cfg, os.path.join(save_dir, "create_splits_config.yaml"))

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

    assert spot_df.shape[0] == spot_df["match"].sum()

    # Pansharpen images
    for image_ms_path, image_pan_path in tqdm(
        zip(spot_df["ms"], spot_df["pan"]), desc="Pansharpening images"
    ):
        image_name = (
            "pansharpened_" + image_ms_path.split("/")[-1].split("_")[2][0:14] + ".tif"
        )
        out_path = os.path.join(save_dir, image_name)
        if not os.path.isfile(out_path):
            print(f"Pansharpening {image_ms_path} and {image_pan_path}")
            gdal_pansharpen(image_ms_path, image_pan_path, out_path)

    # NB: could use orfeo toolbox instead, though processing time is much longer, images first need to be aligned,
    # and installation is complex

    image_names = [
        x
        for x in os.listdir(save_dir)
        if x.endswith(".tif")
        and not x.startswith(".")
        and not x.startswith("compressed_")
    ]

    if cfg.scale_to_eight_bits:
        if cfg.max_value is not None:
            max_value = np.array(cfg.max_value)
            min_value = np.zeros_like(max_value)
        else:
            if not os.path.isfile(os.path.join(save_dir, "quantiles.npy")):
                # Retrieve quantiles by opening all images
                # apply scaling with 0.02 and 0.98 quantile
                # XXX min scaling not working here as there are always 0 on the edges
                min_quantile_list = []
                max_quantile_list = []
                for image in tqdm(image_names, desc="Computing quantiles"):
                    with rasterio.open(os.path.join(save_dir, image), "r") as src:
                        data = src.read()
                        min_quantile_list.append(np.percentile(data, 2, axis=(1, 2)))
                        max_quantile_list.append(np.percentile(data, 98, axis=(1, 2)))

                min_quantile_list = np.vstack(min_quantile_list)
                max_quantile_list = np.vstack(max_quantile_list)

                min_value = np.min(min_quantile_list, axis=0)
                max_value = np.max(max_quantile_list, axis=0)
                data = None

                np.save(
                    os.path.join(save_dir, "quantiles.npy"),
                    [min_quantile_list, max_quantile_list],
                )
            else:
                quantile_list = np.load(os.path.join(save_dir, "quantiles.npy"))
                min_value = np.min(quantile_list[0], axis=0)
                max_value = np.max(quantile_list[1], axis=0)

    for image in tqdm(image_names, desc="Scaling and compressing images"):
        with rasterio.open(os.path.join(save_dir, image), "r") as src:
            data = src.read()
            profile = src.profile.copy()

        if cfg.scale_to_eight_bits:
            for i in range(data.shape[0]):
                data[i, :, :] = np.clip(data[i, :, :], min_value[i], max_value[i])
                data[i, :, :] = (
                    (data[i, :, :] - min_value[i]) / (max_value[i] - min_value[i]) * 255
                )
            data = data.astype(np.uint8)
            profile.update(dtype=rasterio.uint8, BIGTIFF="YES")

        if cfg.compress:
            # Update the profile to include LZW compression and tiling
            profile.update(compress="lzw", BIGTIFF="YES")
        if cfg.tiled:
            profile.update(
                tiled=True,
                blockxsize=256,
                blockysize=256,
                BIGTIFF="YES",  # Tile size: width  # Tile size: height
            )

        # Write to a new TIFF with the updated profile
        with rasterio.open(
            os.path.join(save_dir, "compressed_" + image), "w", **profile
        ) as dst:
            dst.write(data)

        if cfg.remove_uncompressed:
            os.remove(os.path.join(save_dir, image))


if __name__ == "__main__":
    main()
