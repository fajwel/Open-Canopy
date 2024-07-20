import os
import shutil

import hydra
import matplotlib.pyplot as plt
import pandas as pd
import rasterio
import rasterio.features
import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.metrics.metrics_utils import (
    compute_change_metrics,
    extract_tif_from_bounds,
    get_changes,
    get_mask,
    mask_with_geometry,
    plot_detections,
)
from omegaconf import DictConfig, OmegaConf
from rasterio.plot import show
from tqdm import tqdm


def create_fixed_size_bbox(geometry, width=300, height=300):
    # Get the bounds of the geometry
    bounds = geometry.bounds  # (minx, miny, maxx, maxy)

    # Calculate the center of the bounds
    center_x = (bounds[0] + bounds[2]) / 2
    center_y = (bounds[1] + bounds[3]) / 2

    # Calculate the half dimensions
    half_width = width / 2
    half_height = height / 2

    # Create the new bounding box with the same center
    minx = center_x - half_width
    miny = center_y - half_height
    maxx = center_x + half_width
    maxy = center_y + half_height

    bounds = minx, miny, maxx, maxy
    return bounds


@hydra.main(
    version_base=None,
    config_path="configs",
    config_name="compute_change_detection_metrics_config",
)
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    if not os.path.exists(cfg.save_dir):
        os.makedirs(cfg.save_dir)
    if not os.path.exists(os.path.join(cfg.save_dir, "figures")):
        os.makedirs(os.path.join(cfg.save_dir, "figures"))
    else:
        shutil.rmtree(os.path.join(cfg.save_dir, "figures"))
        os.makedirs(os.path.join(cfg.save_dir, "figures"))
    if not os.path.exists(os.path.join(cfg.save_dir, "data")):
        os.makedirs(os.path.join(cfg.save_dir, "data"))

    # save the config next to the data
    OmegaConf.save(cfg, os.path.join(cfg.save_dir, "compute_metrics_config.yaml"))

    # Load predictions and labels, mask on available labels (either labels_1 or labels_2,
    # not both as we suppose one of them is lidarHD, hence available everywhere)
    def resolve_path(path):
        if path is None:
            return path
        if path.startswith("/"):
            return path
        return os.path.join(cfg.root_dir, path)

    scaling_factor = {"m": 1, "dm": 0.1, "cm": 0.01}

    image_to_mask_path = None
    if cfg.labels_1.mask_on:
        image_to_mask_path = resolve_path(cfg.labels_2.path)
        image_to_get_mask_from_path = resolve_path(cfg.labels_1.path)
        image_to_mask_name = "labels_2"
        image_to_get_mask_from_name = "labels_1"
    elif cfg.labels_2.mask_on:
        image_to_mask_path = resolve_path(cfg.labels_1.path)
        image_to_get_mask_from_path = resolve_path(cfg.labels_2.path)
        image_to_mask_name = "labels_1"
        image_to_get_mask_from_name = "labels_2"

    if cfg.extract_tif:
        print(f"Extracting images from bounds of {image_to_get_mask_from_path}")
        reference_image_path = image_to_get_mask_from_path
        # Copy the file
        output_path = os.path.join(
            cfg.save_dir, "data", f"{image_to_get_mask_from_name}.tif"
        )
        shutil.copy(image_to_get_mask_from_path, output_path)

        output_path = os.path.join(cfg.save_dir, "data", f"inputs_{1}.tif")
        extract_tif_from_bounds(
            reference_image_path,
            cfg.inputs_1.path,
            output_path,
            dtype=rasterio.uint8,
        )
        cfg.inputs_1.path = output_path

        output_path = os.path.join(cfg.save_dir, "data", f"inputs_{2}.tif")
        extract_tif_from_bounds(
            reference_image_path,
            cfg.inputs_2.path,
            output_path,
            dtype=rasterio.uint8,
        )
        cfg.inputs_2.path = output_path

        output_path = os.path.join(cfg.save_dir, "data", f"{image_to_mask_name}.tif")
        extract_tif_from_bounds(
            reference_image_path,
            image_to_mask_path,
            output_path,
            dtype=rasterio.uint16,
        )
        image_to_mask_path = output_path

        output_path = os.path.join(cfg.save_dir, "data", "classification.tif")
        extract_tif_from_bounds(
            reference_image_path,
            cfg.classification.path,
            output_path,
            dtype=rasterio.uint8,
        )
        cfg.classification.path = output_path

        output_path = os.path.join(cfg.save_dir, "data", f"predictions_{1}.tif")
        extract_tif_from_bounds(
            reference_image_path,
            cfg.predictions_1.path,
            output_path,
            dtype=rasterio.uint16,
        )
        cfg.predictions_1.path = output_path

        if cfg.predictions_2.path is not None:
            # handle special case when only loss is available and given as image_1
            output_path = os.path.join(cfg.save_dir, "data", f"predictions_{2}.tif")
            extract_tif_from_bounds(
                reference_image_path,
                cfg.predictions_2.path,
                output_path,
                dtype=rasterio.uint16,
            )
            cfg.predictions_2.path = output_path

        if image_to_mask_path is not None:
            mask_plot_path = os.path.join(cfg.save_dir, "figures", "mask")
            mask_gdf = get_mask(
                image_to_get_mask_from_path,
                min_area=1000000,
                plot=True,
                plot_path=mask_plot_path,
                crs=2154,
            )
            if mask_gdf.shape[0] > 1:
                raise ValueError("More than one mask found")

            output_tif_path = os.path.join(
                cfg.save_dir, "data", f"masked_{image_to_mask_name}.tif"
            )

            mask_with_geometry(
                image_to_mask_path,
                mask_gdf["geometry"].iloc[0],
                output_tif_path,
                nodata=0,
            )
            if cfg.labels_1.mask_on:
                cfg.labels_2.path = output_tif_path
            else:
                cfg.labels_1.path = output_tif_path
            # Save figure
            vmin = 0
            vmax = 40
            fig, axes = plt.subplots(1, 1, figsize=(10, 10))
            with rasterio.open(output_tif_path) as src:
                show(
                    src,
                    ax=axes,
                    transform=src.transform,
                    vmin=vmin,
                    vmax=vmax,
                    cmap="Greens",
                    title=f"masked_{image_to_mask_name}",
                )
                mask_gdf.boundary.plot(ax=axes, edgecolor="red", linewidth=2)
            fig.savefig(
                os.path.join(cfg.save_dir, "figures", f"masked_{image_to_mask_name}"),
                bbox_inches="tight",
                pad_inches=0,
            )
            plt.close("all")

            # Mask predictions
            output_tif_path = os.path.join(
                cfg.save_dir, "data", "masked_predictions_1.tif"
            )
            mask_with_geometry(
                resolve_path(cfg.predictions_1.path),
                mask_gdf["geometry"].iloc[0],
                output_tif_path,
                nodata=0,
            )
            cfg.predictions_1.path = output_tif_path
            if cfg.predictions_2.path is not None:
                output_tif_path = os.path.join(
                    cfg.save_dir, "data", "masked_predictions_2.tif"
                )
                mask_with_geometry(
                    resolve_path(cfg.predictions_2.path),
                    mask_gdf["geometry"].iloc[0],
                    output_tif_path,
                    nodata=0,
                )
                cfg.predictions_2.path = output_tif_path

    with rasterio.open(resolve_path(cfg.labels_1.path)) as src:
        bounds = src.bounds

    # Compute metrics
    def get_metrics(bounds, delta, min_area):
        (
            height_1,
            height_2,
            difference,
            detection_lidar,
            gdf_filtered,
        ) = get_changes(
            resolve_path(cfg.labels_1.path),
            resolve_path(cfg.labels_2.path),
            delta,
            min_area=min_area,
            bounds=bounds,
            scaling_factor_1=scaling_factor[cfg.labels_1.unit],
            scaling_factor_2=scaling_factor[cfg.labels_2.unit],
            classification_mask_path=resolve_path(cfg.classification.path),
            classes_to_keep=cfg.classes_to_keep,
            resolution=cfg.resolution,
        )
        (
            pred_1,
            pred_2,
            difference_pred,
            detection_pred,
            gdf_filtered_pred,
        ) = get_changes(
            resolve_path(cfg.predictions_1.path),
            resolve_path(cfg.predictions_2.path),
            delta,
            min_area=min_area,
            bounds=bounds,
            scaling_factor_1=scaling_factor[cfg.predictions_1.unit],
            scaling_factor_2=scaling_factor[cfg.predictions_2.unit],
            classification_mask_path=resolve_path(cfg.classification.path),
            classes_to_keep=cfg.classes_to_keep,
            resolution=cfg.resolution,
        )

        metrics = compute_change_metrics(
            detection_lidar,
            detection_pred,
            difference,
            difference_pred,
            resolution=cfg.resolution,
        )

        return {
            "metrics": metrics,
            "height_1": height_1,
            "height_2": height_2,
            "pred_1": pred_1,
            "pred_2": pred_2,
            "difference": difference,
            "difference_pred": difference_pred,
            "detection_lidar": detection_lidar,
            "detection_pred": detection_pred,
            "gdf_filtered": gdf_filtered,
            "gdf_filtered_pred": gdf_filtered_pred,
        }

    # Compute metrics

    change_metrics = []

    for delta in tqdm(
        cfg.delta_list, desc=f"Computing metrics for delta in {cfg.delta_list}"
    ):
        res = []
        for min_area in tqdm(
            cfg.min_area_list,
            desc=f"Computing metrics for min_area in {cfg.min_area_list}",
        ):
            metrics_out = get_metrics(bounds, delta, min_area)
            res.append(metrics_out["metrics"])
        change_metrics.append(res)

    change_metrics = pd.DataFrame(
        change_metrics, index=cfg.delta_list, columns=cfg.min_area_list
    )

    # Save to Excel file
    output_excel_path = os.path.join(cfg.save_dir, "change_metrics.xlsx")
    dataframes = [
        ("Min difference (row) and min_area (column)", change_metrics),
    ]
    # Create a Pandas Excel writer using openpyxl as the engine
    with pd.ExcelWriter(output_excel_path, engine="openpyxl") as writer:
        start_row = 0
        for metric_name in metrics_out["metrics"].keys():
            # Write one table for each metric (f1, precision, recall, iou...)
            start_col = 0
            for title, df in dataframes:
                df_metric = df.map(lambda x: x[metric_name])
                # Write the title
                worksheet = (
                    writer.sheets["Sheet1"]
                    if "Sheet1" in writer.sheets
                    else writer.book.create_sheet("Sheet1")
                )
                worksheet.cell(
                    row=start_row + 1,
                    column=start_col + 1,
                    value=metric_name + " " + title,
                )

                # Write the DataFrame
                df_metric.to_excel(
                    writer,
                    startrow=start_row + 1,
                    startcol=start_col,
                    index=True,
                    header=True,
                )
                start_col += df.shape[1] + 3
            start_row += 4 + len(cfg.delta_list)

    print(f"Metrics saved to {output_excel_path}")

    # Plot screen shots of disturbances
    # Take large delta  and min area to restrict the number of figures to plot
    metrics_out = get_metrics(bounds, -15, 200)
    n_change_geometries = metrics_out["gdf_filtered"].shape[0]
    print(
        f"There are {n_change_geometries} lidar change geometries for delta "
        f"{15} and min_area {200}. Plotting them with delta {cfg.delta_fig} and min_area {cfg.min_area_fig}."
    )
    for ix_change in tqdm(
        range(n_change_geometries),
        desc="Plotting change geometries and metrics",
    ):
        # Plot for min delta / min min_area
        # Get the fixed size bounding box centered on the geometry
        geometry = metrics_out["gdf_filtered"]["geometry"].iloc[ix_change]
        sub_bounds = create_fixed_size_bbox(geometry, width=300, height=300)

        metrics_out_ix = get_metrics(sub_bounds, cfg.delta_fig, cfg.min_area_fig)

        save_prefix = str(ix_change)

        plot_detections(
            resolve_path(cfg.inputs_1.path),
            resolve_path(cfg.inputs_2.path),
            sub_bounds,
            metrics_out_ix,
            fs=9,
            save_prefix=save_prefix,
            year_1=cfg.year_1,
            year_2=cfg.year_2,
            save_dir=os.path.join(cfg.save_dir, "figures"),
        )


if __name__ == "__main__":
    main()
