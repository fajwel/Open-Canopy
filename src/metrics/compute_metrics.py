import os

# import contextily as ctx
import dask.array as da
import geopandas as gpd
import h5py
import hydra
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from omegaconf import DictConfig, OmegaConf
from rasterio.enums import Resampling
from shapely.geometry import box
from tqdm import tqdm

try:
    from src.metrics.metrics_utils import (
        get_vegetation_and_forest_mask,
        get_window,
        save_dict_of_dicts_to_excel,
    )
except ImportError:
    from metrics_utils import (
        get_vegetation_and_forest_mask,
        get_window,
        save_dict_of_dicts_to_excel,
    )


@hydra.main(
    version_base=None,
    config_path="configs",
    config_name="compute_metrics_config",
)
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    compute_metrics(
        dataset_dir=cfg.root_dir,
        predictions_path=cfg.model.predictions_path,
        save_dir=cfg.model.save_dir,
        labels_path=cfg.labels_path,
        classification_path=cfg.classification_path,
        split=cfg.split,
        bins=cfg.bins,
        resolution=cfg.resolution,
        labels_unit=cfg.labels_unit,
        predictions_unit=cfg.model.predictions_unit,
        resampling_method=cfg.resampling_method,
        classes_to_keep=cfg.classes_to_keep,
        tree_cover_threshold=cfg.tree_cover_threshold,
        forest_mask_path=cfg.forest_mask_path,
        geometries_path=cfg.geometries_path,
        greco_name=cfg.greco_name,
        greco_path=cfg.greco_path,
        save_pred_fig=cfg.save_pred_fig,
    )


def compute_metrics(
    dataset_dir,
    predictions_path,
    save_dir,
    labels_path={
        2021: "2021/lidar.vrt",
        2022: "2022/lidar.vrt",
        2023: "2023/lidar.vrt",
    },
    classification_path={
        2021: "2021/lidar_classification.vrt",
        2022: "2022/lidar_classification.vrt",
        2023: "2023/lidar_classification.vrt",
    },
    split="test",
    bins=[0, 2, 5, 10, 15, 20, 30, 60],
    resolution=1.5,
    labels_unit="dm",
    predictions_unit="dm",
    resampling_method="bilinear",
    classes_to_keep=[5],
    tree_cover_threshold=2,
    forest_mask_path="forest_mask.parquet",
    geometries_path="geometries.geojson",
    greco_name=None,
    greco_path="greco.geojson",
    save_pred_fig=True,
):
    """Compute metrics on some predictions.

    Args:
        dataset_dir (str): Path to the dataset directory
        predictions_path (str): Can be either a path or a dict of paths to tif/vrt of predictions for each year
        save_dir (_type_): Directory where to save the metrics
        labels_path (str, optional): Can be either a path or a dict of paths to tif/vrt of labels for each year. Defaults to {2021 : "2021/lidar.vrt", 2022: "2022/lidar.vrt", 2023: "2023/lidar.vrt"}.
        classification_path (str, optional): Can be either a path or a dict of paths to tif/vrt of classification for each year. Defaults to {2021 : "2021/lidar_classification.vrt", 2022: "2022/lidar_classification.vrt", 2023: "2023/lidar_classification.vrt"}.
        split (str, optional): Split on which we compute the metrics. Defaults to "test".
        bins (list, optional): Bins on which we compute the metrics. Defaults to [0, 2, 5, 10, 15, 20, 30, 60].
        resolution (float, optional): Resolution at which to compute metrics. Defaults to 1.5.
        labels_unit (str, optional): m(meter), dm(decimeter), cm(centimeter). Defaults to "dm".
        predictions_unit (str, optional): m(meter), dm(decimeter), cm(centimeter). Defaults to "dm".
        resampling_method (str, optional): method to match resolution if necessary (can be "max_pooling" or "bilinear", typically use "bilinear" for upsampling and "max_pooling" for downsampling). Defaults to "bilinear".
        classes_to_keep (list, optional): 3: Low vegetation (<=0.5m), 4: Medium vegetation (0.5-1.5m), 5: High vegetation (>=1.5m). Defaults to [5].
        tree_cover_threshold (int, optional): Threshold above which to consider vegetation as tree cover (in meters). Defaults to 2.
        forest_mask_path (str, optional): Path to forest mask. Defaults to "forest_mask.parquet".
        geometries_path (str, optional): Path to geometries. Defaults to "geometries.geojson".
        greco_name (_type_, optional): Select only geometries that are in greco "greco_name", if None, do not filter. Defaults to None.
        greco_path (str, optional): Path to greco geometries. Defaults to "greco.geojson".
        save_pred_fig (bool, optional): Whether to save figure with prediction. Defaults to True.
    """

    def resolve_path(path):
        if path.startswith("/"):
            return path
        return os.path.join(dataset_dir, path)

    save_dir = resolve_path(save_dir)
    if greco_name is not None:
        save_dir = os.path.join(save_dir, "greco_" + greco_name)
    else:
        save_dir = save_dir

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(os.path.join(save_dir, "prediction_figures")):
        os.makedirs(os.path.join(save_dir, "prediction_figures"))
    if not os.path.exists(os.path.join(save_dir, "data")):
        os.makedirs(os.path.join(save_dir, "data"))

    # Load predictions and labels:
    # Use test tiles geometries
    print(f"Retrieve {split} geometries")
    gdf = gpd.read_file(os.path.join(dataset_dir, geometries_path)).query(
        "split==@split"
    )

    # Select geometries that are in available predictions years
    if isinstance(predictions_path, (dict, DictConfig)):
        gdf = gdf.loc[gdf["lidar_year"].isin(predictions_path.keys())]

    if greco_name is not None:
        greco_gdf = gpd.read_file(os.path.join(dataset_dir, greco_path))
        gdf = gdf[
            gdf.geometry.within(
                greco_gdf.query("greco==@greco_name")["geometry"].iloc[0]
            )
        ]

    # Check that geometries are within predictions bounds
    def check_geometries_within_prediction_bounds(
        gdf, prediction_path, lidar_year=None
    ):
        if lidar_year is not None:
            gdf = gdf.query("lidar_year==@lidar_year")
        with rasterio.open(prediction_path) as src:
            pred_bounds = src.bounds
            # Create a bounding box from the VRT bounds
            pred_bounds = box(
                pred_bounds.left,
                pred_bounds.bottom,
                pred_bounds.right,
                pred_bounds.top,
            )

            # Ensure the GeoPandas DataFrame and the VRT bounding box have the same CRS
            if gdf.crs != src.crs:
                gdf = gdf.to_crs(src.crs)

        # Check which geometries are within the VRT bounds
        ix_within = gdf.geometry.within(pred_bounds)
        if gdf.shape[0] != ix_within.sum():
            raise ValueError("Evaluation set not covered by predictions")
        print(
            f"There are {gdf.shape[0]} geometries within the predictions bounds for lidar year {lidar_year}"
        )

    if isinstance(predictions_path, (dict, DictConfig)):
        for year, prediction_path in predictions_path.items():
            check_geometries_within_prediction_bounds(
                gdf, os.path.join(dataset_dir, prediction_path), year
            )
    else:
        prediction_path = predictions_path
        check_geometries_within_prediction_bounds(
            gdf, os.path.join(dataset_dir, prediction_path)
        )

    # Create a figure and an axis object
    fig, ax = plt.subplots(figsize=(10, 10))

    gdf.plot(ax=ax, color="red", alpha=0.7)

    # # Add the Contextily basemap (map of France)
    # Need to disable it for JZ (no access to the internet)
    # ctx.add_basemap(ax, crs=gdf.crs.to_string())

    # Set title and hide axes
    ax.set_title("Test tiles")
    ax.set_axis_off()

    fig.savefig(
        os.path.join(save_dir, "evaluation_set.jpg"),
        bbox_inches="tight",
        pad_inches=0,
    )

    # Compute metrics on each bin
    bins = bins
    bin_keys = [
        str(bins[i]) + "-" + str(bins[i + 1]) for i in range(0, len(bins) - 1)
    ]

    label_count_bin = {i: [] for i in range(1, len(bins))}
    pred_count_bin = {i: [] for i in range(1, len(bins))}

    # Initialize the HDF5 file and create a resizable dataset for each bin of predictions/labels
    def create_hdf5_resizable_dataset(save_path, dataset_keys, n_rows=None):
        with h5py.File(save_path, "w") as hdf:
            # Define the shape and maximum shape of the dataset
            initial_shape = (0,)  # Start with 100 rows and 100 columns
            max_shape = (n_rows,)  # Allow unlimited rows if n_rows=None
            for key in dataset_keys:
                # Create the resizable dataset
                _ = hdf.create_dataset(
                    key, shape=initial_shape, maxshape=max_shape, chunks=True
                )

    def extend_hdf5_array(file_path, dataset_key, array):
        # Use HDF5 to dynamically save chunks of data and avoid out of memory issues
        if array.shape[0] > 0:
            with h5py.File(file_path, "a") as hdf:
                dset = hdf[dataset_key]
                # Extend the dataset to accommodate the new array
                dset.resize(dset.shape[0] + array.shape[0], axis=0)
                # Write the new array data to the dataset
                dset[-array.shape[0] :] = array

    labels_save_path = os.path.join(save_dir, "data", "labels.h5")
    predictions_save_path = os.path.join(save_dir, "data", "predictions.h5")
    create_hdf5_resizable_dataset(labels_save_path, bin_keys)
    create_hdf5_resizable_dataset(predictions_save_path, bin_keys)

    forest_mask_gdf = gpd.read_parquet(resolve_path(forest_mask_path))
    for ix_row, row in tqdm(
        gdf.iterrows(), desc="Retrieve data for each tile"
    ):
        geometry = row["geometry"]
        year = row["lidar_year"]
        if isinstance(labels_path, (dict, DictConfig)):
            label_path = os.path.join(dataset_dir, labels_path[year])
        else:
            label_path = os.path.join(dataset_dir, labels_path)
        if isinstance(classification_path, (dict, DictConfig)):
            classification_path = os.path.join(
                dataset_dir, classification_path[year]
            )
        else:
            classification_path = os.path.join(
                dataset_dir, classification_path
            )
        if isinstance(predictions_path, (dict, DictConfig)):
            prediction_path = os.path.join(dataset_dir, predictions_path[year])
        else:
            prediction_path = os.path.join(dataset_dir, predictions_path)
        if (
            resampling_method == "bilinear"
            or resampling_method == Resampling.bilinear
        ):
            resampling_method = Resampling.bilinear
        else:
            raise ValueError(
                f"Resampling method {resampling_method} is not implemented."
            )
        labels = get_window(
            label_path,
            geometry=geometry,
            resolution=resolution,
            resampling_method=resampling_method,
        ).astype(np.float32)
        # Take union of classification mask and forest mask
        # XX for vegetation mask, always take the mode when resampling is needed
        complete_mask, _ = get_vegetation_and_forest_mask(
            forest_mask_gdf,
            classification_path,
            geometry,
            classes_to_keep,
            resolution=resolution,
            resampling_method=Resampling.mode,
        )
        predictions = get_window(
            prediction_path,
            geometry=geometry,
            resolution=resolution,
            resampling_method=resampling_method,
        ).astype(np.float32)

        scaling_factor = {
            "m": 1,
            "dm": 10,
            "cm": 100,
        }
        predictions = predictions / scaling_factor[predictions_unit]
        labels = labels / scaling_factor[labels_unit]
        complete_mask = np.expand_dims(complete_mask, axis=0)

        if save_pred_fig and (ix_row % 100 == 0):
            # Create a figure and an axis object
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.imshow(predictions.squeeze(), cmap="Greens", vmin=0, vmax=40)
            ax.axis("off")  # Turn off the axis
            fig.savefig(
                os.path.join(
                    save_dir,
                    "prediction_figures",
                    str(ix_row) + "_predictions.jpg",
                ),
                bbox_inches="tight",
                pad_inches=0,
            )
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.imshow(labels.squeeze(), cmap="Greens", vmin=0, vmax=40)
            ax.axis("off")  # Turn off the axis
            fig.savefig(
                os.path.join(
                    save_dir, "prediction_figures", str(ix_row) + "_labels.jpg"
                ),
                bbox_inches="tight",
                pad_inches=0,
            )
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.imshow(complete_mask.squeeze())
            ax.axis("off")  # Turn off the axis
            fig.savefig(
                os.path.join(
                    save_dir,
                    "prediction_figures",
                    str(ix_row) + "_vegetation_mask.jpg",
                ),
                bbox_inches="tight",
                pad_inches=0,
            )
            plt.close("all")

        bin_indices = np.digitize(labels, bins)  # Bin data by labels

        bin_indices_pred = np.digitize(predictions, bins)  # Bin data by labels

        for i in range(
            1, len(bins)
        ):  # do not consider bins out of range (<0 or >200)
            # Select data for the current bin
            # NB: keep only points in bin and with selected class
            bin_mask = bin_indices == i
            if bin_mask.shape != complete_mask.shape:
                print(
                    "Warning: bin mask incompatible with forest mask, this may indicate a problem. Sample will be ignored."
                )
                print(f"{bin_mask.shape=} {complete_mask.shape=}")
                continue
            bin_mask = bin_mask & complete_mask

            bin_labels = labels[bin_mask]
            bin_predictions = predictions[bin_mask]

            # Count pred values in bins
            pred_bin_mask = (bin_indices_pred == i) & complete_mask
            pred_bin_predictions = predictions[pred_bin_mask]

            extend_hdf5_array(labels_save_path, bin_keys[i - 1], bin_labels)
            extend_hdf5_array(
                predictions_save_path, bin_keys[i - 1], bin_predictions
            )

            # TODO use hdf5 also for counts (although not really necessary, but easier to read)
            label_count_bin[i].append(len(bin_labels))
            pred_count_bin[i].append(len(pred_bin_predictions))

    def compute_metrics_with_dask(y_pred, y_true):
        # Mean Error (ME)
        mean_error = da.mean(y_pred - y_true)
        # Absolute Error
        absolute_error = da.abs(y_pred - y_true)
        # Mean Absolute Error (MAE)
        mae = da.mean(absolute_error)
        # Standard Deviation of the Error
        std_e = da.std(y_pred - y_true)
        # Standard Deviation of the Absolute Error
        std_ae = da.std(absolute_error)
        # Normalized Mean Absolute Error (nMAE)
        # Add one to the denominator in order to avoid division by zero
        nmae = da.mean(absolute_error / (1 + da.abs(y_true)))
        # Compute the RMSE
        rmse = da.sqrt(da.mean((y_pred - y_true) ** 2))
        # Metrics for whiskers plot
        percentiles = da.percentile(
            y_pred - y_true,
            [0, 25, 50, 75, 100],
            internal_method="tdigest",
            method="linear",
        ).compute()

        # Compute intersection and union
        tree_cover_true = y_true >= tree_cover_threshold
        tree_cover_pred = y_pred >= tree_cover_threshold

        intersection = da.logical_and(tree_cover_pred, tree_cover_true).sum()
        # Add one to the denominator to avoid division by zero
        union = da.logical_or(tree_cover_pred, tree_cover_true).sum()
        # Compute IoU
        iou = intersection / (union + 1)

        # Compute the metrics
        res = {
            "mean_absolute_error": mae.compute(),
            "mean_error": mean_error.compute(),
            "std_error": std_e.compute(),
            "std_absolute_error": std_ae.compute(),
            "root_mean_squared_error": rmse.compute(),
            "normalized_mean_absolute_error": nmae.compute(),
            "min_error": percentiles[0],
            "max_error": percentiles[4],
            "median_error": percentiles[2],
            "first_quartile_error": percentiles[1],
            "third_quartile_error": percentiles[3],
            "tree_cover_iou": iou.compute(),
        }
        return res

    metrics = {}
    y_pred_list = []
    y_true_list = []

    # Open the HDF5 file in read mode and load the dataset to dask
    # !!!WARNING!!! use small chunks for dask, since percentiles are only an approximation
    # and would diverge otherwise

    hdf_labels = h5py.File(labels_save_path, "r")
    hdf_predictions = h5py.File(predictions_save_path, "r")
    for bin in tqdm(bin_keys, desc="computing metrics on each bin"):
        dataset = hdf_labels[bin]
        y_true = da.from_array(dataset, chunks="0.1 MiB")
        dataset = hdf_predictions[bin]
        y_pred = da.from_array(dataset, chunks="0.1 MiB")
        # Ensure that the arrays have the same shape
        assert (
            y_pred.shape == y_true.shape
        ), "Shapes of y_pred and y_true must match"

        res = compute_metrics_with_dask(y_pred, y_true)
        for key in res:
            if key not in metrics:
                metrics[key] = {}
            metrics[key][bin] = res[key]

        y_pred_list.append(y_pred)
        y_true_list.append(y_true)

    # Add metric on all bins combined:
    print("Compute metrics on all bins combined")
    y_pred_full = da.concatenate(y_pred_list, axis=0)
    y_true_full = da.concatenate(y_true_list, axis=0)
    res = compute_metrics_with_dask(y_pred_full, y_true_full)
    for key in metrics:
        metrics[key]["Full"] = res[key]

    # For nmae, compute full metrics without the first bin
    print("Compute nmae on all bins combined")
    y_pred_full_wo_first_bin = da.concatenate(y_pred_list[1::], axis=0)
    y_true_full_wo_first_bin = da.concatenate(y_true_list[1::], axis=0)
    nmae = da.mean(
        da.abs(y_pred_full_wo_first_bin - y_true_full_wo_first_bin)
        / (1 + da.abs(y_true_full_wo_first_bin))
    ).compute()
    metrics["normalized_mean_absolute_error"]["Full"] = nmae

    hdf_labels.close()
    hdf_predictions.close()
    os.remove(labels_save_path)
    os.remove(predictions_save_path)

    metrics["label_count_bin"] = {
        bin_keys[i - 1]: np.sum(label_count_bin[i])
        for i in range(1, len(bins))
    }
    metrics["pred_count_bin"] = {
        bin_keys[i - 1]: np.sum(pred_count_bin[i]) for i in range(1, len(bins))
    }

    # Save to excel for easier manipulation
    save_dict_of_dicts_to_excel(
        metrics, os.path.join(save_dir, "metrics.xlsx")
    )


if __name__ == "__main__":
    main()
