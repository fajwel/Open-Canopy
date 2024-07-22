import os
import shutil
from datetime import datetime

import contextily as ctx
import geopandas as gpd
import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import rootutils
from matplotlib.patches import Patch
from omegaconf import DictConfig, OmegaConf
from rasterio.mask import mask
from shapely.geometry import Polygon, box
from tqdm import tqdm

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from src.preprocessing.utils.utils import create_virtual_dataset, extract_and_mask_data


def create_square_from_centroid(centroid, side_length):
    """Create a square polygon centered at a given centroid with a specified side length."""
    half_side = side_length / 2
    x, y = centroid.x, centroid.y
    # Define the coordinates of the square
    square_coords = [
        (x - half_side, y - half_side),
        (x - half_side, y + half_side),
        (x + half_side, y + half_side),
        (x + half_side, y - half_side),
    ]
    # Create a polygon from the coordinates and return it
    return Polygon(square_coords)


@hydra.main(
    version_base=None,
    config_path="../config",
    config_name="create_splits_config",
)
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    if not cfg["version"]:
        version = str(datetime.now().strftime("%Y%m%d%H%M"))
    else:
        version = str(cfg["version"])

    save_dir = cfg.save_dir

    save_dir = os.path.join(save_dir, version)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # save the config next to the data
    OmegaConf.save(cfg, os.path.join(save_dir, "create_splits_config.yaml"))

    image_dir = cfg.image_dir
    lidar_dir = cfg.lidar_dir
    lidar_vrt_path = os.path.join(lidar_dir, "lidar", "full.vrt")
    lidar_classif_vrt_path = os.path.join(
        lidar_dir, "lidar", "full_classification_mask.vrt"
    )
    test_side_length = cfg.test_side_length
    buffer = cfg.buffer
    test_area = cfg.test_area
    val_proportion = cfg.val_proportion
    seed = cfg.seed

    year_list = [2021, 2022, 2023]
    image_dir_dict = {
        year: os.path.join(image_dir, str(year)) for year in year_list
    }

    lidar_gdf_path = os.path.join(save_dir, "lidar_gdf.parquet")
    if not cfg.overwrite and os.path.isfile(lidar_gdf_path):
        # Load lidar_gdf computed in previous run
        # WARNING do not use if options have changed
        lidar_gdf = gpd.read_parquet(lidar_gdf_path)
    else:
        image_df_dict = {}
        for year in year_list:
            image_path_list = np.array(
                [
                    # os.path.join(spot_dir, str(year), x)
                    os.path.join(image_dir_dict[year], x)
                    for x in os.listdir(image_dir_dict[year])
                    if not x.startswith(".") and x.endswith(".tif")
                ]
            )
            image_df_dict[year] = pd.DataFrame(
                image_path_list, columns=["image_path_" + str(year)]
            )

        # Add geometries
        def get_geometry_from_tiff(path):
            with rasterio.open(path) as dataset:
                # Get the bounding box of the image
                bbox = dataset.bounds
                # Create a polygon from the bounding box
                return box(bbox.left, bbox.bottom, bbox.right, bbox.top)

        for year in year_list:
            image_df_dict[year]["geometry_image_" + str(year)] = image_df_dict[
                year
            ]["image_path_" + str(year)].apply(get_geometry_from_tiff)
            # Convert the DataFrame to a GeoDataFrame
            image_df_dict[year] = gpd.GeoDataFrame(
                image_df_dict[year],
                geometry="geometry_image_" + str(year),
                crs="EPSG:2154",
            )
            # Problem: geometries do not exactly match for all images on different year so we sort by centroid
            # Calculate the x-coordinate of the centroid of each geometry
            image_df_dict[year]["centroid_x"] = image_df_dict[year][
                "geometry_image_" + str(year)
            ].centroid.x
            image_df_dict[year]["centroid_y"] = image_df_dict[year][
                "geometry_image_" + str(year)
            ].centroid.y
            # Sort the GeoDataFrame by the centroid x-coordinate
            image_df_dict[year] = image_df_dict[year].sort_values(
                by=["centroid_x", "centroid_y"], ascending=True
            )  # Change `ascending` as needed
            image_df_dict[year].drop(
                columns=["centroid_x", "centroid_y"], inplace=True
            )
            image_df_dict[year].reset_index(inplace=True, drop=True)

        # Concat dataframes, check how geometries are matching over years: compute intersection area ratio
        image_gdf = pd.concat(list(image_df_dict.values()), axis=1)

        image_gdf["intersection_ratio"] = (
            image_gdf["geometry_image_2021"]
            .intersection(image_gdf["geometry_image_2022"])
            .intersection(image_gdf["geometry_image_2023"])
            .area.divide(image_gdf["geometry_image_2021"].area)
        )
        assert image_gdf["intersection_ratio"].min() > 0.9

        if cfg.n_images is not None:
            image_gdf = image_gdf.head(cfg.n_images)

        # Load preprocessed lidar tiles
        lidar_gdf = gpd.read_file(
            os.path.join(
                os.path.expanduser(lidar_dir), "processed_geometries.geojson"
            )
        )
        # lidar_gdf = gpd.read_file(os.path.join(os.path.expanduser(lidar_dir)))
        print(f"There are {lidar_gdf.shape[0]} processed lidar tiles")

        lidar_gdf["lidar_year"] = lidar_gdf["acquisition_date"].apply(
            lambda x: int(x[0:4])
        )

        # Match lidar with spot for each year
        res = []
        for year in year_list:
            gdf = lidar_gdf.query("lidar_year==@year")
            # Only the active geometry of the left geodataframe is kept in the join
            image_gdf["geometry_image"] = image_gdf[
                "geometry_image_" + str(year)
            ].copy()
            res.append(
                gpd.sjoin(
                    gdf,
                    image_gdf.set_geometry("geometry_image"),
                    how="inner",
                    predicate="within",
                )
            )

        lidar_gdf = pd.concat(res, axis=0)
        print(
            f"There are {lidar_gdf.shape[0]} lidar geometries contained in spot images"
        )

        # Remove duplicates (in the spatial join, some lidar tiles can appear several times on overlapping image tiles)
        lidar_gdf.drop_duplicates(subset=["nom_pkk"], inplace=True)
        print(
            f"There are {lidar_gdf.shape[0]} lidar geometries contained in spot images after removing duplicates (that are in overlapping spot tiles)"
        )

        # Attribute image_path with matching year to each lidar tile
        lidar_gdf["image_path"] = lidar_gdf.apply(
            lambda row: row["image_path_" + str(row["lidar_year"])], axis=1
        )

        # Check image_path match lidar year
        for year in year_list:
            ix = np.where(lidar_gdf["lidar_year"] == year)[0]
            if len(ix):
                assert str(year) in lidar_gdf.iloc[ix[0]]["image_path"]

        # Exclude tiles of idar that are on zeros of spot (here zeros of spot are on the edges, equivalent to no value)
        # For all lidar tiles, check corresponding image patch, count the number of zeros, discard if > 10

        def count_image_zeros(image_path, geometry, band=0):
            with rasterio.open(image_path) as dataset:
                # all_touched option ensures to keep edges
                data, _ = mask(dataset, geometry, crop=True, all_touched=True)
            return data.size - np.count_nonzero(data)

        lidar_gdf["n_zeros_image"] = [
            count_image_zeros(
                lidar_gdf["image_path"].iloc[i],
                lidar_gdf["geometry"].iloc[[i]],
                band=0,
            )
            for i in tqdm(
                range(lidar_gdf.shape[0]), desc="Counting zeros in each patch"
            )
        ]

        n_max_zeros = cfg.n_max_zeros
        ix_non_zero = lidar_gdf["n_zeros_image"] < n_max_zeros
        print(f"There are {ix_non_zero.sum()} patches without zeros")

        # # Create a figure and an axis object
        # fig, ax = plt.subplots(figsize=(10, 10))

        # # Plot the first geometry column and create a Patch for the legend
        # lidar_gdf.set_geometry("geometry").plot(ax=ax, color="green", alpha=0.7)

        # # Plot the first geometry column and create a Patch for the legend
        # lidar_gdf.set_geometry("geometry").loc[ix_non_zero].plot(ax=ax, color="red", alpha=0.7)

        # # Add the Contextily basemap (map of France)
        # ctx.add_basemap(ax, crs=lidar_gdf.crs.to_string())

        # # Set title and hide axes
        # ax.set_title("Taking out tiles that are on edges of images")
        # ax.set_axis_off()

        lidar_gdf = lidar_gdf.loc[ix_non_zero]

        # full geometry
        combined_full = lidar_gdf.set_geometry("geometry").unary_union

        # Sample test tiles from lidar grid
        lidar_gdf["test_tile"] = lidar_gdf["geometry"].centroid.apply(
            create_square_from_centroid, args=(test_side_length,)
        )
        lidar_gdf["test_tile_buffered"] = lidar_gdf["geometry"].centroid.apply(
            create_square_from_centroid, args=(test_side_length + 2 * buffer,)
        )

        # Check whether each new geometry is contained in the full combined geometry
        is_contained = lidar_gdf["test_tile"].within(combined_full)

        # Sample test tiles
        lidar_gdf["is_contained"] = is_contained
        n_contained = lidar_gdf["is_contained"].sum()

        print(
            f"There are {n_contained} 1km2 tiles which can be extended to "
            f"tiles of side {test_side_length + 2 * buffer}m contained in the dataset of size {lidar_gdf.shape[0]}"
        )

        full_area = lidar_gdf.shape[0]  # in km2
        square_area = test_side_length**2 / 1e6  # in km2

        n_sample_test = int(test_area // square_area)

        # Sample test tiles
        print(f"Sampling {n_sample_test} test tiles")
        lidar_test = lidar_gdf.loc[lidar_gdf["is_contained"]].sample(
            n=n_sample_test, random_state=seed
        )
        lidar_test_with_overlap = lidar_test.copy()

        # Drop tiles that intersect with another one
        lidar_test["intersect"] = False
        ix_test = lidar_test.index
        for i in range(lidar_test.shape[0]):
            if not lidar_test.loc[ix_test[i], "intersect"]:
                for j in range(i + 1, lidar_test.shape[0]):
                    if not lidar_test.loc[ix_test[j], "intersect"]:
                        intersection_area = (
                            lidar_test.loc[ix_test[i], "test_tile"]
                            .intersection(
                                lidar_test.loc[ix_test[j], "test_tile"]
                            )
                            .area
                        )
                        if intersection_area > 0:
                            lidar_test.loc[ix_test[j], "intersect"] = True

        lidar_test = lidar_test[~lidar_test["intersect"]]

        print(
            f"There are {lidar_test.shape[0]} test tiles of side {test_side_length} sampled "
            f"after dropping overlapping ones"
        )

        # # Create a figure and an axis object
        # fig, ax = plt.subplots(figsize=(10, 10))

        # # Plot the first geometry column and create a Patch for the legend
        # lidar_test_with_overlap.set_geometry("test_tile").plot(ax=ax, color="blue", alpha=0.7)

        # # Plot the first geometry column and create a Patch for the legend
        # lidar_test.set_geometry("test_tile").plot(ax=ax, color="red", alpha=0.7)

        # # Add the Contextily basemap (map of France)
        # ctx.add_basemap(ax, crs=lidar_gdf.crs.to_string())

        # # Set title and hide axes
        # ax.set_title("Taking out overlapping test tiles")
        # ax.set_axis_off()

        # Drop test tiles that are on different years (needed for evaluation of change)
        # for each test tile retrieve all acquisition years: intersect test_geometry with geometry to get associated tiles
        def check_unique_year_coverage(row):
            """Check if a test_tile covers severals years."""
            tiles = lidar_gdf.loc[
                lidar_gdf.set_geometry("geometry").within(row["test_tile"])
            ]
            year_coverage = tiles["lidar_year"].unique()
            return len(year_coverage) == 1

        lidar_test["unique_year_coverage"] = lidar_test.apply(
            check_unique_year_coverage, axis=1
        )

        lidar_test = lidar_test.loc[lidar_test["unique_year_coverage"]]
        print(
            f"Removing test tiles that were acquired on several years, keeping {lidar_test.shape[0]} test tiles"
        )

        # Tag 1km2 tiles that are in the test set
        lidar_gdf["split"] = "buffer"
        combined_test = lidar_test.set_geometry("test_tile").unary_union
        lidar_gdf.loc[
            lidar_gdf["geometry"].within(combined_test), "split"
        ] = "test"
        assert lidar_gdf.query("split=='test'").shape[0] == lidar_test.shape[
            0
        ] * ((test_side_length / 1000) ** 2)
        n_test = lidar_gdf.query("split=='test'").shape[0]
        print(
            f"There are {lidar_test.shape[0]} test tiles sampled after dropping overlapping ones, "
            f"which corresponds to {n_test} km2"
        )

        n_sample_test_after_filtering = lidar_test.shape[0]
        area_test = n_sample_test_after_filtering * square_area
        area_test_buffered = (
            n_sample_test_after_filtering
            * ((test_side_length + 2 * buffer) ** 2)
            / 1e6
        )
        area_ratio_test = area_test / full_area
        area_ratio_test_buffered = area_test_buffered / full_area
        print(
            f" test area (km2): {area_test} \n buffered test area (km2): {area_test_buffered} \n"
            f" test area ratio: {area_ratio_test:.2f} \n buffered test area ratio {area_ratio_test_buffered:.2f}"
        )

        # get tiles that are not contained in (extended) test tiles
        combined_test_buffered = lidar_test.set_geometry(
            "test_tile_buffered"
        ).unary_union
        lidar_gdf.loc[
            ~lidar_gdf["geometry"].within(combined_test_buffered), "split"
        ] = "train"
        lidar_train_val = lidar_gdf.query("split=='train'")
        # Sample val tiles
        n_sample_val = int(val_proportion * lidar_train_val.shape[0])
        lidar_val = lidar_train_val.sample(n=n_sample_val, random_state=seed)
        lidar_gdf.loc[lidar_gdf.index.isin(lidar_val.index), "split"] = "val"

        # Create a figure and an axis object
        lidar_gdf.set_geometry("geometry", inplace=True)
        fig, ax = plt.subplots(figsize=(10, 10))

        lidar_gdf.query("split=='buffer'").plot(ax=ax, color="red", alpha=0.7)
        red_patch = Patch(color="red", label="buffer")

        lidar_gdf.query("split=='test'").plot(ax=ax, color="orange", alpha=0.7)
        orange_patch = Patch(color="orange", label="test")

        lidar_gdf.query("split=='train'").plot(ax=ax, color="green", alpha=0.7)
        green_patch = Patch(color="green", label="train")

        lidar_gdf.query("split=='val'").plot(ax=ax, color="blue", alpha=0.7)
        blue_patch = Patch(color="blue", label="val")

        # Add the Contextily basemap (map of France)
        ctx.add_basemap(ax, crs=lidar_gdf.crs.to_string())

        # Add legend with custom patches
        ax.legend(handles=[blue_patch, green_patch, orange_patch, red_patch])

        # Set title and hide axes
        ax.set_title("Splits")
        ax.set_axis_off()

        # save plot
        fig.savefig(
            os.path.join(save_dir, "splits.png"),
            bbox_inches="tight",
            pad_inches=0,
        )

        # Save lidar_gdf for later use
        lidar_gdf.to_parquet(lidar_gdf_path)

    # Save images and lidar
    for year in year_list:
        gdf_year = lidar_gdf.query("lidar_year==@year")
        if gdf_year.shape[0] > 0:
            save_dir_spot = os.path.join(save_dir, str(year), "spot")
            if not os.path.exists(save_dir_spot):
                os.makedirs(save_dir_spot)

            save_dir_lidar = os.path.join(save_dir, str(year), "lidar")
            if not os.path.exists(save_dir_lidar):
                os.makedirs(save_dir_lidar)

            save_dir_lidar_classif = os.path.join(
                save_dir, str(year), "lidar_classification"
            )
            if not os.path.exists(save_dir_lidar_classif):
                os.makedirs(save_dir_lidar_classif)

            image_list = gdf_year["image_path"].unique()
            image_dest_list = []
            lidar_dest_list = []
            classif_dest_list = []

            for file_path in tqdm(
                image_list, desc=f"copying images from {year}"
            ):
                image_name = os.path.basename(file_path)
                dest_file_path = os.path.join(save_dir_spot, image_name)
                image_dest_list.append(dest_file_path)
                if cfg.overwrite or (not os.path.isfile(dest_file_path)):
                    shutil.copy(file_path, dest_file_path)
                    print(f"Successfully saved image {image_name}")
                # For lidar, extract and save digital height model corresponding to each image
                lidar_dest_path = os.path.join(
                    save_dir_lidar, image_name.replace("pansharpened", "lidar")
                )
                lidar_dest_list.append(lidar_dest_path)
                if cfg.overwrite or (not os.path.isfile(lidar_dest_path)):
                    geometry = gdf_year.query("image_path==@file_path")[
                        "geometry_image_" + str(year)
                    ].iloc[0]
                    extract_and_mask_data(
                        lidar_vrt_path, geometry, lidar_dest_path
                    )
                    print(
                        f"Successfully saved lidar height corresponding to {image_name}"
                    )
                classif_dest_path = os.path.join(
                    save_dir_lidar_classif,
                    image_name.replace("pansharpened", "lidar_classification"),
                )
                classif_dest_list.append(classif_dest_path)
                if cfg.overwrite or (not os.path.isfile(classif_dest_path)):
                    geometry = gdf_year.query("image_path==@file_path")[
                        "geometry_image_" + str(year)
                    ].iloc[0]
                    extract_and_mask_data(
                        lidar_classif_vrt_path, geometry, classif_dest_path
                    )
                    print(
                        f"Successfully saved lidar classification corresponding to {image_name}"
                    )

            # Create a vrt for each list of images
            create_virtual_dataset(
                image_dest_list, os.path.join(save_dir, str(year), "spot.vrt")
            )
            create_virtual_dataset(
                lidar_dest_list, os.path.join(save_dir, str(year), "lidar.vrt")
            )
            create_virtual_dataset(
                classif_dest_list,
                os.path.join(save_dir, str(year), "lidar_classification.vrt"),
            )

    # Clean lidar_gdf and save it
    columns_to_save = [
        "url_telech",
        "X",
        "Y",
        "acquisition_date",
        "n_points",
        "geometry",
        "lidar_year",
        "image_path",
        "split",
    ]
    lidar_gdf = lidar_gdf[columns_to_save]

    lidar_gdf = lidar_gdf.rename(
        columns={
            "url_telech": "lidar_url",
            "acquisition_date": "lidar_acquisition_date",
            "n_points": "n_lidar_points",
            "image_path": "image_name",
        },
    )
    lidar_gdf["image_name"] = lidar_gdf["image_name"].apply(
        lambda x: os.path.basename(x)
    )
    lidar_gdf.to_file(
        os.path.join(save_dir, "geometries.geojson"),
        driver="GeoJSON",
        crs=2154,
    )


if __name__ == "__main__":
    main()
