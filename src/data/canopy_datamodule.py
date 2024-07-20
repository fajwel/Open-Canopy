import gc
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import geopandas as gpd
import hydra
import numpy as np
import pandas as pd
import rasterio
import shapely
import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms

from src.data.components import TDataset_gtiff, TiffSampler, compute_data_stat
from src.data.components.dataprep import separate_geometry
from src.utils import mkdir, pylogger
from src.utils.data import gpkg_save
from src.utils.geoaffine import (
    Interpolation,
    itm_collate,
    sample_grid_squares_from_aoi_v2,
    sample_random_squares_from_aoi_v2,
)

log = pylogger.RankedLogger(__name__, rank_zero_only=True)


class GEODataModule(LightningDataModule):
    """`LightningDataModule`

    A `LightningDataModule` implements 7 key methods:

    ```python
        def prepare_data(self):
        # Things to do on 1 GPU/TPU (not on every GPU/TPU in DDP).
        # Download data, pre-process, split, save to disk, etc...

        def setup(self, stage):
        # Things to do on every process in DDP.
        # Load data, set variables, etc...

        def train_dataloader(self):
        # return train dataloader

        def val_dataloader(self):
        # return validation dataloader

        def test_dataloader(self):
        # return test dataloader

        def predict_dataloader(self):
        # return predict dataloader

        def teardown(self, stage):
        # Called on every process in DDP.
        # Clean up after fit or test.
    ```

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://lightning.ai/docs/pytorch/latest/data/datamodule.html
    """

    def __init__(
        self,
        geometry_path: str = "data/",
        layers_names: List[str] = "dtm",
        feats: str = "py@grab('../050_data/feats/gf_feats.gpkg')",
        labelkind: str = "easy3",
        batch_size: int = 16,
        raster_targets: str = "",
        sample_multiplier: int = 1,
        imageside: int = 256,
        imagesize: int = 256,
        mean: float = None,
        std: float = None,
        mean_type: str = "local",
        mean_radius: int = 40,
        num_workers: int = 0,
        iinter: Interpolation = 1,  # LINEAR
        pin_memory=True,
        tsize_base=None,
        tsize_enum_sizes=[1],
        tsize_enum_probs=None,
        tsize_range_frac=0,
        tsize_range_sizes=[0.5, 2],
        trot_angle=90,
        trot_prob=0.5,
        min_overlap=0.2,
        test_overlap=0.5,
    ) -> None:
        """Initialize a `LightningDataModule`.

        Args:
            geometry_path (str, optional): Path to aois. Defaults to "data/".
            batch_size (int, optional): Batch size. Defaults to 16.
            raster_targets (str, optional): Path to precomputer raster target. If empty targets will be rasterize. Defaults to "".
            sample_multiplier (int, optional): How many time an area should be sample (in esperance) per epoch. Defaults to 1.
            imageside (int, optional): Size of a sample in meters. Defaults to 256.
            imagesize (int, optional): Size of a sample in pixel. Defaults to 256.
            mean (float, optional): Mean for global normalization (if None computed). Defaults to None.
            std (float, optional): Standard deviation for normalization(if None computed). Defaults to None.
            mean_type (str, optional): Type of mean to use for normalization. Either global, local, avg_pool or max_pool. Defaults to "local".
            mean_radius (int, optional): radius for neighbour based normalization(avg_pool or max_pool). Defaults to 40.
            num_workers (int, optional): Number of parrarel worker to load the datasets. Defaults to 0.
            iinter (Interpolation, optional): Interpolation type. Defaults to 1.
            tsize_base (_type_, optional): For training : Default image side before augment.None for equal to image size dimension in meters. Defaults to None.
            tsize_enum_sizes (list, optional): For training : Randomly multiply the size by a factor in the sizeswith probs. Defaults to [1].
            tsize_enum_probs (_type_, optional): For training : Randomly multiply the size by a factor in the sizes with probs. Defaults to None.
            tsize_range_frac (int, optional): For training : Randomly sample frac of the train sample with unform size in the range. Defaults to 0.
            tsize_range_sizes (list, optional): For training : Randomly sample frac of the train sample with unform size in the range. Defaults to [0.5, 2].
            trot_angle (int, optional): For training : Randomly rotate. Defaults to 90.
            trot_prob (float, optional): For training : Randomly rotate. Defaults to 0.5.
            min_overlap (float, optional): For training : Minimum area of a sample that must be inside the raster. Defaults to 0.2.
        """

        super().__init__()
        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.layers_names = layers_names
        if isinstance(self.layers_names, str):
            self.layers_names = [self.layers_names]

        # data transformations
        self.transforms = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        self.batch_size_per_device = batch_size
        self.epoch = 0

        self.rgen = np.random.default_rng(np.random.randint(0, 2**32 - 1))
        log.info(f"Using {self.rgen=} for training dataset")

        try:
            self.output_dir = (
                hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
            )
        except AttributeError:
            self.output_dir = "./logs"
        log.info(f"Using {self.output_dir=}")

        self.mean_radius = mean_radius

    def prepare_data(self) -> None:
        """Download data if needed. Lightning ensures that `self.prepare_data()` is called only
        within a single process on CPU, so you can safely add your downloading logic within. In
        case of multi-node training, the execution of this hook depends upon
        `self.prepare_data_per_node()`.

        Do not use it to assign state (self.x = y).
        """

        # Download dataset if not present

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice! Also, it is called after
        `self.prepare_data()` and there is a barrier in between which ensures that all the processes proceed to
        `self.setup()` once the data is prepared and available for use.

        Args:
            stage(str): The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """
        # Set multiprocessing method
        torch.multiprocessing.set_start_method("spawn", force=True)

        fold_evalprep = Path(self.output_dir + "/eval_prepared")

        geometries_data = gpd.read_file(self.hparams.geometry_path)
        self.datadir = Path(self.hparams.geometry_path).parent

        # group per year
        self.years = list(set(geometries_data["lidar_year"]))
        relevent_split = ["train", "val", "test"]
        if stage == "validate":
            relevent_split = ["val"]
        elif stage == "test":
            relevent_split = ["test", "predict"]
        self.gf_aois = {
            "name": [],
            "split": [],
            "year": [],
            "geometry": [],
            "vrt": [],
            "vrt_spot": [],
            "vrt_lidar": [],
            "vrt_class": [],
        }
        for year in self.years:
            geometries_year = geometries_data.query(f"lidar_year == {year}")
            for split in relevent_split:
                geometries_split = geometries_year.query(f"split == '{split}'")
                self.gf_aois["name"].append(f"{split}_{year}")
                self.gf_aois["split"].append(split)
                self.gf_aois["year"].append(year)
                self.gf_aois["geometry"].append(
                    shapely.union_all(
                        geometries_split["geometry"], grid_size=100
                    )
                )
                self.gf_aois["vrt"].append(self.datadir / str(year))
                self.gf_aois["vrt_spot"].append(
                    self.datadir / str(year) / "spot.vrt"
                )
                self.gf_aois["vrt_lidar"].append(
                    self.datadir / str(year) / "lidar.vrt"
                )
                self.gf_aois["vrt_class"].append(
                    self.datadir / str(year) / "lidar_classification.vrt"
                )
        # Create geodataframe with areas of interest
        self.gf_aois = gpd.GeoDataFrame(
            self.gf_aois, crs=geometries_data.crs
        ).set_index("name")

        # get target profile
        self.raster_targets = {}
        for aoi_name, row_aoi in self.gf_aois.iterrows():
            with rasterio.open(row_aoi["vrt_spot"]) as src:
                self.raster_targets[aoi_name] = {"profile": src.profile}

        # Assign number of samples to each gf_aois proportional to area (divisible by batchsize)
        batch_size = self.hparams.batch_size
        sample_multiplier = self.hparams.sample_multiplier
        imageside = self.hparams.imageside
        self.gf_aois["train_samples"] = 0
        for id, row in self.gf_aois.iterrows():
            self.gf_aois.loc[id, "train_samples"] = (
                np.ceil(
                    row["geometry"].area
                    / imageside**2
                    * sample_multiplier
                    / batch_size
                ).astype(int)
                * batch_size
            )

        if self.hparams.mean is None or self.hparams.std is None:
            log.info("Computing dataset mean and std for normalisation")
            mean, std = compute_data_stat(
                self.gf_aois.query("split in ['train', 'val']"),
                ["vrt_spot"],
                scale=12,
            )
        else:
            mean, std = self.hparams.mean, self.hparams.std
        log.info(f"global {mean=}, {std=}")

        iinter = Interpolation(self.hparams.iinter)
        num_workers = self.hparams.num_workers

        # Create sampler for each aois
        self.samplers = {}
        for aoi_name, row_aoi in self.gf_aois.iterrows():
            # Create Sampler
            inputSamplers = [
                TiffSampler(
                    row_aoi["vrt_spot"],
                    aoi_name,
                    iinter,
                    mask_geometry=row_aoi["geometry"],
                )
            ]
            targetSamplers = [
                TiffSampler(row_aoi["vrt_lidar"], aoi_name, iinter),
                TiffSampler(
                    row_aoi["vrt_class"],
                    aoi_name,
                    0,  # Nearest Neighbour
                ),
            ]
            self.samplers[aoi_name] = [
                inputSamplers,
                targetSamplers,
            ]

        WH = (self.hparams.imagesize, self.hparams.imagesize)

        # Instantiate reference grid-evaluating dataloaders for evaluation
        gf_square_dict = self.generate_grid(num_workers=num_workers)

        self.dloaders_gridsampling = {}
        # Create grid dataloaders for evaluation
        for aoi_name, gf_squares in gf_square_dict.items():
            gpkg_save(
                gf_squares,
                mkdir(fold_evalprep / "squares"),
                f"{aoi_name}_squares_r{self.trainer.global_rank}",
            )

            inputSamplers, targetSamplers = self.samplers[aoi_name]
            tdata_eval = TDataset_gtiff(
                gf_squares,
                inputSamplers,
                targetSamplers,
                mean=0,
                std=1,
                mean_type="global",
                WH=WH,
                generate_targets=True,
                return_debug_info=False,
            )
            tloader_eval = DataLoader(
                tdata_eval,
                batch_size=self.hparams.batch_size,
                num_workers=self.hparams.num_workers,
                collate_fn=itm_collate,
                drop_last=False,
                pin_memory=self.hparams.pin_memory,
                # persistent_workers=True,
            )

            self.dloaders_gridsampling[aoi_name] = tloader_eval

        # Check targets, assign number of squares inside
        self.gf_aois["eval_samples"] = {
            k: len(v.dataset.gf_squares)
            for k, v in self.dloaders_gridsampling.items()
        }

        # Print combined data stats
        for split, iids in self.gf_aois.groupby("split").groups.items():
            gf = self.gf_aois.loc[iids].drop(columns=["vrt"])
            with pd.option_context(
                "display.max_rows",
                None,
                "display.max_columns",
                None,
                "display.width",
                512,
            ):
                log.info(
                    "Split: {}, Area: {:.3f} km2, Samples train={} eval={}:\n{}".format(
                        split,
                        gf.area.sum() / 10**6,
                        gf.train_samples.sum(),
                        gf.eval_samples.sum(),
                        gf,
                    )
                )

        self.iinter = iinter
        self.WH = WH

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        Returns:
            The train dataloader.
        """
        log.info("Generating train dataloader")
        tstart = time.time()
        gc.collect()
        tsize_base = self.hparams.tsize_base
        if tsize_base is None:
            tsize_base = self.hparams.imageside
        tsize_enum_sizes = self.hparams.tsize_enum_sizes
        tsize_enum_probs = self.hparams.tsize_enum_probs
        tsize_range_frac = self.hparams.tsize_range_frac
        tsize_range_sizes = self.hparams.tsize_range_sizes
        trot_angle = self.hparams.trot_angle
        trot_prob = self.hparams.trot_prob
        min_overlap = self.hparams.min_overlap

        # Create squares dataset and appropriate dataset/dataloader
        square_dict = self.generate_random_sample(
            tsize_base,
            tsize_enum_sizes,
            tsize_enum_probs,
            tsize_range_frac,
            tsize_range_sizes,
            trot_angle,
            trot_prob,
            min_overlap,
            num_workers=self.hparams.num_workers,
        )

        log.info(f"Generated squares in {time.time() - tstart:.2f}s")
        tdataloader = time.time()

        gf_train_squares = []
        train_datasets = []
        for aoi_name, row_aoi in self.gf_aois[
            self.gf_aois["split"] == "train"
        ].iterrows():
            # create aoi dataset
            gf_aoi_squares = square_dict[aoi_name]
            inputSamplers, targetSamplers = self.samplers[aoi_name]
            tdata_train = TDataset_gtiff(
                gf_aoi_squares,
                inputSamplers,
                targetSamplers,
                mean=0,
                std=1,
                mean_type="global",
                WH=self.WH,
                generate_targets=True,
                return_debug_info=False,
            )
            train_datasets.append(tdata_train)

            gf_train_squares.append(gf_aoi_squares)

        gf_train_squares = pd.concat(
            gf_train_squares, axis=0, ignore_index=True
        )
        # Log for debugging
        # dep_fold = (
        #     self.output_dir
        #     + f"/runtime/e{self.epoch}r{self.trainer.global_rank}"
        # )
        # gpkg_save(gf_train_squares, dep_fold, "train_squares")

        data_train = torch.utils.data.ConcatDataset(train_datasets)
        dload_train = DataLoader(
            data_train,
            shuffle=True,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            collate_fn=itm_collate,
            drop_last=True,  # Just a safeguard
            pin_memory=self.hparams.pin_memory,
            # persistent_workers=True,
        )
        log.info(f"Generated dataloader in {time.time() - tdataloader:.2f}s")

        self.epoch += 1

        log.info(f"Generated train dataloader in {time.time() - tstart:.2f}s")
        return dload_train

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        Returns:
            The validation dataloader.
        """
        gc.collect()
        validation_names = self.gf_aois.query("split == 'val'").index.tolist()
        train_names = []
        # self.gf_aois.query("split == 'train'").index.tolist()
        valLoaders = [
            self.dloaders_gridsampling[name]
            for name in validation_names + train_names
        ]
        return valLoaders

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        Returns:
            The test dataloader.
        """
        gc.collect()
        test_names = self.gf_aois.query("split == 'test'").index.tolist()
        testLoaders = [self.dloaders_gridsampling[name] for name in test_names]
        return testLoaders

    def predict_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        Returns:
            The test dataloader.
        """
        pred_names = self.gf_aois.query("split == 'pred'").index.tolist()
        predLoaders = [self.dloaders_gridsampling[name] for name in pred_names]
        return predLoaders

    def on_after_batch_transfer(self, batch, dataloader_idx):
        """Lightning hook that is called after data is moved to device. Used to normalize the data.

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]): The current batch of data, consisting of a tensor of data, a tensor of targets, and a dictionary of metadata.
            dataloader_idx (int): The index of the dataloader that produced this batch.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]: The transformed batch, consisting of a tensor of normalized data, a tensor of targets, and a dictionary of metadata.
        """
        return batch

    def teardown(self, stage: Optional[str] = None) -> None:
        """Lightning hook for cleaning up after `trainer.fit()`, `trainer.validate()`,
        `trainer.test()`, and `trainer.predict()`.

        Args:
            stage (Optional[str]): The stage being torn down. Either "fit", "validate", "test", or "predict".
                Defaults to None.
        """
        pass

    def state_dict(self) -> Dict[Any, Any]:
        """Called when saving a checkpoint. Implement to generate and save the datamodule state.

        Returns:
                A dictionary containing the datamodule state that you want to save.
        """
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Called when loading a checkpoint. Implement to reload datamodule state given datamodule
        `state_dict()`.

        Args:
            state_dict (Dict[str, Any]): The datamodule state returned by `self.state_dict()`.

        Returns:
            None
        """
        pass

    def generate_grid(self, num_workers=1):
        """Instantiate reference grid-evaluating dataloaders for evaluation Also populate the
        transform field of the gf_aois."""
        gf_squares_dict = {}
        self.gf_aois["transform"] = None

        print("Generating grid-evaluating dataloaders")
        pool = torch.multiprocessing.Pool(max(num_workers, 1))
        results = {}
        for aoi_name, row_aoi in self.gf_aois.iterrows():
            WH = (self.hparams.imagesize, self.hparams.imagesize)

            with rasterio.open(row_aoi["vrt_spot"]) as src:
                raster_afft = src.profile["transform"]
                self.gf_aois.loc[aoi_name, "transform"] = raster_afft
            poly_aoi = row_aoi["geometry"]
            if row_aoi["split"] == "train" or row_aoi["split"] == "val":
                eval_stride = self.hparams.imageside
                shift = 0, 0
            elif row_aoi["split"] == "test" or row_aoi["split"] == "pred":
                eval_stride = self.hparams.imageside - int(
                    self.hparams.imageside * self.hparams.test_overlap
                )
                shift = (-eval_stride // 2, -eval_stride // 2)
            # Split multipolygon into clusters of polygons
            if isinstance(poly_aoi, shapely.geometry.MultiPolygon):
                poly_list = separate_geometry(poly_aoi)
            else:
                poly_list = [poly_aoi]

            results[aoi_name] = []
            for poly in poly_list:
                if poly.is_empty:
                    continue
                result = pool.apply_async(
                    sample_grid_squares_from_aoi_v2,
                    args=(
                        poly,
                        self.hparams.imageside,
                        self.gf_aois.crs,
                        eval_stride,
                        shift,
                        raster_afft,
                    ),
                )

                results[aoi_name].append(result)

        for aoi_name, result_list in results.items():
            if len(result_list) == 0:
                gf_squares_dict[aoi_name] = gpd.GeoDataFrame()
            else:
                gf_squares = []
                for result in result_list:
                    gf_squares.append(result.get())

                gf_squares = pd.concat(gf_squares, axis=0, ignore_index=True)
                gf_squares["aoi_name"] = aoi_name
                gf_squares_dict[aoi_name] = gf_squares

        pool.close()
        pool.join()

        return gf_squares_dict

    def generate_random_sample(
        self,
        tsize_base,
        tsize_enum_sizes,
        tsize_enum_probs,
        tsize_range_frac,
        tsize_range_sizes,
        trot_angle,
        trot_prob,
        min_overlap,
        num_workers=1,
    ):
        pool = torch.multiprocessing.Pool(max(num_workers, 1))
        results = {}
        square_dict = {}
        for aoi_name, row_aoi in self.gf_aois[
            self.gf_aois["split"] == "train"
        ].iterrows():
            rgen = np.random.default_rng(self.rgen.integers(0, 2**32 - 1))
            poly_aoi = row_aoi["geometry"]
            raster_afft = row_aoi["transform"]
            n_samples = row_aoi["train_samples"]
            area = poly_aoi.area
            for geom in separate_geometry(poly_aoi):
                n_samples_geom = int(n_samples * geom.area / area)
                result = pool.apply_async(
                    sample_random_squares_from_aoi_v2,
                    args=(
                        geom,
                        n_samples_geom,
                        rgen,
                        tsize_base,
                        tsize_enum_sizes,
                        tsize_enum_probs,
                        tsize_range_frac,
                        tsize_range_sizes,
                        trot_angle,
                        trot_prob,
                        min_overlap,
                        raster_afft,
                    ),
                )
                if aoi_name in results.keys():
                    results[aoi_name].append(result)
                else:
                    results[aoi_name] = [result]

        for aoi_name, result_list in results.items():
            squares = []
            for result in result_list:
                squares.extend(result.get())

            gf_aoi_squares = gpd.GeoDataFrame(
                geometry=squares, crs=self.gf_aois.crs
            )
            gf_aoi_squares["aoi_name"] = aoi_name
            square_dict[aoi_name] = gf_aoi_squares
        pool.close()
        pool.join()
        return square_dict


if __name__ == "__main__":
    _ = GEODataModule()
