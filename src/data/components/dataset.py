from typing import Any, Dict, List, Optional, Tuple, Union

import geopandas as gpd
import numpy as np
import omegaconf
import pandas as pd
import rasterio
import shapely
import shapely.affinity
import torch

from src.utils.geoaffine import (
    convert_icell_to_tcell_v2,
    convert_vicell_to_vtcell,
    from_vecfeatures_determine_targets,
    sample_icell_from_raster_v2,
    sample_vicell_from_vector,
)


class Sampler:
    def __init__(self, aoi_name, mask_geometry=None):
        self.aoi_name = aoi_name
        self.mask_geometry = mask_geometry

    def sample(self, geometry, side_px, safft_world_to_icell=None):
        pass

    def get_mask(self, safft_world_to_icell, safft_icell_to_tcell, side_px):
        mask = shapely.affinity.affine_transform(
            self.mask_geometry, safft_world_to_icell
        )
        mask = shapely.affinity.affine_transform(mask, safft_icell_to_tcell)
        raster_mask = rasterio.features.rasterize(
            shapes=[(mask, 0)],
            out_shape=side_px,
            fill=255,
            dtype=np.uint8,
            all_touched=False,
        )
        return raster_mask


class TiffSampler(Sampler):
    def __init__(self, raster_path, aoi_name, iinter, mask_geometry=None):
        super().__init__(aoi_name, mask_geometry)
        self.raster_path = raster_path
        self.iinter = iinter

    def sample(
        self,
        geometry,
        side_px,
        safft_world_to_icell=None,
        safft_icell_to_tcell=None,
    ):
        icell = sample_icell_from_raster_v2(self.raster_path, geometry)
        tcell = convert_icell_to_tcell_v2(
            icell["img"], icell["square"], side_px, self.iinter
        )
        if not self.mask_geometry is None:
            area_mask = self.get_mask(
                icell["safft_world_to_icell"],
                tcell["safft_icell_to_tcell"],
                side_px,
            )
            tcell["img"].mask = np.logical_or(
                tcell["img"].mask, area_mask[..., None]
            )
        return (
            tcell["img"],
            icell["safft_world_to_icell"],
            tcell["safft_icell_to_tcell"],
            icell["window"],
        )


class VectorSampler(Sampler):
    def __init__(self, gf_feats, label_names, aoi_name):
        super().__init__(aoi_name)
        self.gf_feats = gf_feats
        self.label_names = label_names

    def sample(
        self, geometry, side_px, safft_world_to_icell, safft_icell_to_tcell
    ):
        assert (
            safft_world_to_icell is not None
        ), "Vector sampler called before any raster. Need to sample raster first"
        vicell = sample_vicell_from_vector(
            self.gf_feats, geometry, safft_world_to_icell
        )
        vtcell = convert_vicell_to_vtcell(
            vicell["gf_cfeats"],
            vicell["square"],
            safft_icell_to_tcell=safft_icell_to_tcell,
        )
        targets = from_vecfeatures_determine_targets(
            vtcell["gf_cfeats"], self.label_names, side_px
        )
        return (
            targets["ssegm_mask"],
            safft_world_to_icell,
            safft_icell_to_tcell,
            None,
        )


class TDataset_gtiff(torch.utils.data.Dataset):
    def __init__(
        self,
        gf_squares: Union[gpd.GeoDataFrame, Dict[str, gpd.GeoDataFrame]],
        inputSamplers: List[Sampler],
        targetSamplers: List[Sampler],
        mean: float,
        std: float,
        mean_type: str,
        WH: int = None,
        generate_targets=True,
        downsampleSampler: List[Sampler] = None,
        context_factor_m: Union[None, int] = None,
        context_size_px: Union[None, int] = None,
        return_debug_info: bool = False,
    ):
        """Torch dataset for loading Tiff geodata.

        Args:
            gf_squares (gpd.GeoDataFrame): GeoDataFrame of squares to load
            gf_feats (gpd.GeoDataFrame): GeoDataFrame of features to load
            hshades (Dict[str, Path]): Dictionary of path to rasterfield
            label_names (List[str]): List of label names
            mean (float): Mean for global normalization (if None computed)
            std (float): Standard deviation for normalization(if None computed)
            iinter (Interpolation): Interpolation type
            mean_type (str): Type of mean to use for normalization. Either global, local, avg_pool or max_pool
            mean_radius (int): Radius for neighbour based normalization(avg_pool or max_pool)
            WH (Union[None, Tuple[int, int]], optional): Size of the image. Defaults to None.
            return_debug_info (bool, optional): Whether to return debug info. Defaults to False.
            generate_target (bool, optional): Whether to generate target. Defaults to True.
        """
        self.gf_squares = gf_squares
        self.aoi_name = inputSamplers[0].aoi_name
        self.inputSamplers = inputSamplers
        self.targetSamplers = targetSamplers
        for sampler in self.inputSamplers:
            assert sampler.aoi_name == self.aoi_name, "sampler AOI conflict"
        for sampler in self.targetSamplers:
            assert sampler.aoi_name == self.aoi_name, "sampler AOI conflict"

        self.mean_type = mean_type
        self.mean = mean
        self.std = std
        self.WH = WH
        self.return_debug_info = return_debug_info
        if downsampleSampler is not None:
            self.downsampleSampler = downsampleSampler
        else:
            self.downsampleSampler = inputSamplers
        self.context_factor_m = context_factor_m
        self.context_size_px = (context_size_px, context_size_px)
        self.use_context = (
            context_size_px is not None and context_factor_m is not None
        )
        self.generate_targets = generate_targets

    def __len__(self):
        """Return the number of samples in the dataset.

        Returns:
            int: The number of samples in the dataset.
        """
        return len(self.gf_squares)

    def sample_inputs(self, geometry, side_px):
        safft_world_to_icell = None
        images = []
        window = None
        for sampler in self.inputSamplers:
            (
                img,
                safft_world_to_icell,
                safft_icell_to_tcell,
                new_window,
            ) = sampler.sample(geometry, side_px, safft_world_to_icell)
            if img.ndim == 2:
                img = np.expand_dims(img, axis=-1)
            images.append(img)
            if new_window is not None:
                assert (
                    window is None or window == new_window
                ), "Window conflict"
                window = new_window

        image = np.concatenate(images, axis=-1)
        mask = image.mask
        if mask is np.ma.nomask:
            mask = np.zeros_like(image, dtype=bool)

        # Normalize image. Missing values become exactly 0
        # remove global mean before conversion to tensor
        if isinstance(self.mean_type, str):
            if self.mean_type == "global":
                mean = self.mean
            else:
                mean = image.mean(axis=(0, 1))
        elif isinstance(self.mean_type, omegaconf.listconfig.ListConfig):
            mean_local = image.mean(axis=(0, 1))
            mean = [
                mean_local[i] if self.mean_type[i] == "local" else self.mean[i]
                for i in range(len(self.mean_type))
            ]
        else:
            raise ValueError(
                f"Mean type not supported: {self.mean_type} of type {type(self.mean_type)}"
            )
        mean = np.asarray(mean)
        img_nrm = (image.filled(mean) - mean) / self.std

        I_nrm = torch.as_tensor(img_nrm).to(torch.float32)
        assert (
            I_nrm.ndim != 2
        ), f"Image has to be 3D, current shape{I_nrm.shape}"
        # convert from HWC to CHW
        I_nrm = I_nrm.permute(2, 0, 1)

        return (
            I_nrm,
            mask,
            safft_world_to_icell,
            safft_icell_to_tcell,
            window,
        )

    def sample_target(
        self, geometry, safft_world_to_icell, safft_icell_to_tcell, WH
    ):
        targets = []
        for sampler in self.targetSamplers:
            (
                target,
                safft_world_to_icell,
                safft_icell_to_tcell,
                window,
            ) = sampler.sample(
                geometry, WH, safft_world_to_icell, safft_icell_to_tcell
            )
            if target.ndim == 2:
                target = np.expand_dims(target, axis=-1)
            targets.append(target)
        target = np.concatenate(targets, axis=-1)

        # convert from HWC to CHW
        target = torch.as_tensor(target)
        return target.permute(2, 0, 1).squeeze()

    def __getitem__(self, index):
        """Return the sample at the given index.

        Args:
            index (int): The index of the sample to return.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]: A tuple containing the sample, the target, and the metadata.
        """
        row_square = self.gf_squares.iloc[index]

        (
            I_nrm,
            mask,
            safft_world_to_icell,
            safft_icell_to_tcell,
            window,
        ) = self.sample_inputs(
            row_square["geometry"],
            self.WH,
        )
        if len(mask.shape) > 2:
            mask = mask.any(axis=-1)
        if mask.sum() != 0:
            print(f"Mask sum is not empty for index {index}")
        meta = {
            "index": index,
            "aoi_name": self.aoi_name,
            "square": row_square["geometry"],
            "safft_world_to_icell": safft_world_to_icell,
            "safft_icell_to_tcell": safft_icell_to_tcell,
            "window": window,
            "mask": mask,
        }

        if self.use_context:
            # increase square size by context_factor_m
            context_square = shapely.affinity.scale(
                row_square["geometry"],
                xfact=self.context_factor_m,
                yfact=self.context_factor_m,
            )
            meta["context_square"] = context_square

            I_context, _, _, _ = self.sample_inputs(
                self.downsampleSampler[row_square["aoi_name"]],
                context_square,
                self.context_size_px,
            )

        if self.generate_targets:
            targets = self.sample_target(
                row_square["geometry"],
                safft_world_to_icell,
                safft_icell_to_tcell,
                self.WH,
            )
            # set targets to 255 if out of bounds
            mask = torch.as_tensor(mask)
            if (
                targets.dtype == torch.float32
                or targets.dtype == torch.float64
            ):
                targets[mask.expand_as(targets)] = torch.inf
            elif targets.dtype in [
                torch.int64,
                torch.int32,
            ]:
                targets[mask.expand_as(targets)] = 255
            else:
                raise ValueError(
                    f"Target dtype not supported: {targets.dtype}"
                )
        else:
            targets = torch.Tensor([])
        if self.use_context:
            return I_nrm, targets, meta, I_context
        return I_nrm, targets, meta
