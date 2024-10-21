import logging
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import geopandas as gpd
import matplotlib as mpl
import numpy as np
import rasterio.features
from rasterio.windows import Window
from shapely.affinity import affine_transform

from utils import mkdir

log = logging.getLogger(__name__)


def get_aois_v3(path) -> gpd.GeoDataFrame:
    gf_aois = gpd.read_file(path).set_index("aoi_name")
    ppath = Path(path).parent
    if (ppath / "hillshades").exists():
        gf_aois["path_hs"] = {
            x.with_suffix("").name: x for x in (ppath / "hillshades").glob("*.tif")
        }
        # assert not gf_aois['path_hs'].isna().any()
    if (ppath / "dtms").exists():
        gf_aois["path_dtm"] = {
            x.with_suffix("").name: x for x in (ppath / "dtms").glob("*.tif")
        }
        # assert not gf_aois['path_dtm'].isna().any()
    if (ppath / "mstp").exists():
        gf_aois["path_mstp"] = {
            x.with_suffix("").name: x for x in (ppath / "mstp").glob("*.tif")
        }
        # assert not gf_aois['path_mstp'].isna().any()
    return gf_aois


def get_feats_with_proper_labels(path, labelkind):
    """

    Args:
        path (str): path to features file (vector)
        labelkind (Dict): how should the labels be mapped

    Returns:
        DataFrame: features with proper labels
    """
    from omegaconf import DictConfig

    gf_feats = gpd.read_file(path)
    assert gf_feats.is_valid.all()
    if isinstance(labelkind, str):
        # old preconfigure labelkind
        if labelkind == "easy2":
            gf_feats["label"] = gf_feats["label_easy5"]
            label_names = ["mound", "temple_site"]
        elif labelkind == "easy3":
            gf_feats["label"] = gf_feats["label_easy5"]
            gf_feats.loc[
                gf_feats["label"].isin(["reservoir", "watercourse"]), "label"
            ] = "water"
            label_names = ["mound", "water", "temple_site"]
        else:
            raise RuntimeError()
        return gf_feats, label_names
    elif isinstance(labelkind, dict) or isinstance(labelkind, DictConfig):
        # new labelkind
        # create mapping from dict
        mapping_dict = {}
        for label_name, labels in labelkind.items():
            for label in labels:
                mapping_dict[label] = label_name
        gf_feats["label"] = gf_feats["type_cls6_str"]
        gf_feats["label"] = gf_feats["label"].map(mapping_dict)

        label_names = list(labelkind.keys())
        return gf_feats, label_names
    else:
        raise ValueError(f"Unsupported type for labelkind : {type(labelkind)}")


def set_sparse_rasterio_profile(profile, dtype):
    profile = profile.copy()
    profile["sparse_ok"] = True
    profile["dtype"] = dtype
    if dtype == "float32":
        profile["nodata"] = -9999
    elif dtype == "uint8":
        profile["nodata"] = 255
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")
    return profile


class Window_writer_rasterio:
    # Single channel rasterio writer (can support multiple channels  with the indexes argument)
    def __init__(self, path, profile, dtype=None, indexes=1):
        self.path = path
        self.profile = set_sparse_rasterio_profile(profile, dtype)
        if dtype == "float32" or dtype == "float16":
            self.profile["compress"] = "ZSTD"
            self.dst = rasterio.open(str(path), "w+", **self.profile, predictor=2)
        else:
            self.dst = rasterio.open(str(path), "w+", **self.profile)
        self.indexes = indexes

    def write(self, mdata, window):
        dst_data = self.dst.read(window=window, indexes=self.indexes, masked=True)
        dst_data[~mdata.mask] = mdata[~mdata.mask]
        self.dst.write(dst_data, window=window, indexes=self.indexes)

    def close(self):
        self.dst.close()


class Window_writer_rasterio_via_numpy:
    # Single channel numpy-based writer (not faster)
    def __init__(self, path, profile, dtype=None):
        self.path = path
        self.profile = set_sparse_rasterio_profile(profile, dtype)
        dst_np = np.zeros([self.profile["height"], self.profile["width"]], np.float32)
        self.dst_mnp = np.ma.array(data=dst_np, mask=np.ones_like(dst_np))

    def write(self, mdata, window):
        S = window.toslices()
        self.dst_mnp[S][~mdata.mask] = mdata[~mdata.mask]

    def close(self):
        self.dst = rasterio.open(str(self.path), "w+", **self.profile)
        self.dst.write(self.dst_mnp, indexes=1)
        self.dst.close()


def gpd_affine_transform(gf, shapely_matrix):
    return gf.assign(geometry=gf.affine_transform(shapely_matrix))


def vecfeatures_to_ssegm_mask(
    gf_cfeats: gpd.GeoDataFrame,  # geo-df with a 'label' field
    label_names: List[str],
    window_size: Tuple[int, int],
) -> np.ndarray:  # uint8 ndarray
    # Restrict labels
    rfeats = gf_cfeats[gf_cfeats.label.isin(label_names)]
    if "ignore" in label_names:
        bg_class_id = len(label_names) - 1
    else:
        bg_class_id = len(label_names)
    if len(rfeats) == 0:
        ssegm_mask = np.full(window_size, bg_class_id, dtype=np.uint8)
    else:
        mapping, i = {}, 0
        for label in label_names:
            if label == "ignore":
                mapping[label] = 255
            else:
                mapping[label] = i
                i += 1
        label_ids = rfeats["label"].map(mapping)
        ssegm_mask = rasterio.features.rasterize(
            shapes=list(zip(rfeats["geometry"], label_ids)),
            out_shape=window_size,
            fill=bg_class_id,
            dtype=np.uint8,
            all_touched=False,
        )
    return ssegm_mask


def produce_colortable_n(n_colors):
    cmap = mpl.cm.jet
    cm_subsection = np.linspace(0, 1, n_colors)
    colors = [cmap(x) for x in cm_subsection]
    colors255 = (np.array(colors)[:, [2, 1, 0]] * 255).astype(np.uint8)
    colortable = np.ones((256, 1, 3), dtype=np.uint8) * 30
    colortable[len(colors255)] = 255
    colortable[0 : len(colors255)] = colors255[:, None, :]
    return colortable


def get_data_window_efficient(src):
    # Code copiedi rom rasterio/windows.py:get_data_window
    arr_mask = src.read_masks(1) == 255
    v = []
    for nz in arr_mask.nonzero():
        if nz.size:
            v.append((nz.min(), nz.max() + 1))
        else:
            v.append((0, 0))
    window = Window.from_slices(*v)
    return window


def read_tif_downscaled(inpath, scale=4, crop=True):
    with rasterio.open(inpath) as src:
        if crop:
            window = get_data_window_efficient(src)
            H, W = window.height, window.width
        else:
            window = None
            H, W = np.array(src.shape)
        shape_dscale = (src.count, H // scale, W // scale)
        img_dscale = src.read(
            window=window,
            out_shape=shape_dscale,
            resampling=rasterio.enums.Resampling.nearest,
            masked=True,
        )
    return img_dscale


def export_targets_tif_to_jpg(inpath, outpath, n_colors, scale=4, crop=True):
    # Export raster to .jpg, at 1/scale resolution
    img_dscale = read_tif_downscaled(inpath, scale=scale, crop=crop)
    colortable = produce_colortable_n(n_colors)
    img_cv = cv2.applyColorMap(img_dscale, colortable)
    cv2.imwrite(outpath, img_cv)


def prepare_raster_targets(
    gf_feats: gpd.GeoDataFrame,
    gf_aois: gpd.GeoDataFrame,
    label_names: List[str],
    fold_evalref: Path,
    jpg_export_scale: int = 4,
) -> Dict[str, Dict]:
    """
    Prepare raster targets to help evaluation
    - Extract to disk, return paths and profiles
    """
    raster_targets = {}
    for aoi_name, row_aoi in gf_aois.iterrows():
        poly_aoi = row_aoi["geometry"]
        with rasterio.open(str(row_aoi["rasterpath"])) as src:
            raster_profile = src.profile
        raster_path = mkdir(fold_evalref) / f"{aoi_name}_targets.tif"
        raster_targets[aoi_name] = {
            "profile": raster_profile,
            "path": raster_path,
        }
        wwr = Window_writer_rasterio(raster_path, raster_profile, "uint8")
        gf_rasterfeats = gf_feats[gf_feats.intersects(poly_aoi)]
        safft_world_to_raster = (~raster_profile["transform"]).to_shapely()
        gf_rasterfeats_t = gpd_affine_transform(gf_rasterfeats, safft_world_to_raster)
        raster_shape = (raster_profile["height"], raster_profile["width"])
        data_raster = vecfeatures_to_ssegm_mask(
            gf_rasterfeats_t, label_names, raster_shape
        )
        poly_aoi_raster = affine_transform(poly_aoi, safft_world_to_raster)
        mask_raster = ~rasterio.features.rasterize(
            shapes=[(poly_aoi_raster, 1)],
            out_shape=raster_shape,
            fill=0,
            dtype=np.uint8,
        ).astype(bool)
        # Add good AOI mask here
        wwr.dst.write(np.ma.array(data=data_raster, mask=mask_raster), indexes=1)
        wwr.close()

        out_path = str(Path(raster_path).with_suffix(".jpg"))
        export_targets_tif_to_jpg(
            raster_path, out_path, len(label_names), jpg_export_scale
        )
    return raster_targets


def gpkg_save(gf, fold, name):
    if gf.empty:
        return
    gf = gf.copy()
    if len(gf):
        gf.index.name = "fid"  # QGIS compatibility
    Path(fold).mkdir(exist_ok=True, parents=True)
    gf.to_file(f"{fold}/{name}.gpkg", layer=name, driver="GPKG", index=True)


def find_area_overlaps(gf_features, gf_areas):
    """Associate features to areas they intersect.

    Note: Spatial join MUST be done with an "intersects" predicate
    Note: Each feature gets CSV string to allow saving to saving to .gpkg
    """
    areas = gf_features.sjoin(gf_areas, how="left", predicate="intersects")[
        "index_right"
    ].fillna("no_area")
    areas = areas.groupby(level=0).apply(lambda x: ",".join(sorted(x)))
    areas[areas == "no_area"] = ""
    assert areas.index.equals(gf_features.index)
    return areas
