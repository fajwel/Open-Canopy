import functools

import numpy as np
from scipy.interpolate import griddata
from scipy.stats import binned_statistic_2d


def filter_las(las, classes_to_filter=[1, 65, 66]):
    # Remove points classified as 1, 65 or 66
    # cf. https://geoservices.ign.fr/sites/default/files/2023-10/DC_LiDAR_HD_1-0_PTS.pdf
    filtered = functools.reduce(
        lambda x, y: x & y,
        [las.classification != c for c in classes_to_filter],
    )
    points = las.points[filtered]
    return points


def get_quantile_fct(quantile):
    if quantile == 1:
        get_quantile = np.nanmax
    elif quantile == 0:
        get_quantile = np.nanmin
    else:
        get_quantile = functools.partial(np.nanquantile, q=quantile)
    return get_quantile


def get_elevation_relative_to_ground(
    las,
    width,
    height,
    method="nearest",
    quantile=0.95,
    classes_to_filter=[1, 65, 66],
    dtm=None,
):
    points = filter_las(las, classes_to_filter=classes_to_filter)

    if dtm is None:
        # Get points classified as ground (classification code 2) or water (9)
        ground_points = las.points[
            (las.classification == 2) | (las.classification == 9)
        ]
        # Interpolate ground elevation
        # (use "nearest" rather than "linear" to make it faster)

        z_ground = griddata(
            (ground_points.x, ground_points.y),
            ground_points.z,
            (points.x, points.y),
            method=method,
        )
        # take the difference
        z = points.z - z_ground
    else:
        z = points.z

    # get stat on each pixel
    get_quantile = get_quantile_fct(quantile)
    dsm, _, _, _ = binned_statistic_2d(
        points.x,
        points.y,
        z,
        statistic=get_quantile,
        bins=[width, height],
    )
    dsm = np.flipud(dsm.T)

    if dtm is not None:
        if dtm.shape[0] != dsm.shape[0] or dtm.shape[1] != dsm.shape[1]:
            raise ValueError("DTM and DSM must have the same shape")
        # Use dtm as ground elevation
        dsm = dsm - dtm

    return dsm


def interpolate_missing_data(data):
    x = np.arange(data.shape[1])
    y = np.arange(data.shape[0])
    xx, yy = np.meshgrid(x, y)
    # Points where the data is not NaN
    known_x = xx[~np.isnan(data)]
    known_y = yy[~np.isnan(data)]
    known_v = data[~np.isnan(data)]
    # Interpolate using scipy's griddata function
    interpolated_data = griddata(
        (known_x, known_y), known_v, (xx, yy), method="linear"
    )
    return np.where(np.isnan(data), interpolated_data, data)
