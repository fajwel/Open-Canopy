import logging
import math
import warnings
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, TypedDict, Union

import affine
import cv2
import geopandas as gpd
import numpy as np
import rasterio
import rasterio.features
import shapely
import skimage
import torch
from PIL import Image
from pyproj import CRS
from shapely.affinity import affine_transform
from shapely.geometry import LinearRing, MultiPolygon, Polygon
from shapely.ops import unary_union
from torch.utils.data.dataloader import default_collate
from triangle import triangulate

from src.utils.data import gpd_affine_transform, vecfeatures_to_ssegm_mask

log = logging.getLogger(__name__)


Affine_transform_shapely = Tuple[int, int, int, int, int, int]


def afft_shapely_to_ocv(safft: Affine_transform_shapely) -> np.ndarray:
    (a, b, d, e, x_off, y_off) = safft
    return np.array([[a, b, x_off], [d, e, y_off]])


def afft_shapely_to_affine(
    safft: Affine_transform_shapely,
) -> affine.Affine:
    (a, b, d, e, x_off, y_off) = safft
    return affine.Affine(a, b, x_off, d, e, y_off)


def afft_ocv_to_shapely(wmat: np.ndarray) -> Affine_transform_shapely:
    ((a, b, x_off), (d, e, y_off)) = wmat
    return (a, b, d, e, x_off, y_off)


class AoiSampler:
    """Copied from rastervision at pytorch_learner/dataset/utils/aoi_sampler.py.

    Given a set of polygons representing the AOI, allows efficiently sampling points inside the AOI
    uniformly at random.

    To achieve this, each polygon is first partitioned into triangles (triangulation). Then, to
    sample a single point, we first sample a triangle at random with probability proportional to
    its area and then sample a point within that triangle uniformly at random.
    """

    def __init__(self, polygons: Sequence[Polygon]) -> None:
        # merge overlapping polygons, if any
        merged_polygons = unary_union(polygons)
        if isinstance(merged_polygons, Polygon):
            merged_polygons = [merged_polygons]
        self.polygons = MultiPolygon(merged_polygons)
        self.triangulate(self.polygons)

    def triangulate(self, polygons):
        triangulations = [self.triangulate_polygon(p) for p in polygons.geoms]
        self.triangulations = triangulations
        self.origins = np.vstack([t["origins"] for t in triangulations])
        self.vec_AB = np.vstack([t["bases"][0] for t in triangulations])
        self.vec_AC = np.vstack([t["bases"][1] for t in triangulations])
        areas = np.concatenate([t["areas"] for t in triangulations])
        self.weights = areas / areas.sum()
        self.ntriangles = len(self.origins)

    def sample(self, n: int = 1) -> np.ndarray:
        """
        Sample a random point within the AOI, using the following algorithm:
            - Randomly sample one triangle (ABC) with probability proportional
            to its area.
            - Starting at A, travel a random distance along vectors AB and AC.
            - Return the final position.

        Args:
            n (int, optional): Number of points to sample. Defaults to 1.

        Returns:
            np.ndarray: (n, 2) 2D coordinates of the sampled points.
        """
        tri_idx = np.random.choice(self.ntriangles, p=self.weights, size=n)
        origin = self.origins[tri_idx]
        vec_AB = self.vec_AB[tri_idx]
        vec_AC = self.vec_AC[tri_idx]
        # the fractions to travel along each of the two vectors
        r, s = np.random.uniform(size=(2, n, 1))
        # If the fractions will land us in the wrong half of the parallelogram
        # defined by vec AB and vec AC, reflect them into the correct half.
        mask = (r + s) > 1
        r[mask] = 1 - r[mask]
        s[mask] = 1 - s[mask]
        loc = origin + (r * vec_AB + s * vec_AC)
        return loc

    def triangulate_polygon(self, polygon: Polygon) -> dict:
        """Extract vertices and edges from the polygon (and its holes, if any) and pass them to the
        Triangle library for triangulation."""
        vertices, edges = self.polygon_to_graph(polygon.exterior)

        # We ignore hole as empty patch will be filter later
        holes = None  # polygon.interiors
        if not holes:
            args = {
                "vertices": vertices,
                "segments": edges,
            }
        else:
            for hole in holes:
                hole_vertices, hole_edges = self.polygon_to_graph(hole)
                # make the indices point to entries in the global vertex list
                hole_edges += len(vertices)
                # append hole vertices to the global vertex list
                vertices = np.vstack([vertices, hole_vertices])
                edges = np.vstack([edges, hole_edges])

            # the triangulation algorithm requires a sample point inside each
            # hole
            hole_centroids = np.stack([hole.centroid for hole in holes])

            args = {
                "vertices": vertices,
                "segments": edges,
                "holes": hole_centroids,
            }

        tri = triangulate(args, opts="p")
        simplices = tri["triangles"]
        vertices = np.array(tri["vertices"])
        origins, bases = self.triangle_origin_and_basis(vertices, simplices)

        out = {
            "vertices": vertices,
            "simplices": simplices,
            "origins": origins,
            "bases": bases,
            "areas": self.triangle_area(vertices, simplices),
        }
        return out

    def polygon_to_graph(
        self, polygon: LinearRing
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Given the exterior of a polygon, return its graph representation.

        Args:
            polygon (LinearRing): The exterior of a polygon.

        Returns:
            Tuple[np.ndarray, np.ndarray]: An (N, 2) array of vertices and
            an (N, 2) array of indices to vertices representing edges.
        """
        vertices = np.array(polygon.coords)
        # Discard the last vertex - it is a duplicate of the first vertex and
        # duplicates cause problems for the Triangle library.
        vertices = vertices[:-1]

        N = len(vertices)
        # Tuples of indices to vertices representing edges.
        # mod N ensures edge from last vertex to first vertex by making the
        # last tuple [N-1, 0].
        edges = np.column_stack([np.arange(0, N), np.arange(1, N + 1)]) % N

        return vertices, edges

    def triangle_side_lengths(
        self, vertices: np.ndarray, simplices: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate lengths of all 3 sides of each triangle specified by the simplices array.

        Args:
            vertices (np.ndarray): (N, 2) array of vertex coords in 2D.
            simplices (np.ndarray): (N, 3) array of indexes to entries in the
                vertices array. Each row represents one triangle.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: ||AB||, ||BC||, ||AC||
        """
        A = vertices[simplices[:, 0]]
        B = vertices[simplices[:, 1]]
        C = vertices[simplices[:, 2]]
        AB, AC, BC = B - A, C - A, C - B
        ab = np.linalg.norm(AB, axis=1)
        bc = np.linalg.norm(BC, axis=1)
        ac = np.linalg.norm(AC, axis=1)
        return ab, bc, ac

    def triangle_origin_and_basis(
        self, vertices: np.ndarray, simplices: np.ndarray
    ) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """For each triangle ABC, return point A, vector AB, and vector AC.

        Args:
            vertices (np.ndarray): (N, 2) array of vertex coords in 2D.
            simplices (np.ndarray): (N, 3) array of indexes to entries in the
                vertices array. Each row represents one triangle.

        Returns:
            Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]: 3 arrays of shape
                (N, 2), organized into tuples like so:
                (point A, (vector AB, vector AC)).
        """
        A = vertices[simplices[:, 0]]
        B = vertices[simplices[:, 1]]
        C = vertices[simplices[:, 2]]
        AB = B - A
        AC = C - A
        return A, (AB, AC)

    def triangle_area(
        self, vertices: np.ndarray, simplices: np.ndarray
    ) -> np.ndarray:
        """Calculate area of each triangle specified by the simplices array using Heron's formula.

        Args:
            vertices (np.ndarray): (N, 2) array of vertex coords in 2D.
            simplices (np.ndarray): (N, 3) array of indexes to entries in the
                vertices array. Each row represents one triangle.

        Returns:
            np.ndarray: (N,) array of areas
        """
        a, b, c = self.triangle_side_lengths(vertices, simplices)
        p = (a + b + c) * 0.5
        area = p * (p - a) * (p - b) * (p - c)
        area[area < 0] = 0
        area = np.sqrt(area)
        return area


def sample_grid_squares_from_aoi_v2(
    poly: Polygon,
    size: int,
    crs: CRS,
    stride: Optional[int] = None,
    shift: Optional[Tuple[int, int]] = (0, 0),
    raster_afft: Optional[affine.Affine] = None,
) -> gpd.GeoDataFrame:
    # Create squares around polygon, with 2-square buffer
    (l, d, r, t) = poly.bounds
    if stride is None:
        stride = size
    w_across = np.ceil((r - l) / stride)
    h_across = np.ceil((t - d) / stride)
    x_counts = np.arange(-2, w_across + 3)
    y_counts = -np.arange(-2, h_across + 3)
    poly_origin = np.array([l, t])
    # Make sure relative pixel offsets are integers
    if raster_afft is not None:
        shift = np.array(shift)
        r_origin = (raster_afft.xoff, raster_afft.yoff)
        r_pixel_size = (raster_afft.a, raster_afft.e)
        r_offpixels = (poly_origin - r_origin) / r_pixel_size
        poly_origin = r_offpixels.round() * r_pixel_size + r_origin + shift
    # Make coordinates
    x_centres = x_counts * stride + poly_origin[0]
    y_centres = y_counts * stride + poly_origin[1]
    xv, yv = np.meshgrid(x_centres, y_centres)
    half_left = size // 2 - 1
    half_right = size - half_left
    gf_squares = gpd.GeoDataFrame(
        geometry=[
            shapely.geometry.box(
                x - half_left, y - half_left, x + half_right, y + half_right
            )
            for x, y in zip(xv.flat, yv.flat)
        ],
        crs=crs,
    )
    gf_poly = gpd.GeoDataFrame({"geometry": {0: poly}}, crs=crs)
    gf_squares = (
        gf_squares.sjoin(gf_poly, how="inner", predicate="intersects")
        .drop(columns=["index_right"])
        .reset_index(drop=True)
    )
    if not gf_squares.dissolve().loc[0, "geometry"].contains(poly):
        warnings.warn("Union of squares does not contain the target polygon")
    # gpkg_save(gf_squares, '.', 'squares')
    return gf_squares


def create_squares_from_points(rgen, points_np, min_size, max_size):
    sizes = rgen.integers(min_size, max_size, len(points_np))
    lhalfs = sizes // 2 - 1
    rhalfs = sizes - lhalfs
    squares = [
        shapely.geometry.box(x - lh, y - lh, x + rh, y + rh)
        for (x, y), lh, rh in zip(points_np, lhalfs, rhalfs)
    ]
    return squares


def rotate_shapes(rgen, shapes, angle, angle_prob):
    angles = rgen.integers(-angle, angle, len(shapes), endpoint=True)
    if angle_prob < 1.0:
        # Some shapes should NOT be rotated
        do_not_rotate = rgen.binomial(
            1, 1 - angle_prob, size=len(angles)
        ).astype(bool)
        angles[do_not_rotate] = 0
    rot_shapes = [
        shapely.affinity.rotate(shape, angle)
        for shape, angle in zip(shapes, angles)
    ]
    return rot_shapes


def sample_squares_inside_aoi_leqn(
    n, rgen, aoi_polygons, min_size, max_size, angle, angle_prob
):
    """Generation starts from centerpoints, some squares will not fit inside aoi and will have to
    be filtered."""
    # Create squares that probably fit
    aoi_sampler = AoiSampler(aoi_polygons)
    points_np = aoi_sampler.sample(n)
    squares = create_squares_from_points(rgen, points_np, min_size, max_size)
    if angle is not None:
        squares = rotate_shapes(rgen, squares, angle, angle_prob)
    # Leave only squares that definitely fit
    squares = [
        box for box in squares if any(box.within(aoi) for aoi in aoi_polygons)
    ]
    return squares


def sample_squares_inside_aoi(
    n, rgen, aoi_polygons, min_size, max_size, angle, angle_prob, n_batch=256
):
    """Make sure to sample exactly n squares."""
    squares = []
    while len(squares) < n:
        squares.extend(
            sample_squares_inside_aoi_leqn(
                n_batch,
                rgen,
                aoi_polygons,
                min_size,
                max_size,
                angle,
                angle_prob,
            )
        )
    return squares[:n]


def compute_tcell_safft(
    SQR_icell: np.ndarray, WH: Optional[Tuple[int, int]] = None
) -> Tuple[Affine_transform_shapely, int, int]:
    """
    Takes bbox points, starting from lower right, in a ccw order. Produces
    transform that would place them like this:

     -------------> X coord
    |  2 (0, 0)       1 (width, 0)
    |
    |
    |
    V  3 (0, height)

    Y coord (Note: is inverted!)

    Args:
        SQR_icell: coordinates of bbox corners, numpy(N, 2)
        WH: if None, keep scale to nearest int
            if tuple -> set (width, height) from it

    Returns:
        Tuple: [
            shapely affine transform (a, b, d, e, x_off, y_off),\
            width, height]
    """
    vec_21 = SQR_icell[1] - SQR_icell[2]
    vec_32 = SQR_icell[3] - SQR_icell[2]
    # Compute rotation angle
    angle_at2 = np.arctan2(-vec_21[1], vec_21[0])  # Y-coord is inverted, so -
    cosp = math.cos(angle_at2)
    sinp = math.sin(angle_at2)
    # Compute scaling factor
    vec_width = np.linalg.norm(vec_21)
    vec_height = np.linalg.norm(vec_32)
    if WH is None:
        width, height = np.round(vec_width), np.round(vec_height)
    else:
        width, height = WH
    mW, mH = width / vec_width, height / vec_height
    # Everything except translation
    a, b, d, e = cosp * mW, -sinp * mH, sinp * mW, cosp * mH
    # Compute translation (the point at 2 should be at [0, 0])
    x2, y2 = SQR_icell[2]
    x_off = -(a * x2 + b * y2)
    y_off = -(d * x2 + e * y2)
    safft = (a, b, d, e, x_off, y_off)
    return safft, width, height


def compute_tcell_affine_transform_ocv(SQR_icell, WH=None):
    """Same thing as compute_tcell_safft, but via OCV."""
    if WH is None:
        vec_width = np.linalg.norm(SQR_icell[1] - SQR_icell[2])
        vec_height = np.linalg.norm(SQR_icell[3] - SQR_icell[2])
        width, height = vec_width, vec_height
    else:
        width, height = WH
    wmat = cv2.getAffineTransform(
        SQR_icell[1:4].astype(np.float32),
        np.array([[width, 0], [0, 0], [0, height]]).astype(np.float32),
    )
    return wmat, width, height


def sample_icell_from_raster(
    rasterpath: str,
    square: shapely.geometry.Polygon,
    gf_feats: gpd.GeoDataFrame,  # geodf with features
):
    """Sample ICELL (image-coordinate aligned cell, centered around square)"""
    # Slightly bigger square to prevent border artifacts
    big_square = shapely.affinity.scale(square, xfact=1.1, yfact=1.1)
    # We sample with filled=False to get "masked" images. This allows us to
    # later easily set unfilled values to reasonable values (good for DTMs,
    # prevents some crazy values from slipping through), also looks prettier
    # when displayed
    with rasterio.open(rasterpath) as src:
        img_icell, transform_icell = rasterio.mask.mask(
            src, shapes=[big_square], crop=True, indexes=1, filled=False
        )
        img_icell = img_icell.astype("float32")
        safft_e32_to_icell = (~transform_icell).to_shapely()
    # Only features touching square (and cut by big_square)
    gf_cfeats = gf_feats[gf_feats.intersects(square)]
    # Cut big rivers
    gf_cfeats = gpd.clip(gf_cfeats, big_square)
    # Affine transform both the features and the square
    gf_cfeats_icell = gpd_affine_transform(gf_cfeats, safft_e32_to_icell)
    square_icell = affine_transform(square, safft_e32_to_icell)
    return gf_cfeats_icell, square_icell, img_icell, safft_e32_to_icell


def rasterio_urcrop(rasterpath, shapes):
    # Unrolled rasterio routine that samples MaskedArray window and returns it
    with rasterio.open(rasterpath) as src:
        # raster_geometry_mask
        window = rasterio.features.geometry_window(
            src, shapes, pad_x=0, pad_y=0
        )
        transform = src.window_transform(window)
        out_shape = (int(window.height), int(window.width))
        shape_mask = rasterio.features.geometry_mask(
            shapes,
            transform=transform,
            invert=False,
            out_shape=out_shape,
            all_touched=False,
        )
        # mask
        if src.count == 1:
            out_image = src.read(
                window=window, out_shape=out_shape, masked=True, indexes=1
            )
        else:
            out_image = src.read(
                window=window,
                out_shape=out_shape,
                masked=True,
                indexes=range(1, src.count + 1),
            )
        out_image.mask = out_image.mask | shape_mask
        if out_image.ndim == 3:
            # convert from CHW to HWC
            out_image = out_image.transpose(1, 2, 0)
    return out_image, transform, window


class Interpolation(Enum):
    NEAREST = 0
    LINEAR = 1
    CUBIC = 3


def get_cv2_interpolation(iinter: Interpolation):
    iinter = Interpolation(iinter)
    return {
        Interpolation.NEAREST: cv2.INTER_NEAREST,
        Interpolation.LINEAR: cv2.INTER_LINEAR,
        Interpolation.CUBIC: cv2.INTER_CUBIC,
    }[iinter]


def warpaffine_masked(
    img: np.ma.MaskedArray | np.ndarray,
    wmat: np.ndarray,
    tWH: Tuple[int, int],
    iinter: Interpolation,
    mean_value: Optional[int | float] = None,
    inverse_map: bool = False,
) -> np.ma.MaskedArray:
    """Warp masked array safely by transforming data and mask separately.

    TODO: Can be optimized further by abstaining from double warping and
        passing mean value explicitly
    """
    mask_flags = cv2.INTER_NEAREST
    if inverse_map:
        mask_flags |= cv2.WARP_INVERSE_MAP
    wimg = warpaffine_meanfill(img, wmat, tWH, iinter, mean_value, inverse_map)
    # Warp mask separately with NN interpolation, apply
    if isinstance(img, np.ma.MaskedArray):
        smask: np.ndarray[Any, np.dtype[np.bool_]] = img.mask
    else:
        smask = np.zeros_like(img, dtype=np.bool_)
    wmask = cv2.warpAffine(
        smask.astype(int),
        wmat,
        tWH,
        flags=mask_flags,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=1,
    ).astype(bool)
    return np.ma.array(data=wimg, mask=wmask)


def warpaffine_meanfill(
    img: np.ma.MaskedArray | np.ndarray,
    wmat: np.ndarray,
    tWH: Tuple[int, int],
    iinter: Interpolation,
    mean_value: Optional[int | float] = None,
    inverse_map: bool = False,
) -> np.ndarray:
    flags = get_cv2_interpolation(iinter)
    if inverse_map:
        flags |= cv2.WARP_INVERSE_MAP
    if mean_value is None:
        if img.dtype == np.uint8:
            mean_value = int(np.round(img.mean()))
        elif img.dtype == np.float32:
            mean_value = float(img.mean())
        else:
            raise ValueError(f"{img.dtype=} must be uint8 or float32")
    if isinstance(img, np.ma.MaskedArray):
        img = img.filled(mean_value)
    wimg = cv2.warpAffine(
        img,
        wmat,
        tWH,
        flags=flags,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=mean_value,
    )
    return wimg


class Icell(TypedDict):
    img: np.ma.MaskedArray[Any, np.dtype[np.float32]]
    square: shapely.geometry.Polygon
    safft_world_to_icell: Affine_transform_shapely
    window: rasterio.windows.Window


def sample_icell_from_raster_v2(
    rasterpath: str,
    square: shapely.geometry.Polygon,
) -> Icell:
    """Sample ICELL (image-coordinate aligned cell, centered around square)"""
    # / Cut data from rasterio, get transform/window
    # Slightly bigger square to prevent border artifacts
    big_square = shapely.affinity.scale(square, xfact=1.1, yfact=1.1)
    img_icell, afft_icell_to_world, window_icell = rasterio_urcrop(
        rasterpath, [big_square]
    )
    img_icell = img_icell.astype("float32")  # Make sure always float
    # / Transform annotation features
    # Only features touching square

    # Affine transform features and square
    safft_world_to_icell = (~afft_icell_to_world).to_shapely()
    square_icell = affine_transform(square, safft_world_to_icell)
    return Icell(
        img=img_icell,
        square=square_icell,
        safft_world_to_icell=safft_world_to_icell,
        window=window_icell,
    )


class VIcell(TypedDict):
    gf_cfeats: gpd.GeoDataFrame
    square: shapely.geometry.Polygon
    safft_world_to_vcell: Affine_transform_shapely


def sample_vicell_from_vector(
    gf_feats: gpd.GeoDataFrame,  # geodf with features
    square: shapely.geometry.Polygon,
    safft_world_to_vcell: Affine_transform_shapely,
) -> VIcell:
    """Sample VCELL (Vector image-coordinate aligned cell, centered around square)"""
    # Slightly bigger square to prevent border artifacts
    big_square = shapely.affinity.scale(square, xfact=1.1, yfact=1.1)

    # / Transform annotation features
    # Only features touching square
    potential_gf_cfeats_index = gf_feats.sindex.query(square)
    potential_gf_cfeats = gf_feats.iloc[potential_gf_cfeats_index]
    gf_cfeats = potential_gf_cfeats[potential_gf_cfeats.intersects(square)]
    # Cut big rivers (important to re-sort afterwards)!
    gf_cfeats = gpd.clip(gf_cfeats, big_square).sort_index()
    # Affine transform features and square
    gf_cfeats_vcell = gpd_affine_transform(gf_cfeats, safft_world_to_vcell)
    square_vcell = affine_transform(square, safft_world_to_vcell)
    return VIcell(
        gf_cfeats=gf_cfeats_vcell,
        square=square_vcell,
        safft_world_to_vcell=safft_world_to_vcell,
    )


class Tcell(TypedDict):
    img: np.ma.MaskedArray[Any, np.dtype[np.float32]]
    square: shapely.geometry.Polygon
    safft_icell_to_tcell: Affine_transform_shapely


def convert_icell_to_tcell_v2(
    img_icell: np.ma.MaskedArray[Any, np.dtype[np.float32]],
    square_icell: shapely.geometry.Polygon,
    WH: Optional[Tuple[int, int]],
    iinter: Interpolation,
) -> Tcell:
    # Compute proper transform matrix from the square
    SQR_icell = np.array(square_icell.exterior.coords)
    safft_icell_to_tcell, tcell_height, tcell_width = compute_tcell_safft(
        SQR_icell, WH
    )
    # Affine transform the image
    tWH = (tcell_width, tcell_height)
    wmat_icell_to_tcell = afft_shapely_to_ocv(safft_icell_to_tcell)
    img_tcell = warpaffine_masked(img_icell, wmat_icell_to_tcell, tWH, iinter)
    # Affine transform features and square
    square_tcell = affine_transform(square_icell, safft_icell_to_tcell)
    return Tcell(
        img=img_tcell,
        square=square_tcell,
        safft_icell_to_tcell=safft_icell_to_tcell,
    )


class VTcell(TypedDict):
    gf_cfeats: gpd.GeoDataFrame
    square: shapely.geometry.Polygon
    safft_icell_to_tcell: Affine_transform_shapely


def convert_vicell_to_vtcell(
    gf_cfeats_vicell: gpd.GeoDataFrame,
    square_icell: shapely.geometry.Polygon,
    safft_icell_to_tcell: Affine_transform_shapely = None,
) -> Tcell:
    if safft_icell_to_tcell is None:
        # Compute proper transform matrix from the square
        SQR_icell = np.array(square_icell.exterior.coords)
        safft_icell_to_tcell, _, _ = compute_tcell_safft(SQR_icell)
    # Affine transform features and square
    gf_cfeats_vtcell = gpd_affine_transform(
        gf_cfeats_vicell, safft_icell_to_tcell
    )
    square_vtcell = affine_transform(square_icell, safft_icell_to_tcell)
    return Tcell(
        gf_cfeats=gf_cfeats_vtcell,
        square=square_vtcell,
        safft_icell_to_tcell=safft_icell_to_tcell,
    )


def apply_affine_transform_get_img_tcell(
    img_icell: np.ma.MaskedArray,
    wmat: np.ndarray,
    tWH: np.ndarray,
    afft_lib: Literal["cv2", "pil", "skimage"],
    iinter: Interpolation,
) -> np.ndarray:
    # Deprecated
    # Fill mask with something non-offensive (otherwise weights go to NaN)
    img_icell_filled = img_icell.filled(img_icell.mean())
    if afft_lib == "cv2":
        flags = get_cv2_interpolation(iinter)
        img_tcell = cv2.warpAffine(img_icell_filled, wmat, tWH, flags=flags)
    elif afft_lib == "pil":
        resample = {
            Interpolation.NEAREST: Image.Resampling.NEAREST,
            Interpolation.LINEAR: Image.Resampling.BILINEAR,
            Interpolation.CUBIC: Image.Resampling.BICUBIC,
        }[iinter]
        inv_wmat = np.linalg.inv(np.concatenate([wmat, [[0, 0, 1]]]))
        img_tcell = np.array(
            Image.fromarray(img_icell_filled).transform(
                (tWH[0], tWH[1]),
                Image.Transform.AFFINE,
                inv_wmat[:2].flatten(),
                resample=resample,
            )
        )
    elif afft_lib == "skimage":
        inv_wmat = np.linalg.inv(np.concatenate([wmat, [[0, 0, 1]]]))
        img_tcell = skimage.transform.warp(
            img_icell_filled,
            skimage.transform.AffineTransform(matrix=inv_wmat),
            output_shape=tWH,
            order=iinter,
            preserve_range=True,
        )
    else:
        raise RuntimeError(f"Unknown {afft_lib=}")
    return img_tcell


def sample_random_squares_from_aoi(
    poly: Polygon,
    n_samples: int,
    rgen: np.random.Generator,
    size: Union[int, Tuple[int, int]],
    overlap: Union[float, Tuple[float, float]],
    angle: Optional[int],
    angle_prob: float,
    raster_afft: Optional[affine.Affine] = None,
    n_batch: int = 256,
) -> List[Polygon]:
    """Generate squares from centerpoints, keep only those that satisfy overlap% conditions with
    aoi."""
    if isinstance(size, tuple):
        min_size, max_size = size
    else:
        min_size, max_size = size, size + 1

    if isinstance(overlap, tuple):
        min_overlap, max_overlap = overlap
    else:
        min_overlap, max_overlap = overlap, 1.05

    def sample_leqn():
        # Create boxes that probably fit
        aoi_sampler = AoiSampler([poly])
        points_np = aoi_sampler.sample(n_batch)
        # Make sure relative pixel offsets are integers
        if raster_afft:
            r_origin = (raster_afft.xoff, raster_afft.yoff)
            r_pixel_size = (raster_afft.a, raster_afft.e)
            r_offpixels = (points_np - r_origin) / r_pixel_size
            points_np = r_offpixels.round() * r_pixel_size + r_origin
        squares = create_squares_from_points(
            rgen, points_np, min_size, max_size
        )
        if (angle is not None) and (angle_prob > 0):
            squares = rotate_shapes(rgen, squares, angle, angle_prob)
        # Leave only squares that overlap
        good_squares = []
        for square in squares:
            overlap = square.intersection(poly).area / square.area
            if min_overlap <= overlap <= max_overlap:
                good_squares.append(square)
        return good_squares

    # Make sure to sample exactly n squares
    squares: List[Polygon] = []
    while len(squares) < n_samples:
        squares.extend(sample_leqn())
    return squares[:n_samples]


def create_squares_from_points_and_sizes(rgen, points_np, sizes):
    lhalfs = sizes // 2 - 1
    rhalfs = sizes - lhalfs
    squares = [
        shapely.geometry.box(x - lh, y - lh, x + rh, y + rh)
        for (x, y), lh, rh in zip(points_np, lhalfs, rhalfs)
    ]
    return squares


def sample_random_squares_from_aoi_v2(
    poly: Polygon,
    n_samples: int,
    rgen: np.random.Generator,
    size_base: int,
    size_enum_sizes: List[int],
    size_enum_probs: Optional[List[float]],
    size_range_frac: float,
    size_range_sizes: List[int],
    rot_angle: Optional[int],
    rot_prob: float,
    overlap: Union[float, Tuple[float, float]],
    raster_afft: Optional[affine.Affine] = None,
    n_batch: int = 256,
) -> List[Polygon]:
    """Generate squares from centerpoints, keep only those that satisfy overlap% conditions with
    aoi."""
    if isinstance(overlap, tuple):
        min_overlap, max_overlap = overlap
    else:
        min_overlap, max_overlap = overlap, 1.05

    def sample_leqn():
        # Create boxes that probably fit
        aoi_sampler = AoiSampler([poly])
        points_np = aoi_sampler.sample(n_batch)
        # Make sure relative pixel offsets are integers
        if raster_afft:
            r_origin = (raster_afft.xoff, raster_afft.yoff)
            r_pixel_size = (raster_afft.a, raster_afft.e)
            r_offpixels = (points_np - r_origin) / r_pixel_size
            points_np = r_offpixels.round() * r_pixel_size + r_origin

        # Generating sizes
        enum_sizes = (np.array(size_enum_sizes) * size_base).astype(int)
        sizes = rgen.choice(enum_sizes, len(points_np), p=size_enum_probs)
        if size_range_frac > 0:
            l, h = (np.array(size_range_sizes) * size_base).astype(int)
            mask = rgen.random(len(sizes)) < size_range_frac
            sizes[mask] = rgen.integers(l, h, size=mask.sum())
        squares = create_squares_from_points_and_sizes(rgen, points_np, sizes)
        if (rot_angle is not None) and (rot_prob > 0):
            squares = rotate_shapes(rgen, squares, rot_angle, rot_prob)
        # Leave only squares that overlap
        good_squares = []
        for square in squares:
            overlap = square.intersection(poly).area / square.area
            if min_overlap <= overlap <= max_overlap:
                good_squares.append(square)
        return good_squares

    # Make sure to sample exactly n squares
    squares: List[Polygon] = []
    while len(squares) < n_samples:
        squares.extend(sample_leqn())
    return squares[:n_samples]


def _set_sparse_rasterio_profile(profile, dtype):
    profile = profile.copy()
    profile["sparse_ok"] = True
    profile["dtype"] = dtype
    if dtype == "float32":
        profile["nodata"] = 0
    elif dtype == "uint8":
        profile["nodata"] = 255
    else:
        raise ValueError()
    return profile


def reproject_to_tcell_warp(
    data_tcell: Union[
        np.ma.MaskedArray[Any, np.dtype[np.float32]],
        np.ndarray[Any, np.dtype[np.float32]],
    ],
    window_icell: rasterio.windows.Window,
    safft_icell_to_tcell: Affine_transform_shapely,
    iinter: Interpolation,
) -> np.ma.MaskedArray[Any, np.dtype[np.float32]]:
    """Simpler (and worse) reprojection."""
    wmat_icell_to_tcell = afft_shapely_to_ocv(safft_icell_to_tcell)
    WH_icell = (window_icell.width, window_icell.height)
    mdata_icell = warpaffine_masked(
        data_tcell, wmat_icell_to_tcell, WH_icell, iinter, inverse_map=True
    )
    return mdata_icell


def reproject_to_tcell_rasterize(
    data_tcell: Union[
        np.ma.MaskedArray[Any, np.dtype[np.float32]],
        np.ndarray[Any, np.dtype[np.float32]],
    ],
    window_icell: rasterio.windows.Window,
    safft_icell_to_tcell: Affine_transform_shapely,
    goodpoly_world: Polygon,
    safft_world_to_icell: Affine_transform_shapely,
    iinter: Interpolation,
) -> np.ma.MaskedArray[Any, np.dtype[np.float32]]:
    """Reprojection that sources final mask from goodpoly."""
    wmat_icell_to_tcell = afft_shapely_to_ocv(safft_icell_to_tcell)
    WH_icell = (window_icell.width, window_icell.height)
    data_icell = warpaffine_meanfill(
        data_tcell, wmat_icell_to_tcell, WH_icell, iinter, inverse_map=True
    )
    goodpoly_icell = affine_transform(goodpoly_world, safft_world_to_icell)
    mask_icell = ~rasterio.features.rasterize(
        shapes=[(goodpoly_icell, 1)],
        out_shape=data_icell.shape,
        fill=0,
        dtype=np.uint8,
    ).astype(bool)
    return np.ma.MaskedArray(data=data_icell, mask=mask_icell)


def from_vecfeatures_determine_targets(
    gf_cfeats: gpd.GeoDataFrame,
    label_names: List[str],
    window_size: Tuple[int, int],
) -> Dict[str, np.ndarray]:
    # Determine targets
    targets = {}
    # // Segmentation targets
    ssegm_mask = vecfeatures_to_ssegm_mask(gf_cfeats, label_names, window_size)
    targets["ssegm_mask"] = ssegm_mask.astype(np.int64)
    return targets


class TDataset_gtiff(torch.utils.data.Dataset):
    def __init__(
        self,
        gf_squares: gpd.GeoDataFrame,
        gf_feats: gpd.GeoDataFrame,
        hshades: Dict[str, Path],
        label_names: List[str],
        mean: float,
        std: float,
        iinter: Interpolation,
        WH: Union[None, Tuple[int, int]] = None,
        return_debug_info: bool = False,
    ):
        self.gf_squares = gf_squares
        self.gf_feats = gf_feats
        self.hshades = hshades
        self.label_names = label_names
        self.mean = mean
        self.std = std
        self.iinter = iinter
        self.WH = WH
        self.return_debug_info = return_debug_info

    def __len__(self):
        return len(self.gf_squares)

    def __getitem__(self, index):
        row_square = self.gf_squares.iloc[index]

        # (gf_cfeats_icell, square_icell, img_icell, safft_e32_to_icell
        icell = sample_icell_from_raster_v2(
            str(self.hshades[row_square["aoi_name"]]),
            row_square["geometry"],
            self.gf_feats,
        )
        tcell = convert_tcell_to_icell_v2(
            icell["gf_cfeats"],
            icell["square"],
            icell["img"],
            self.WH,
            self.iinter,
        )
        targets = from_vecfeatures_determine_targets(
            tcell["gf_cfeats"], self.label_names, self.WH
        )

        # Normalize image. Missing values become exactly 0
        img_nrm = (tcell["img"].filled(self.mean) - self.mean) / self.std
        I_nrm = torch.Tensor(img_nrm).unsqueeze(0)

        meta = {
            "index": index,
            "row_square": row_square,
            "icell": icell,
            "tcell": tcell,
        }
        # Remove heavy images unless debugging
        if not self.return_debug_info:
            del meta["icell"]["img"]
            del meta["tcell"]["img"]

        return I_nrm, targets, meta


def itm_collate(batch):
    if len(batch[0]) == 3:
        im_torch, target, meta = zip(*batch)
        return [default_collate(im_torch), default_collate(target), meta]
    elif len(batch[0]) == 4:
        im_torch, target, meta, context = zip(*batch)
        return [
            default_collate(im_torch),
            default_collate(target),
            meta,
            default_collate(context),
        ]
    else:
        raise ValueError("Unknown batch format")
