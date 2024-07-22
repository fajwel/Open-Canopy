"""Visualize SPOT 6-7 data associated to a 1km x 1km geometry."""

import os

import geopandas as gpd
import matplotlib.pyplot as plt
import rasterio
from rasterio.windows import from_bounds

# Load geometries
# Supposing you are at the root of Open-Canopy
opencanopy_path = "datasets"
gdf = gpd.read_file(
    os.path.join(opencanopy_path, "canopy_height", "geometries.geojson")
)
# Select geometry
ix = 0
geometry = gdf["geometry"].iloc[ix]
year = gdf["lidar_year"].iloc[ix]

# Retrieve associated spot data
with rasterio.open(
    os.path.join(opencanopy_path, "canopy_height", str(year), "spot.vrt")
) as src:
    bounds = geometry.bounds
    window = from_bounds(*bounds, transform=src.transform)
    data = src.read(window=window)


# Visualization
# On some areas images are quite dark,
# you can clip upper values for visualization purposes (not training)
def clip_image(image, max_value=0.5):
    image = image / 255
    image = image.clip(min=0, max=max_value)
    image = image / image.max()
    return image


data = clip_image(data)
plt.imshow(data.transpose(1, 2, 0))
plt.show()

# see plot_detections function in src/metrics/metrics_utils.py for more visualizations
