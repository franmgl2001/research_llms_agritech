import math
import requests
from PIL import Image
from io import BytesIO

def deg2num(lat_deg: float, lon_deg: float, zoom: int) -> (int, int):
    """
    Convert latitude/longitude to XYZ tile indices at given zoom.
    """
    lat_rad = math.radians(lat_deg)
    n = 2.0 ** zoom
    xtile = int((lon_deg + 180.0) / 360.0 * n)
    ytile = int(
        (1.0 - math.log(math.tan(lat_rad) + 1.0 / math.cos(lat_rad)) / math.pi)
        / 2.0
        * n
    )
    return xtile, ytile

def fetch_satellite_tile(lat: float, lon: float, zoom: int) -> Image.Image:
    """
    Fetches the satellite tile (raster image) centered on the tile containing (lat, lon).
    Uses Esri World Imagery, which at zoom ≳17 will show agricultural parcels clearly.
    """
    x, y = deg2num(lat, lon, zoom)
    # ArcGIS REST tiles: /MapServer/tile/{level}/{row}/{col}
    url = f"https://server.arcgisonline.com/ArcGIS/rest/services/" \
          f"World_Imagery/MapServer/tile/{zoom}/{y}/{x}"
    resp = requests.get(url)
    resp.raise_for_status()
    return Image.open(BytesIO(resp.content))

if __name__ == "__main__":
    # Example: fetch a tile over a farming region
    lat, lon = 23.09965347,-102.67018025  # e.g. near agricultural area in upstate NY
    zoom = 17                      # high enough to see individual fields
    tile_img = fetch_satellite_tile(lat, lon, zoom)
    tile_img.show()   