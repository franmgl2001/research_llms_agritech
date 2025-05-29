import math
import requests
import numpy as np
import cv2
from io import BytesIO
from PIL import Image
import json
from shapely.geometry import Polygon, Point, mapping
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from shapely.ops import unary_union
import torch
import time

def deg2num(lat_deg, lon_deg, zoom):
    """Convert latitude/longitude to XYZ tile indices."""
    lat_rad = math.radians(lat_deg)
    n = 2.0 ** zoom
    xtile = int((lon_deg + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.log(math.tan(lat_rad) + 1.0/math.cos(lat_rad)) / math.pi) / 2.0 * n)
    return xtile, ytile

def num2deg(x, y, zoom):
    """Inverse of deg2num: returns (lat, lon) for fractional tile coords."""
    n = 2.0 ** zoom
    lon = x / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2*y/n)))
    lat = math.degrees(lat_rad)
    return lat, lon

def fetch_tile_xy(tile_x, tile_y, zoom):
    """Fetch Esri World_Imagery tile by tile indices."""
    url = f"https://server.arcgisonline.com/ArcGIS/rest/services/" \
          f"World_Imagery/MapServer/tile/{zoom}/{tile_y}/{tile_x}"
    resp = requests.get(url)
    resp.raise_for_status()
    return Image.open(BytesIO(resp.content))

def latlon_to_pixel(lat, lon, zoom, tile_x, tile_y):
    """Map a lat/lon back into pixel coords within the 256×256 tile."""
    lat_rad = math.radians(lat)
    n = 2.0**zoom
    fx = (lon + 180.0)/360.0 * n
    fy = (1.0 - math.log(math.tan(lat_rad) + 1.0/math.cos(lat_rad)) / math.pi)/2.0 * n
    px = (fx - tile_x) * 256
    py = (fy - tile_y) * 256
    return px, py

def masks_to_polygons(masks, zoom, tile_x, tile_y, tolerance=2e-4):
    """Convert SAM masks to geo-polygons and simplify them."""
    polys = []
    for m in masks:
        mask = (m["segmentation"].astype(np.uint8) * 255)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if len(cnt) < 3:
                continue
            pts = cnt[:,0,:]
            geo_pts = []
            for x_px, y_px in pts:
                xt = tile_x + x_px/256.0
                yt = tile_y + y_px/256.0
                lat, lon = num2deg(xt, yt, zoom)
                geo_pts.append((lon, lat))
            raw = Polygon(geo_pts)
            # sometimes many tiny rings overlap—merge them first if you like:
            # raw = unary_union(raw)
            # now simplify:
            simp = raw.simplify(tolerance, preserve_topology=True)
            if simp.is_valid and simp.area > 0:
                polys.append(simp)
    return polys

def find_closest_polygon(polygons, point_geom):
    """Return the polygon that contains the point, or else the one with min distance.
    Only considers polygons with area >= 0.5 hectares (5000 square meters)."""
    # Filter polygons by minimum area (0.5 ha = 5000 m²)
    valid_polygons = [p for p in polygons if p.area * 111319.9 * 111319.9 * math.cos(math.radians(point_geom.y)) >= 5000]
    
    if not valid_polygons:
        return None
        
    # First check if any valid polygon contains the point
    for p in valid_polygons:
        if p.contains(point_geom):
            return p
            
    # If no containing polygon, find closest
    dists = [(p.distance(point_geom), p) for p in valid_polygons]
    return min(dists, key=lambda x: x[0])[1]

def process_tile(tile_x, tile_y, coords, mask_generator, zoom, tolerance=1e-4):
    """Segment a tile and associate the closest, simplified mask polygon to each point."""
    start_time = time.time()
    tile_img = fetch_tile_xy(tile_x, tile_y, zoom)
    print(f"Time taken to fetch tile: {time.time() - start_time} seconds")
    tile_np = np.array(tile_img)
    masks = mask_generator.generate(tile_np)
    print(f"Time taken to generate masks: {time.time() - start_time} seconds")
    # pass tolerance here:
    polys = masks_to_polygons(masks, zoom, tile_x, tile_y, tolerance=tolerance)
    print(f"Time taken to generate polygons: {time.time() - start_time} seconds")
    features = []
    for lat, lon in coords:
        pt = Point(lon, lat)
        closest = find_closest_polygon(polys, pt)
        if closest:
            feature = {
                "type": "Feature",
                "geometry": mapping(closest),
                "properties": {
                    "source": "SAM_on_Esri_World_Imagery",
                    "zoom": zoom,
                    "query_point": {"lat": lat, "lon": lon}
                }
            }
            features.append(feature)
    print(f"Time taken to process tile: {time.time() - start_time} seconds")
    return features


def segment_points(coords, zoom=18, sam_checkpoint="sam_vit_h_4b8939.pth"):
    """
    Segment land-cover polygons for a list of (lat, lon) points.
    Returns a GeoJSON FeatureCollection.
    """
    start_time = time.time()
    # Initialize SAM model and mask generator once
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint).to(device)
    print(f"Time taken to initialize SAM model: {time.time() - start_time} seconds")
    mask_generator = SamAutomaticMaskGenerator(
    sam,
    points_per_side=32,         # default is 64–128; lower = fewer proposals
    pred_iou_thresh=0.90,       # discard low-quality masks early
    stability_score_thresh=0.90,
)
    print(f"Time taken to initialize mask generator: {time.time() - start_time} seconds")
    # Group coordinates by tile indices
    tiles = {}
    for lat, lon in coords:
        tx, ty = deg2num(lat, lon, zoom)
        tiles.setdefault((tx, ty), []).append((lat, lon))
    print(f"Time taken to group coordinates by tile indices: {time.time() - start_time} seconds")
    all_features = []
    for (tx, ty), pts in tiles.items():
        feats = process_tile(tx, ty, pts, mask_generator, zoom)
        all_features.extend(feats)
    print(f"Time taken to process all tiles: {time.time() - start_time} seconds")

    return {
        "type": "FeatureCollection",
        "features": all_features
    }

if __name__ == "__main__":
    # Example list of coordinates
    coordinates = [
    (20.270502, -101.025952),
    (20.251555, -101.020969),
    (20.445666, -101.053608),
    (20.454675, -101.048234),
    (20.452664, -101.047245),
    (20.47632,  -101.07839),
    (20.463271, -101.096651),
    (20.46327,  -101.09665),
    (20.46831,  -101.09627),
    (20.48029,  -101.0757),
    (20.47973,  -101.08109),
    (20.48146,  -101.09437),
    (20.484,    -101.09733),
    (20.47987,  -101.08125),
    (20.49514,  -101.06171),
    (20.48833,  -101.05247),
    (20.46346,  -101.08335),
    (20.46313,  -101.09155),
    (20.50549,  -101.04493),
    (20.51049,  -101.02136),
]
    geojson = segment_points(coordinates, zoom=16)
    with open('field_polygons.json', 'w') as f:
        json.dump(geojson, f, indent=2)
    print(json.dumps(geojson, indent=2))