import json

with open('field_polygons.json', 'r') as f:
    data = json.load(f)


for feature in data['features']:
    coords = feature['geometry']['coordinates'][0]
    formatted = '|'.join(f"{lat}, {lon}" for lon, lat in coords)
    print(formatted + '\n\n\n')