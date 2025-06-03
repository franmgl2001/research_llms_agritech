"""
This script is used to run the TSViT model on the PASTIS dataset and save the embeddings and masks.

The scripts reads the pickle files from the PASTIS dataset and runs the model on them.
The model is a TSViT model that is trained on the PASTIS dataset.
The model is saved in the weights/best.pth file.
The script saves the embeddings and masks in the embeddings/ and masks/ directories.

The script uses the PASTIS_segmentation_transform to transform the data into the correct format for the model.
"""

from models.TSViTdense import TSViT
import torch
from transforms.transforms import PASTIS_segmentation_transform
import pickle
import matplotlib.pyplot as plt
import numpy as np
import os
import rasterio
from rasterio.transform import from_origin
from rasterio.crs import CRS
# Label dictionary
label_dict = {
    0: "Background",
    1: "Meadow",
    2: "Soft winter wheat",
    3: "Corn",
    4: "Winter barley",
    5: "Winter rapeseed",
    6: "Spring barley",
    7: "Sunflower",
    8: "Grapevine",
    9: "Beet",
    10: "Winter triticale",
    11: "Winter durum wheat",
    12: "Fruits, vegetables, flowers",
    13: "Potatoes",
    14: "Leguminous fodder",
    15: "Soybeans",
    16: "Orchard",
    17: "Mixed cereal",
    18: "Sorghum",
    19: "Void label",
}

# Model configuration
model_config = {
    "img_res": 24,
    "patch_size": 2,
    "patch_size_time": 1,
    "patch_time": 4,
    "num_classes": 19,
    "max_seq_len": 60,
    "dim": 128,
    "temporal_depth": 4,
    "spatial_depth": 4,
    "depth": 4,
    "heads": 4,
    "pool": "cls",
    "num_channels": 11,
    "dim_head": 32,
    "dropout": 0.0,
    "emb_dropout": 0.0,
    "scale_dim": 4,
}

# Model configuration
model_config = {
    "img_res": 24,
    "patch_size": 2,
    "patch_size_time": 1,
    "patch_time": 4,
    "num_classes": 19,
    "max_seq_len": 60,
    "dim": 128,
    "temporal_depth": 4,
    "spatial_depth": 4,
    "depth": 4,
    "heads": 4,
    "pool": "cls",
    "num_channels": 11,
    "dim_head": 32,
    "dropout": 0.0,
    "emb_dropout": 0.0,
    "scale_dim": 4,
}

# Load model
model = TSViT(model_config)
model.load_state_dict(torch.load("weights/best.pth", map_location=torch.device("cpu")))
model.eval()

# Transformation pipeline
transform_pipeline = PASTIS_segmentation_transform(model_config, True)


# Process each pickle file
os.makedirs("figs", exist_ok=True)
os.makedirs("figs/geo_outputs", exist_ok=True)  # Directory for GeoTIFF outputs
for i in os.listdir("ff_pickles"):
    sample = pickle.load(open(f"ff_pickles/{i}", "rb"))

    # Adjust sample keys if necessary
    if "image" in sample:
        sample["img"] = sample["image"].astype(np.float32)
        del sample["image"]
        sample["labels"] = sample["mask"]
        del sample["mask"]

    # Extract CRS and transform
    if "crs" in sample and "transform" in sample:
        crs = sample["crs"]
        transform = sample["transform"]

    # Apply transformation pipeline
    transformed_sample = transform_pipeline(sample)
    inputs = transformed_sample["inputs"].unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        # Run the model and get predictions
        mask, embeddings = model(inputs)
        predictions = torch.argmax(mask.squeeze(0), dim=0).cpu().numpy()
        labels = transformed_sample["labels"].squeeze(0).cpu().numpy()
        # Save the embeddings
        torch.save(embeddings, f"embeddings/{i}_embeddings.pt")
    # Save predictions as a numpy array
    np.save(f"masks/{i}_predictions.npy", predictions)
    # Save predictions as GeoTIFF
    if "crs" in sample and "transform" in sample:
        output_geo_path = f"figs/geo_outputs/{i}_predictions.tif"
        with rasterio.open(
            output_geo_path,
            "w",
            driver="GTiff",
            height=predictions.shape[0],
            width=predictions.shape[1],
            count=1,
            dtype=predictions.dtype,
            crs=CRS.from_wkt(crs.to_wkt()),
            transform=transform,
        ) as dst:
            dst.write(predictions, 1)

        print(f"Processed and saved {i} with geographic reference.")
