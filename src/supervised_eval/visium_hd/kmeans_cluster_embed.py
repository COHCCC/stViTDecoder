import sys
import os
import time
import logging
from typing import Dict, Tuple
import numpy as np
import pandas as pd
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.optim import Adam
from spatialdata import SpatialData, read_zarr, transform
from spatialdata.dataloader.datasets import ImageTilesDataset
from PIL import Image
import timm
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import shutil
from shapely.geometry import Polygon
from math import sqrt
import importlib

# -----------------------
# Step 1: Import Custom Modules
# -----------------------
if 'visiumHD' not in sys.path:
    sys.path.append('visiumHD')

# Reload custom modules
import visiumHD.select_utils as select_utils
importlib.reload(select_utils)
import transform_utils
importlib.reload(transform_utils)

from transform_utils import transform_with_cell_types

# -----------------------
# Step 2: Load and process SpatialData
# -----------------------
def load_spatial_data(path_read: str) -> SpatialData:
    """
    Load spatial data from the specified directory.

    Args:
    - path_read: Path to the root directory containing the spatial data.

    Returns:
    - SpatialData object with images, shapes, and tables.
    """
    path_write = os.path.join(path_read, "789.zarr")
    assert os.path.isdir(path_write), "Zarr directory not found"
    visium_sdata = read_zarr(path_write)
    
    return SpatialData(
        images={"Mayo_VisiumHD_789_full_image": visium_sdata.images["Mayo_VisiumHD_789_full_image"]},
        shapes={"Mayo_VisiumHD_789_square_016um": visium_sdata.shapes["Mayo_VisiumHD_789_square_016um"]},
        tables={"square_016um": visium_sdata.tables["square_016um"]}
    )

def calculate_centroid_and_max_edge_length(polygon: Polygon):
    """
    Calculate the centroid and maximum edge length of a polygon.

    Args:
    - polygon: Shapely Polygon object.

    Returns:
    - Tuple of centroid and maximum edge length.
    """
    centroid = polygon.centroid
    coords = list(polygon.exterior.coords)
    edge_lengths = [sqrt((coords[i][0] - coords[i + 1][0]) ** 2 + (coords[i][1] - coords[i + 1][1]) ** 2) for i in range(len(coords) - 1)]
    return centroid, max(edge_lengths)

# -----------------------
# Step 3: Merge cluster information with observations
# -----------------------
def merge_cluster_to_obs(obs_df: pd.DataFrame, cluster_file: str) -> pd.DataFrame:
    """
    Merge cluster information from a CSV file with SpatialData observations.

    Args:
    - obs_df: DataFrame containing observations from SpatialData.
    - cluster_file: Path to CSV file containing cluster information.

    Returns:
    - Updated DataFrame with cluster information merged.
    """
    cluster_df = pd.read_csv(cluster_file)
    merged_df = obs_df.merge(cluster_df[["Barcode", "Cluster"]], left_index=True, right_on="Barcode", how="left")
    merged_df.set_index("Barcode", inplace=True)
    merged_df.index.name = 'index'
    return merged_df

# -----------------------
# Step 4: Create ImageTilesDataset with Transformation
# -----------------------
def create_image_tiles_dataset(merged, transformed_square):
    """
    Create ImageTilesDataset from SpatialData.

    Args:
    - merged: SpatialData object after preprocessing.
    - transformed_square: Transformed shape data.

    Returns:
    - ImageTilesDataset instance.
    """
    return ImageTilesDataset(
        sdata=merged,
        regions_to_images={"Mayo_VisiumHD_789_square_016um": "Mayo_VisiumHD_789_full_image"},
        regions_to_coordinate_systems={"Mayo_VisiumHD_789_square_016um": "global"},
        table_name="square_016um",
        tile_dim_in_units = int(transformed_square.radius[0]),
        transform=transform_with_cell_types,
        rasterize=True,
        rasterize_kwargs={"target_width": int(transformed_square.radius[0])},
    )

# -----------------------
# Step 5: Extract and save features directly from tensors
# -----------------------
def extract_features_directly(dataset, metadata_df, output_dir):
    """
    Extract features directly from dataset tensors using GigaPath.

    Args:
    - dataset: Dataset object with tensors.
    - metadata_df: Metadata DataFrame with splits and labels.
    - output_dir: Directory to save features.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True).eval().to(device)

    transform = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    os.makedirs(output_dir, exist_ok=True)

    for idx, (tensor, label) in tqdm(enumerate(dataset), total=len(dataset), desc="Extracting features"):
        row = metadata_df.iloc[idx]
        split = row['split']
        obs_idx = row['input']
        label_dir = os.path.join(output_dir, str(row['label']))
        os.makedirs(label_dir, exist_ok=True)

        feature_path = os.path.join(label_dir, f"{split}_{obs_idx}.pt")
        tensor = transform(tensor.unsqueeze(0).to(device))

        with torch.no_grad():
            feature = model(tensor).squeeze()
        torch.save(feature.cpu(), feature_path)

    print(f"Features saved to {output_dir}")

# -----------------------
# Step 6: Metadata creation and compression
# -----------------------
def create_split_metadata(obs_df, output_csv_path):
    """
    Create metadata for train, val, and test splits.

    Args:
    - obs_df: Observations DataFrame from SpatialData.
    - output_csv_path: Path to save metadata CSV.
    """
    metadata = obs_df[['Cluster']].copy()
    metadata.reset_index(inplace=True)
    metadata.columns = ['input', 'label']

    num_samples = len(metadata)
    train_split = int(0.7 * num_samples)
    val_split = int(0.2 * num_samples)
    metadata['split'] = ['train'] * train_split + ['val'] * val_split + ['test'] * (num_samples - train_split - val_split)
    metadata.to_csv(output_csv_path, index=False)
    print(f"Metadata saved to {output_csv_path}")

def compress_features(output_dir, zip_path):
    """
    Compress extracted features into a zip file.

    Args:
    - output_dir: Directory containing features.
    - zip_path: Path to save compressed zip file.
    """
    shutil.make_archive(zip_path.replace('.zip', ''), 'zip', output_dir)
    print(f"Compressed features saved to {zip_path}")

# -----------------------
# Step 7: Main function for feature extraction
# -----------------------
def feature_extraction_main():
    path_read = "/coh_labs/dits/nsong/Mayo_VisiumHD/789/outs/"
    cluster_file = "/coh_labs/dits/nsong/Mayo_VisiumHD/789/outs/binned_outputs/square_016um/analysis/clustering/gene_expression_kmeans_5_clusters/clusters.csv"
    metadata_csv_path = "/coh_labs/dits/nsong/Mayo_VisiumHD/789/metadata.csv"
    feature_output_dir = "/coh_labs/dits/nsong/Mayo_VisiumHD/789/features"
    zip_output_path = "/coh_labs/dits/nsong/Mayo_VisiumHD/789/features.zip"

    # Load spatial data
    merged = load_spatial_data(path_read)
    print("SpatialData has been created.")

    # Preprocess shapes
    square = merged["Mayo_VisiumHD_789_square_016um"]
    transformed_square = transform(square, to_coordinate_system="global")
    transformed_square['centroid'], transformed_square['radius'] = zip(*transformed_square['geometry'].apply(calculate_centroid_and_max_edge_length))
    print(transformed_square.radius[0])

    # Merge cluster data
    merged_obs = merge_cluster_to_obs(merged["square_016um"].obs, cluster_file)
    merged["square_016um"].obs = merged_obs
    print(merged["square_016um"].obs.head(5))

    # Create dataset
    dataset = create_image_tiles_dataset(merged, transformed_square)

    # Create metadata
    create_split_metadata(merged_obs, metadata_csv_path)

    # Extract features
    metadata = pd.read_csv(metadata_csv_path)
    extract_features_directly(dataset, metadata, feature_output_dir)

    # Compress features
    compress_features(feature_output_dir, zip_output_path)


if __name__ == "__main__":
    feature_extraction_main()