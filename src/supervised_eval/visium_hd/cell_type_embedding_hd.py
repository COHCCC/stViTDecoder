import sys
import os
import time
import logging
from typing import Dict, Tuple
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.optim import Adam
from spatialdata import SpatialData, read_zarr, transform
from spatialdata.dataloader.datasets import ImageTilesDataset
from PIL import Image
import timm
from sklearn.model_selection import train_test_split
from sklearn.mixture import GaussianMixture
from tqdm import tqdm
import shutil
from shapely.geometry import Polygon
from spatialdata.models import ShapesModel
from math import sqrt
import importlib


# Reload custom modules if needed
if 'visiumHD' not in sys.path:
    sys.path.append('visiumHD')
import select_utils
importlib.reload(select_utils)
import transform_utils
importlib.reload(transform_utils)
from transform_utils import transform_with_cell_types

# Load spatial data
def load_spatial_data(path_read: str) -> SpatialData:
    path_write = os.path.join(path_read, "789.zarr")
    assert os.path.isdir(path_write), "Zarr directory not found"
    visium_sdata = read_zarr(path_write)
    return SpatialData(
        images={"Mayo_VisiumHD_789_full_image": visium_sdata.images["Mayo_VisiumHD_789_full_image"]},
        shapes={"Mayo_VisiumHD_789_square_016um": visium_sdata.shapes["Mayo_VisiumHD_789_square_016um"]},
        tables={"square_016um": visium_sdata.tables["square_016um"]}
    )

################################################
# Generate clusters based on fold change thershold
################################################
# def generate_clusters_from_genes(anndata_obj, gene_sets):
#     anndata_obj.var_names_make_unique()
#     sc.pp.normalize_total(anndata_obj, target_sum=1e4, inplace=True)

#     cluster_df = pd.DataFrame(index=anndata_obj.obs.index)
#     cluster_df["Cluster"] = 0  # Default cluster as 0 (unlabeled)

#     for cluster, genes in gene_sets.items():
#         valid_genes = [gene for gene in genes if gene in anndata_obj.var_names]
#         if valid_genes:
#             cluster_df[f"log2_sum_{cluster}"] = np.log2(anndata_obj[:, valid_genes].X.sum(axis=1) + 1).A1

#     # Assign clusters based on thresholds
#     cluster_df.loc[cluster_df["log2_sum_fibroblast"] > 6, "Cluster"] = 1
#     cluster_df.loc[(cluster_df["log2_sum_epithelial"] > 4.8) & (cluster_df["Cluster"] == 0), "Cluster"] = 2
#     return cluster_df

################################################
# Generate clusters based on GMM 
################################################
def classify_gene_expression_gmm(expression):
    """
    Classify gene expression values into two groups (on/off) using Gaussian Mixture Model (GMM).

    Args:
    - expression: 1D numpy array of gene expression values.

    Returns:
    - labels: Array of 0s and 1s indicating the classification.
    """
    # Reshape the data for GMM
    expression_reshaped = expression.reshape(-1, 1)

    # Fit GMM with two components
    gmm = GaussianMixture(n_components=2, random_state=0)
    gmm.fit(expression_reshaped)

    # Predict the component for each expression value
    labels = gmm.predict(expression_reshaped)

    # Ensure that the higher mean corresponds to label 1 (high expression)
    means = gmm.means_.flatten()
    if means[0] > means[1]:
        labels = np.where(labels == 0, 1, 0)

    return labels

def generate_clusters_with_gmm_and_priority_three_types(anndata_obj, gene_sets):
    """
    Generate clusters for three cell types based on GMM classification with priority logic.

    Args:
    - anndata_obj: AnnData object containing gene expression data.
    - gene_sets: Dictionary containing gene sets for each cell type with labels as keys.

    Returns:
    - cluster_df: DataFrame with Cluster column added based on GMM classification and priority logic.
    """
    # Ensure unique gene names and normalize the data
    anndata_obj.var_names_make_unique()
    sc.pp.normalize_total(anndata_obj, target_sum=1e4, inplace=True)

    # Initialize DataFrame to store clusters
    cluster_df = pd.DataFrame(index=anndata_obj.obs.index)
    cluster_df["Cluster"] = 0  # Default cluster as 0 (unlabeled)

    # Store GMM thresholds for each gene set
    gmm_thresholds = {}

    # Step 1: Process each gene set to classify with GMM
    for cluster, genes in gene_sets.items():
        # Filter valid genes present in the data
        valid_genes = [gene for gene in genes if gene in anndata_obj.var_names]

        if valid_genes:
            # Sum the expression levels of the genes in the set
            expression = anndata_obj[:, valid_genes].X.sum(axis=1).A1

            # Use GMM to classify the expression levels
            gmm_labels = classify_gene_expression_gmm(expression)

            # Save the threshold for high expression
            gmm_thresholds[cluster] = {
                "expression": expression,
                "labels": gmm_labels,
            }

    # Step 2: Assign clusters based on priority
    # Priority: astrocytes (1) > endothelial (2) > mesenchymal (3)
    if "astrocytes" in gmm_thresholds:
        astrocytes_labels = gmm_thresholds["astrocytes"]["labels"]
        cluster_df.loc[astrocytes_labels == 1, "Cluster"] = 1

    if "endothelial" in gmm_thresholds:
        endothelial_labels = gmm_thresholds["endothelial"]["labels"]
        cluster_df.loc[(endothelial_labels == 1) & (cluster_df["Cluster"] == 0), "Cluster"] = 2

    if "mesenchymal" in gmm_thresholds:
        mesenchymal_labels = gmm_thresholds["mesenchymal"]["labels"]
        cluster_df.loc[(mesenchymal_labels == 1) & (cluster_df["Cluster"] == 0), "Cluster"] = 3

    return cluster_df

# Update obs with gene-based clusters
def update_obs_with_clusters(obs_df, anndata_obj, gene_sets):
    cluster_df = generate_clusters_with_gmm_and_priority_three_types(anndata_obj, gene_sets)
    obs_df = obs_df.merge(cluster_df["Cluster"], left_index=True, right_index=True, how="left")
    return obs_df


def filter_spatialdata_by_cluster(merged: SpatialData):

    filtered_obs = merged["square_016um"].obs[merged["square_016um"].obs["Cluster"] != 0]
    filtered_indices = filtered_obs["location_id"].values
    filtered_regions = merged["Mayo_VisiumHD_789_square_016um"].loc[filtered_indices]

    merged.tables["square_016um"] = merged.tables["square_016um"][filtered_obs.index]

    merged["Mayo_VisiumHD_789_square_016um"] = ShapesModel.parse(filtered_regions)

    print("Filtered obs shape:", merged["square_016um"].obs.shape)
    print("Filtered Regions shape:", merged["Mayo_VisiumHD_789_square_016um"].shape)

    return merged

# Calculate centroid and max edge length for polygons
def calculate_centroid_and_max_edge_length(polygon: Polygon):
    centroid = polygon.centroid
    coords = list(polygon.exterior.coords)
    edge_lengths = [sqrt((coords[i][0] - coords[i + 1][0]) ** 2 + (coords[i][1] - coords[i + 1][1]) ** 2) for i in range(len(coords) - 1)]
    return centroid, max(edge_lengths)


def create_image_tiles_dataloader(merged, transformed_square, batch_size=32, num_workers=4):
    """
    Create ImageTilesDataset and return DataLoader for SpatialData.

    Args:
    - merged: SpatialData object.
    - transformed_square: Transformed Regions (shapes).
    - batch_size: Number of samples per batch.
    - num_workers: Number of worker processes for data loading.

    Returns:
    - DataLoader instance for the dataset.
    """
    print("Initializing ImageTilesDataset...")
    dataset = ImageTilesDataset(
        sdata=merged,
        regions_to_images={"Mayo_VisiumHD_789_square_016um": "Mayo_VisiumHD_789_full_image"},
        regions_to_coordinate_systems={"Mayo_VisiumHD_789_square_016um": "global"},
        table_name="square_016um",
        tile_dim_in_units=int(transformed_square.radius[0]),
        transform=transform_with_cell_types,
        rasterize=True,
        rasterize_kwargs={"target_width": int(transformed_square.radius[0])},
    )

    print(f"Dataset length: {len(dataset)}")
    try:
        sample = dataset[0]
        print(f"Sample loaded: {sample[0].shape}, Label: {sample[1]}")
    except Exception as e:
        print(f"Error when accessing dataset sample: {e}")
        raise

    print("Initializing DataLoader...")

    # 如果 num_workers 为 0，设置 prefetch_factor 为 None
    prefetch_factor = 2 if num_workers > 0 else None

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=True,
        prefetch_factor=prefetch_factor,  # 根据 num_workers 动态设置
        persistent_workers=num_workers > 0,  # 仅在 num_workers > 0 时启用
    )

    try:
        for tensors, labels in dataloader:
            print(f"First batch loaded: Tensors shape: {tensors.shape}, Labels shape: {labels.shape}")
            break
    except Exception as e:
        print(f"Error when accessing DataLoader batch: {e}")
        raise

    return dataloader
# Create metadata
def create_split_metadata(obs_df, output_csv_path):
    metadata = obs_df[obs_df["Cluster"] != 0][["Cluster"]].copy()  # Filter out unlabeled spots
    metadata.reset_index(inplace=True)
    metadata.columns = ['input', 'label']

    train_split = int(0.7 * len(metadata))
    val_split = int(0.2 * len(metadata))
    metadata['split'] = ['train'] * train_split + ['val'] * val_split + ['test'] * (len(metadata) - train_split - val_split)
    metadata.to_csv(output_csv_path, index=False)
    print(f"Metadata saved to {output_csv_path}")


def extract_features_directly(dataloader, metadata_df, output_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True).eval().to(device)

    transform = transforms.Compose([
        # transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        # transforms.CenterCrop(224),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    os.makedirs(output_dir, exist_ok=True)

    for batch_idx, (tensors, labels) in tqdm(enumerate(dataloader), total=len(dataloader), desc="Extracting features"):
        print(f"Batch {batch_idx}: tensors shape: {tensors.shape}, labels shape: {labels.shape}")
        
        # 打印模型和张量的设备信息
        print(f"Model device: {next(model.parameters()).device}")
        print(f"Tensors device: {tensors.device}")

        # 将张量移动到正确的设备
        tensors = tensors.to(device)
        print(f"Tensors moved to {tensors.device}")
        
        labels = labels.numpy()
        print(f"Labels: {labels}")

        # 提取特征
        with torch.no_grad():
            features = model(tensors).cpu()
            print(f"Extracted features shape: {features.shape}")

        # 保存特征
        for i, label in enumerate(labels):
            row = metadata_df.iloc[batch_idx * dataloader.batch_size + i]
            split = row['split']
            obs_idx = row['input']
            label_dir = os.path.join(output_dir, str(label))
            os.makedirs(label_dir, exist_ok=True)

            feature_path = os.path.join(label_dir, f"{split}_{obs_idx}.pt")
            torch.save(features[i], feature_path)

    print(f"Features saved to {output_dir}")

# Compress features
def compress_features(output_dir, zip_path):
    shutil.make_archive(zip_path.replace('.zip', ''), 'zip', output_dir)
    print(f"Compressed features saved to {zip_path}")

# -----------------------
# Step 3: Main Function
# -----------------------
def feature_extraction_main():
    path_read = "/coh_labs/dits/nsong/Mayo_VisiumHD/789/outs/"
    metadata_csv_path = "/coh_labs/dits/nsong/Mayo_VisiumHD/789/metadata_3_types.csv"
    feature_output_dir = "/coh_labs/dits/nsong/Mayo_VisiumHD/789/features_3_types"
    zip_output_path = "/coh_labs/dits/nsong/Mayo_VisiumHD/789/features_3_types.zip"

    gene_sets = {
        "astrocytes": ['S100B', 'AQP4'],
        "endothelial": ['IGFBP7', 'FLT1', 'COBLL1'],
        "mesenchymal": ['VEGFA', 'NDRG1', 'CA9']
    }

    # Load spatial data
    merged = load_spatial_data(path_read)
    print("SpatialData has been created.")

    # Transform Regions (shapes) to global coordinates
    square = merged["Mayo_VisiumHD_789_square_016um"]
    transformed_square = transform(square, to_coordinate_system="global")
    transformed_square['centroid'], transformed_square['radius'] = zip(*transformed_square['geometry'].apply(calculate_centroid_and_max_edge_length))

    # Update obs with gene-based clusters
    anndata_obj = merged.tables["square_016um"]
    merged_obs = update_obs_with_clusters(merged["square_016um"].obs, anndata_obj, gene_sets)
    merged["square_016um"].obs = merged_obs
    print(merged["square_016um"].obs["Cluster"].value_counts())

    # Apply filtering to SpatialData
    merged = filter_spatialdata_by_cluster(merged)

    # Create metadata
    create_split_metadata(merged["square_016um"].obs, metadata_csv_path)

    # Create DataLoader instead of raw dataset
    dataloader = create_image_tiles_dataloader(merged, transformed_square, batch_size=32, num_workers=4)
    # Test DataLoader
    for tensors, labels in dataloader:
        print(f"Tensors shape: {tensors.shape}, Labels shape: {labels.shape}")
        break  # 只测试一批数据
    # Extract features
    metadata = pd.read_csv(metadata_csv_path)
    extract_features_directly(dataloader, metadata, feature_output_dir)

    compress_features(feature_output_dir, zip_output_path)

if __name__ == "__main__":
    feature_extraction_main()