import argparse
import os
import pandas as pd
import numpy as np
import torch
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from torchvision import transforms
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import scanpy as sc
from PIL import Image
from urllib.request import urlopen
from sklearn.cluster import KMeans
from huggingface_hub import HfApi, login

def process_image(args):
    image_name, img_data_dir, transform, tile_encoder = args
    file_path = os.path.join(img_data_dir, image_name)
    img = Image.open(file_path).convert("RGB")
    input_tensor = transform(img).unsqueeze(0)
    
    with torch.no_grad():
        output = tile_encoder(input_tensor).squeeze()
    
    features = output.cpu().numpy().flatten()
    return features

def extract_gigapath_features(img_data_dir):
    # Initialize GigaPath model with the specified configuration
    tile_encoder = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True).eval()
    print("GigaPath model loaded.")
    
    # Define the transformation for the GigaPath model
    transform = transforms.Compose(
        [
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    
    images = [f for f in os.listdir(img_data_dir) if f.endswith(('.jpg', '.jpeg', '.tif', '.png'))]
    if not images:
        raise FileNotFoundError(f"No images found in directory {img_data_dir}")
    
    num_workers = cpu_count()
    print(f"Using {num_workers} workers for parallel processing.")
    
    with Pool(num_workers) as pool:
        args = [(image_name, img_data_dir, transform, tile_encoder) for image_name in images]
        cls_features = pool.map(process_image, args)
    
    cls_features_array = np.array(cls_features)
    
    # Normalize the features using StandardScaler
    scaler = StandardScaler()
    normalized_cls_features = scaler.fit_transform(cls_features_array)
    
    return normalized_cls_features


def add_barcode(img_features, image_data_dir, tissue_positions_path, n_components=10):
    # List and extract coordinates from image filenames
    images = [f for f in os.listdir(image_data_dir) if f.endswith(('.jpg', '.jpeg'))]
    coords = []
    for img in images:
        x, y = img[:-4].split('-')
        coords.append([int(y), int(x)])  # Assuming filenames are in the format 'y-x.jpg'

    # Create DataFrame for coordinates and ResNet/ViT features
    # coords_df = pd.DataFrame(coords, columns=['imagerow', 'imagecol'])
    coords_df = pd.DataFrame(coords, columns=['pxl_row_in_fullres', 'pxl_col_in_fullres'])
    img_features_df = pd.DataFrame(img_features)

    pca = PCA(n_components=n_components)
    img_features_pca = pca.fit_transform(img_features_df)
    img_features_pca_df = pd.DataFrame(img_features_pca)

    img_w_coord = pd.concat([coords_df, img_features_pca_df], axis=1)

    # Merge with tissue position data to align with barcodes
    # tissue_position = pd.read_csv(tissue_positions_path, sep=',', names=["barcode", "tissue", "row", "col", "imagerow", "imagecol"])
    tissue_position = pd.read_csv(tissue_positions_path, sep=',')
    combined_feature_matrix = pd.merge(tissue_position, img_w_coord, how='inner', on=['pxl_row_in_fullres', 'pxl_col_in_fullres'])

    # Drop unnecessary columns to keep only barcode and features
    # img_w_barcode = combined_feature_matrix.drop(["tissue", 'imagerow', 'imagecol', "row", "col"], axis=1)
    img_w_barcode = combined_feature_matrix.drop(["in_tissue",	"array_row", "array_col", "pxl_row_in_fullres",	"pxl_col_in_fullres"], axis=1)
    
    return img_w_barcode


def gene_expression_feature_extraction(path):
    adata = sc.read_10x_h5(os.path.join(path, "filtered_feature_bc_matrix.h5"))

    adata.var_names_make_unique()
    print(adata.var_names.is_unique)
    spatial = pd.read_csv(os.path.join(path, "spatial", "tissue_positions.csv"), sep=",", header=None, na_filter=False, skiprows=1, index_col=0)
    print("Spatial data head:\n", spatial.head())

    # Add spatial data to adata.obs
    adata.obs["x1"] = spatial[1]
    print("Adata.obs head after adding spatial data:\n", adata.obs.head())

    # Check unique values in x1
    print("Unique values in x1:", adata.obs["x1"].unique())

    # Select captured samples
    adata = adata[adata.obs["x1"] == 1]
    print("Shape of adata after filtering by x1:", adata.shape)

    # Ensure variable names are uppercase
    adata.var_names = [i.upper() for i in adata.var_names]
    adata.var["genename"] = adata.var.index.astype("str")
    print("Adata.var names:\n", adata.var_names)

    # # Perform CPM normalization, log transformation, and scaling
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.scale(adata)
    return adata


def integrate_image_and_gene_expression_features(img_features, image_data_dir, tissue_positions_path, gene_expression_data, n_components=10):
    # Extract coordinates and align image features
    images = [f for f in os.listdir(image_data_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    coords = [[int(img[:-4].split('-')[1]), int(img[:-4].split('-')[0])] for img in images]  # Assuming format 'y-x.jpg'

    # coords_df = pd.DataFrame(coords, columns=['imagerow', 'imagecol'])
    coords_df = pd.DataFrame(coords, columns=['pxl_row_in_fullres', 'pxl_col_in_fullres'])
    img_features_df = pd.DataFrame(img_features)

    pca = PCA(n_components=n_components)
    img_features_pca = pca.fit_transform(img_features_df)
    img_features_pca_df = pd.DataFrame(img_features_pca)

    img_w_coord = pd.concat([coords_df, img_features_pca_df], axis=1)

    # tissue_position = pd.read_csv(tissue_positions_path, sep=',', names=["barcode", "tissue", "row", "col", "imagerow", "imagecol"])
    tissue_position = pd.read_csv(tissue_positions_path, sep=',')
    # img_w_barcode = pd.merge(tissue_position, img_w_coord, how='inner', on=['imagerow', 'imagecol']).drop(["tissue", 'imagerow', 'imagecol', "row", "col"], axis=1)
    img_w_barcode = pd.merge(tissue_position, img_w_coord, how='inner', on=["pxl_row_in_fullres",	"pxl_col_in_fullres"]).drop(["in_tissue",	"array_row", "array_col", "pxl_row_in_fullres",	"pxl_col_in_fullres"], axis=1)

    # PCA on gene expression data
    gene_exp_df = pd.DataFrame(gene_expression_data.to_df())
    pca = PCA(n_components=n_components)
    gene_exp_pca = pca.fit_transform(gene_exp_df)
    exp_emb = pd.DataFrame(gene_exp_pca, index=gene_exp_df.index)
    exp_emb.reset_index(inplace=True)
    exp_emb.rename(columns={'index': 'barcode'}, inplace=True)

    # Merge image features with PCA-transformed gene expression data
    integrated_data = pd.merge(img_w_barcode, exp_emb, on='barcode', how='inner')
    return integrated_data


def run_kmeans_with_dynamic_k(path, feature_matrix, tissue_positions_path, start_k, model_type):
    # Construct the folder path based on the model type
    model_folder_path = os.path.join(path, "annotation", model_type)
    
    # Check if the folder exists, if not, create it
    if not os.path.exists(model_folder_path):
        os.makedirs(model_folder_path)
    
    # Reading the cluster file and finding the largest cluster number
    try:
        # cluster_data = pd.read_csv(os.path.join(path, "analysis", "clustering", "graphclust", "clusters.csv"), sep=',')
        cluster_data = pd.read_csv(os.path.join(path, "analysis", "clustering", "gene_expression_graphclust", "clusters.csv"), sep=',')
        if 'Cluster' in cluster_data.columns:
            max_cluster = cluster_data['Cluster'].max()
        elif 'cluster' in cluster_data.columns:
            max_cluster = cluster_data['cluster'].max()
        else:
            raise ValueError("Cluster column not found")
    except FileNotFoundError:
        raise FileNotFoundError("Cluster file not found at the specified path")
    
    # Looping through the range of clusters from start_k to max_cluster
    for k in range(start_k, max_cluster + 1):
        print(f"Running KMeans for K = {k}")
        
        # Running KMeans
        kmeans = KMeans(n_clusters=k, random_state=0, n_init=10)
        features = feature_matrix.iloc[:, 1:]  # Excluding the barcode column
        kmeans.fit(features)

        # Creating a new DataFrame with barcodes and their corresponding cluster labels
        cluster_labels = pd.DataFrame({
            'barcode': feature_matrix.iloc[:, 0],
            'cluster': kmeans.labels_ + 1
        })
        
        # tissue_position = pd.read_csv(tissue_positions_path, sep=',', names=["barcode", "tissue", "row", "col", "imagerow", "imagecol"])
        tissue_position = pd.read_csv(tissue_positions_path, sep=',')
        labels_with_spatial_locations = tissue_position.merge(cluster_labels, how='outer', on=['barcode'])
        
        cluster_labels_filename = os.path.join(model_folder_path, f"cluster_labels_k_{k}.csv")
        labels_with_locations_filename = os.path.join(model_folder_path, f"labels_with_locations_k_{k}.csv")
        
        #print(f"Saving cluster labels to {cluster_labels_filename}")
        cluster_labels.to_csv(cluster_labels_filename, index=False)
        print(f"Cluster labels saved successfully to {cluster_labels_filename}")
        
        #print(f"Saving labels with locations to {labels_with_locations_filename}")
        labels_with_spatial_locations.to_csv(labels_with_locations_filename, index=False, header=None)
        print(f"Labels with locations saved successfully to {labels_with_locations_filename}")



def main(path):
    HUGGINGFACE_HUB_TOKEN = "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"
    login(token=HUGGINGFACE_HUB_TOKEN)

    radius = 90
    img_data_dir = os.path.join(path, 'image')
    tissue_positions_path = os.path.join(path, "spatial", "tissue_positions.csv")
    n_components = 10 

    print("----------GIGAPATH------------")
    normalized_vit_features = extract_gigapath_features(img_data_dir)
    print(normalized_vit_features.shape)
    
    adata = gene_expression_feature_extraction(path)
    print("Gene expressions are stored as Adata.")
    
    img_vit_w_barcode = add_barcode(normalized_vit_features, img_data_dir, tissue_positions_path)
    print("vit feature size:", img_vit_w_barcode.shape)
    
    integrated_data_vit = integrate_image_and_gene_expression_features(normalized_vit_features, img_data_dir, tissue_positions_path, adata, n_components)
    print("integrated vit feature size:", integrated_data_vit.shape)

    print("------------Kmeans Clustering (Dynamic)----------------")
    run_kmeans_with_dynamic_k(path, img_vit_w_barcode, tissue_positions_path, 4, 'gigapath')
    run_kmeans_with_dynamic_k(path, integrated_data_vit, tissue_positions_path, 4, 'integrated_gigapath')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run GigaPath feature extraction and integration.")
    parser.add_argument("path", type=str, help="Path to the directory containing image and spatial data.")
    args = parser.parse_args()
    
    main(args.path)