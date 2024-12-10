import os
import cv2
import torch
import numpy as np
import pandas as pd
import scanpy as sc
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.mixture import GaussianMixture
import timm
from torchvision import transforms
from tqdm import tqdm
import shutil
import tempfile

# Function to classify gene expression using GMM with 3 components
def classify_gene_expression_gmm_tertiles(expression):
    # Reshape the data for GMM
    expression_reshaped = expression.reshape(-1, 1)

    # Fit GMM with three components
    gmm = GaussianMixture(n_components=3, random_state=0)
    gmm.fit(expression_reshaped)

    # Predict the component for each expression value
    labels = gmm.predict(expression_reshaped)

    # Get the means of the three components
    means = gmm.means_.flatten()

    # Sort means and assign labels accordingly (lowest mean -> 0, middle -> 1, highest -> 2)
    sorted_indices = np.argsort(means)
    label_mapping = {sorted_indices[0]: 0, sorted_indices[1]: 1, sorted_indices[2]: 2}
    labels = np.vectorize(label_mapping.get)(labels)

    return labels

# Function to split data into train, val, and test sets
def split_data(df):
    train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['Label'], random_state=42)
    train_df, val_df = train_test_split(train_df, test_size=0.2, stratify=train_df['Label'], random_state=42)

    # Add split information to each DataFrame
    train_df['split'] = 'train'
    val_df['split'] = 'val'
    test_df['split'] = 'test'

    # Combine all DataFrames
    metadata_df = pd.concat([train_df, val_df, test_df])

    # Create a new column for input that combines split and unique identifiers
    metadata_df['input'] = metadata_df.apply(lambda row: f"{row['split']}_{int(row['x'])}_{int(row['y'])}", axis=1)

    return metadata_df

# Function to extract patches from WSI
def extract_patches(spatial, WSI_path, radius, img_data_dir):
    img_ori = cv2.imread(WSI_path)
    if img_ori is None:
        raise FileNotFoundError(f"Failed to load image at {WSI_path}")

    img_height, img_width = img_ori.shape[:2]
    os.makedirs(img_data_dir, exist_ok=True)

    for idx, row in spatial.iterrows():
        if row[1] == 0:
            continue  # Skip if not captured

        x = int(row[5])
        y = int(row[4])

        x_start = max(0, x - radius)
        x_end = min(img_width, x + radius)
        y_start = max(0, y - radius)
        y_end = min(img_height, y + radius)

        tiles = img_ori[y_start:y_end, x_start:x_end]

        if tiles.size == 0:
            print(f"Warning: Empty tile at position ({x}, {y})")
            continue

        cv2.imwrite(os.path.join(img_data_dir, f'{x}-{y}.jpg'), tiles)

# Function to generate and save features
def extract_gigapath_features(metadata_df, img_data_dir, output_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tile_encoder = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True).eval().to(device)

    transform = transforms.Compose(
        [
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )

    os.makedirs(output_dir, exist_ok=True)

    # Create directories for each label (0, 1, 2)
    for label in metadata_df['Label'].unique():
        label_dir = os.path.join(output_dir, str(label))
        os.makedirs(label_dir, exist_ok=True)

    # Process each row in metadata and generate corresponding features with progress bar
    for index, row in tqdm(metadata_df.iterrows(), total=metadata_df.shape[0], desc="Generating features"):
        img_file = os.path.join(img_data_dir, f"{int(row['x'])}-{int(row['y'])}.jpg")
        if not os.path.exists(img_file):
            print(f"Image file {img_file} not found.")
            continue

        img = Image.open(img_file).convert("RGB")
        input_tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            output = tile_encoder(input_tensor).squeeze()

        # Save the features in the appropriate directory
        pt_file_name = f"{row['input']}.pt"
        pt_file_path = os.path.join(output_dir, str(row['Label']), pt_file_name)
        torch.save(output.cpu(), pt_file_path)

# Function to compress embeddings
def compress_embeddings(gene_folder, output_dir, gene_name):
    zip_file_path = os.path.join(gene_folder, f"{gene_name}_embeddings.zip")
    shutil.make_archive(zip_file_path.replace('.zip', ''), 'zip', output_dir)
    print(f"Embeddings folder compressed and saved to {zip_file_path}")

# Main function
def main():
    # Define paths and parameters
    main_path = "/mnt/nsong-data-disk/Craig_Spatial/Craig_SPA3_D/outs"
    subfolder = "Spatial_enrichment_tertiles_classifier"
    path = os.path.join(main_path, subfolder)
    WSI_path = os.path.join(main_path, "Craig_SPA3_D.tif")
    radius = 90

    # Ensure the main path exists
    os.makedirs(path, exist_ok=True)

    # Load the CSV file containing the top genes sorted by Moran's I
    enrichment_csv = os.path.join(main_path, "Spatial_enrichment_classifier", "Spatial_Enrichment.csv")
    top_genes_df = pd.read_csv(enrichment_csv)

    # Get the top ten gene names using the correct column name 'Name'
    top_ten_genes = top_genes_df['Name'].head(10).tolist()

    # Load and process the gene expression data once
    adata = sc.read_10x_h5(os.path.join(main_path, "filtered_feature_bc_matrix.h5"))
    adata.var_names_make_unique()

    # Load spatial data
    spatial = pd.read_csv(os.path.join(main_path, "spatial", "tissue_positions.csv"),
                          sep=",", header=None, na_filter=False, skiprows=1, index_col=0)

    # Add spatial data to adata.obs
    adata.obs["x1"] = spatial[1]

    # Select captured samples based on the spatial data
    adata = adata[adata.obs["x1"] == 1]

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # Extract barcodes
    barcodes = adata.obs.index

    # Loop through each gene
    for gene_name in top_ten_genes:
        print(f"Processing gene: {gene_name}")

        # Create a subfolder for the gene
        gene_folder = os.path.join(path, gene_name)
        os.makedirs(gene_folder, exist_ok=True)

        # Step 1: Extract gene expression data and apply GMM classification for tertiles
        if gene_name not in adata.var_names:
            print(f"Gene {gene_name} not found in the dataset. Skipping.")
            continue

        gene_expression = adata[:, gene_name].X.toarray().flatten()

        # Apply GMM classification for tertiles (0, 1, 2)
        gene_labels = classify_gene_expression_gmm_tertiles(gene_expression)

        # Combine the barcodes with gene labels
        result_df = pd.DataFrame({
            'Barcode': barcodes,
            f'{gene_name}_Expression': gene_expression,
            'Label': gene_labels
        })

        # Include spatial coordinates
        spatial_data = spatial[[2, 3, 4, 5]].copy()
        spatial_data.columns = ['row', 'col', 'y', 'x']

        # Combine gene expression DataFrame with spatial data
        final_df = result_df.join(spatial_data, on='Barcode')

        # Save the labeled gene expression data
        output_path = os.path.join(gene_folder, f"{gene_name}_labeled_exp_tertiles.csv")
        final_df.to_csv(output_path, index=False)

        # Step 2: Split the data
        metadata_df = split_data(final_df)

        # Step 3: Save the final metadata CSV
        meta = metadata_df[['input', 'Label', 'split']].copy()
        meta.columns = ['input', 'label', 'split']
        metadata_path = os.path.join(gene_folder, "meta_tertiles.csv")
        meta.to_csv(metadata_path, index=False)

        # Step 4: Extract patches from WSI using a temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            img_data_dir = temp_dir  # Use temporary directory for image patches
            extract_patches(spatial, WSI_path, radius, img_data_dir)

            # Step 5: Generate and save features
            output_dir = os.path.join(gene_folder, f"{gene_name}_embeddings_tertiles")
            extract_gigapath_features(metadata_df, img_data_dir, output_dir)

            # Step 6: Compress the embeddings folder
            compress_embeddings(gene_folder, output_dir, gene_name)

            # Remove the embeddings folder to save space
            shutil.rmtree(output_dir)

        # Remove the labeled expression CSV to save space
        os.remove(output_path)

        print(f"Processing for gene {gene_name} complete.")

    print("All genes processed successfully.")

# Execute the main function
if __name__ == "__main__":
    main()
