import os
import cv2
import torch
import numpy as np
from PIL import Image
import timm
from torchvision import transforms
from sklearn.model_selection import train_test_split
import pandas as pd
import scanpy as sc
from tqdm import tqdm
import shutil
from huggingface_hub import login

os.environ["HUGGINGFACE_HUB_TOKEN"] = "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"
login()
# Step 1: Load and process the gene expression data
def load_and_process_gene_expression(path):
    adata = sc.read_10x_h5(os.path.join(path, "filtered_feature_bc_matrix.h5"))
    adata.var_names_make_unique()
    
    spatial = pd.read_csv(os.path.join(path, "spatial", "tissue_positions.csv"), sep=",", header=None, na_filter=False, skiprows=1, index_col=0)
    adata.obs["x1"] = spatial[1]
    adata = adata[adata.obs["x1"] == 1]
    barcodes = adata.obs.index

    oligodendrocytes_genes = ['OLIG1', 'OLIG2', 'PDGFRA', 'MYRF', 'PLP1', 'MOG']
    microglia_genes = ['AIF1', 'CD68', 'ITGAM', 'CD14', 'PTPRC', 'CD80', 'ENG']
    hypoxia_genes = ['HIF1A', 'VEGFA', 'MMP2']

    oligodendrocytes_genes = [gene for gene in oligodendrocytes_genes if gene in adata.var_names]
    microglia_genes = [gene for gene in microglia_genes if gene in adata.var_names]
    hypoxia_genes = [gene for gene in hypoxia_genes if gene in adata.var_names]

    adata.obs['log2_sum_oligodendrocytes'] = np.log2(adata[:, oligodendrocytes_genes].X.sum(axis=1) + 1)
    adata.obs['log2_sum_microglia'] = np.log2(adata[:, microglia_genes].X.sum(axis=1) + 1)
    adata.obs['log2_sum_hypoxia'] = np.log2(adata[:, hypoxia_genes].X.sum(axis=1) + 1)

    result_df = pd.DataFrame({
        'Barcode': barcodes,
        'log2_sum_oligodendrocytes': adata.obs['log2_sum_oligodendrocytes'].values,
        'log2_sum_microglia': adata.obs['log2_sum_microglia'].values,
        'log2_sum_hypoxia': adata.obs['log2_sum_hypoxia'].values
    })

    spatial_data = spatial[[2, 3, 4, 5]].copy()
    spatial_data.columns = ['row', 'col', 'y', 'x']
    final_df = result_df.join(spatial_data, on='Barcode')

    return final_df

# Step 2: Process the data and assign labels
def process_gene_expression(df):
    # Initialize the Label column with 0
    df['Label'] = 0

    # Apply conditions for labeling according to priority
    df.loc[df['log2_sum_oligodendrocytes'] > 5, 'Label'] = 1
    df.loc[(df['log2_sum_hypoxia'] > 6.5) & (df['Label'] == 0), 'Label'] = 2  # Microglia, only if not labeled as oligodendrocytes
    df.loc[(df['log2_sum_microglia'] > 4.2) & (df['Label'] == 0), 'Label'] = 3  # Hypoxia, only if not labeled as oligodendrocytes or microglia

    # Print the number of spots labeled as 1, 2, and 3
    print(f"Number of spots labeled as 1 (oligodendrocytes): {df[df['Label'] == 1].shape[0]}")
    print(f"Number of spots labeled as 2 (hypoxia): {df[df['Label'] == 2].shape[0]}")
    print(f"Number of spots labeled as 3 (microglia): {df[df['Label'] == 3].shape[0]}")

    return df

# Step 3: Split the data into train, val, and test sets
def split_data(df):
    df = df[df['Label'] != 0]

    train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['Label'], random_state=42)
    train_df, val_df = train_test_split(train_df, test_size=0.2, stratify=train_df['Label'], random_state=42)

    train_df['split'] = 'train'
    val_df['split'] = 'val'
    test_df['split'] = 'test'

    metadata_df = pd.concat([train_df, val_df, test_df])
    metadata_df['input'] = metadata_df.apply(lambda row: f"{row['split']}_{int(row['x'])}_{int(row['y'])}", axis=1)
    
    return metadata_df

# Step 4: Extract patches from WSI
def extract_patches(path, WSI_path, radius):
    img_ori = cv2.imread(WSI_path)
    if img_ori is None:
        raise FileNotFoundError(f"Failed to load image at {WSI_path}")

    img_height, img_width = img_ori.shape[:2]  
    os.makedirs(os.path.join(path, "hypoxia_prob_image_GBM"), exist_ok=True)

    with open(os.path.join(path, "spatial", "tissue_positions.csv"), 'r') as file:
        for line in file:
            if line.startswith('b') or int(line.rstrip().split(',')[1]) == 0:
                continue
            x = int(line.rstrip().split(',')[5])
            y = int(line.rstrip().split(',')[4])

            x_start = max(0, x-radius)
            x_end = min(img_width, x+radius)
            y_start = max(0, y-radius)
            y_end = min(img_height, y+radius)

            tiles = img_ori[y_start:y_end, x_start:x_end]

            if tiles.size == 0:
                print(f"Warning: Empty tile at position ({x}, {y})")
                continue

            cv2.imwrite(os.path.join(path, 'hypoxia_prob_image_GBM', f'{x}-{y}.jpg'), tiles)

# Step 5: Generate and save features
def extract_gigapath_features(metadata_df, img_data_dir, output_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tile_encoder = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True).eval().to(device)
    
    transform = transforms.Compose(
        [
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            # transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    
    os.makedirs(output_dir, exist_ok=True)

    for label in metadata_df['Label'].unique():
        label_dir = os.path.join(output_dir, str(label))
        os.makedirs(label_dir, exist_ok=True)
    
    for index, row in tqdm(metadata_df.iterrows(), total=metadata_df.shape[0], desc="Generating features"):
        img_file = os.path.join(img_data_dir, f"{row['x']}-{row['y']}.jpg")
        img = Image.open(img_file).convert("RGB")
        input_tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            output = tile_encoder(input_tensor).squeeze()

        pt_file_name = f"{row['input']}.pt"
        pt_file_path = os.path.join(output_dir, str(row['Label']), pt_file_name)
        torch.save(output.cpu(), pt_file_path)

# Main execution flowraig_Spatial/Craig_SPA5_D/outs/Craig_SPA5_D.tif
if __name__ == "__main__":
    path = "/mnt/nsong-data-disk/Craig_Spatial/Craig_SPA8_A/outs"
    WSI_path = "/mnt/nsong-data-disk/Craig_Spatial/Craig_SPA8_A/outs/Craig_SPA8_A.tif"  
    radius = 90  
    output_dir = os.path.join(path, 'GBM_hypoxia_embeddings') 

    final_df = load_and_process_gene_expression(path)
    output_path = "/mnt/nsong-data-disk/Craig_Spatial/Craig_SPA8_A/outs/oligo_microglia_hypoxia_labeled_spots.csv"
    final_df.to_csv(output_path, index=False)

    df = process_gene_expression(final_df)
    metadata_df = split_data(df)

    meta = metadata_df[['input', 'Label', 'split']].copy()
    meta.columns = ['input', 'label', 'split']
    metadata_path = "/mnt/nsong-data-disk/Craig_Spatial/Craig_SPA8_A/outs/oligo_microglia_hypoxia_meta.csv"
    meta.to_csv(metadata_path, index=False)

    extract_patches(path, WSI_path, radius)
    img_data_dir = os.path.join(path, "hypoxia_prob_image_GBM")
    extract_gigapath_features(metadata_df, img_data_dir, output_dir)

    zip_file_path = os.path.join(path, "GigaPath_oligo_microglia_hypoxia_embeddings.zip")  
    shutil.make_archive(zip_file_path.replace('.zip', ''), 'zip', output_dir)

    print(f"Embeddings folder compressed and saved to {zip_file_path}")
    print("Processing complete.")