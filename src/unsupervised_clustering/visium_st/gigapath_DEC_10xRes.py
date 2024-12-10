import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from torchvision import transforms
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from PIL import Image
from huggingface_hub import login
from tqdm import tqdm


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Autoencoder definition
class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Autoencoder, self).__init__()
        # Encoder part
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        # Decoder part (not used in DEC, but useful for pre-training)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return z, x_recon

# DEC model definition with Autoencoder
class DEC(nn.Module):
    def __init__(self, autoencoder, n_clusters=10):
        super(DEC, self).__init__()
        self.autoencoder = autoencoder
        # Define the cluster layer
        self.cluster_layer = nn.Parameter(torch.Tensor(n_clusters, autoencoder.encoder[-1].out_features))
        nn.init.xavier_normal_(self.cluster_layer.data)

    def forward(self, x):
        z, _ = self.autoencoder(x)
        q = self.soft_assignment(z)
        return z, q

    def soft_assignment(self, z):
        # Calculate the distance to the cluster centers
        dist = torch.sum((z.unsqueeze(1) - self.cluster_layer) ** 2, dim=2)
        q = 1.0 / (1.0 + dist)  # Using Student t-distribution
        q = q.pow((2 + 1) / 2)
        q = (q.t() / torch.sum(q, dim=1)).t()
        return q

def target_distribution(q):
    weight = q ** 2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()

# Function for processing images and extracting GigaPath features
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
    
    from multiprocessing import Pool, cpu_count
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

    # Create DataFrame for coordinates and features
    coords_df = pd.DataFrame(coords, columns=['pxl_row_in_fullres', 'pxl_col_in_fullres'])
    img_features_df = pd.DataFrame(img_features)

    pca = PCA(n_components=n_components)
    img_features_pca = pca.fit_transform(img_features_df)
    img_features_pca_df = pd.DataFrame(img_features_pca)

    img_w_coord = pd.concat([coords_df, img_features_pca_df], axis=1)

    # Merge with tissue position data to align with barcodes
    tissue_position = pd.read_csv(tissue_positions_path, sep=',')
    combined_feature_matrix = pd.merge(tissue_position, img_w_coord, how='inner', on=['pxl_row_in_fullres', 'pxl_col_in_fullres'])

    # Drop unnecessary columns to keep only barcode and features
    img_w_barcode = combined_feature_matrix.drop(["in_tissue",	"array_row", "array_col", "pxl_row_in_fullres",	"pxl_col_in_fullres"], axis=1)
    
    return img_w_barcode

# Pre-training the autoencoder
def pretrain_autoencoder(autoencoder, features, epochs=50, lr=0.001):
    optimizer = optim.Adam(autoencoder.parameters(), lr=lr)
    for epoch in range(epochs):
        autoencoder.train()
        optimizer.zero_grad()
        z, x_recon = autoencoder(features)
        recon_loss = nn.MSELoss()(x_recon, features)
        recon_loss.backward()
        optimizer.step()
        print(f'Epoch {epoch+1}/{epochs}, Reconstruction Loss: {recon_loss.item():.4f}')

def run_dec_with_autoencoder(path, feature_matrix, tissue_positions_path, start_k, model_type):
    model_folder_path = os.path.join(path, "annotation", f"dec_{model_type}")
    
    if not os.path.exists(model_folder_path):
        os.makedirs(model_folder_path)
    
    try:
        cluster_data = pd.read_csv(os.path.join(path, "analysis", "clustering", "gene_expression_graphclust", "clusters.csv"), sep=',')
        if 'Cluster' in cluster_data.columns:
            max_cluster = cluster_data['Cluster'].max()
        elif 'cluster' in cluster_data.columns:
            max_cluster = cluster_data['cluster'].max()
        else:
            raise ValueError("Cluster column not found")
    except FileNotFoundError:
        raise FileNotFoundError("Cluster file not found at the specified path")
    
    input_dim = feature_matrix.shape[1] - 1
    hidden_dim = 500
    latent_dim = 10

    autoencoder = Autoencoder(input_dim=input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim).to(device)

    # Pretrain the autoencoder
    features_tensor = torch.Tensor(feature_matrix.iloc[:, 1:].values).float().to(device)
    pretrain_autoencoder(autoencoder, features_tensor)

    for k in range(start_k, max_cluster + 1):
        print(f"Running DEC for K = {k}")

        # Using KMeans to initialize cluster centers
        z, _ = autoencoder(features_tensor)
        z = z.detach().cpu().numpy()
        kmeans = KMeans(n_clusters=k, n_init=20)
        y_pred = kmeans.fit_predict(z)
        
        dec_model = DEC(autoencoder, n_clusters=k).to(device)
        dec_model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_, dtype=torch.float).to(device)

        optimizer = optim.Adam(dec_model.parameters(), lr=0.001)
        batch_size = 256
        train_loader = torch.utils.data.DataLoader(features_tensor, batch_size=batch_size, shuffle=True)

        # Train the DEC model
        for epoch in range(100):
            dec_model.train()
            epoch_loss = 0
            for batch in train_loader:
                batch = batch.float()

                # Forward pass
                z, q = dec_model(batch)

                # Calculate target distribution p
                p = target_distribution(q).detach()

                # KL divergence loss
                kl_loss = nn.KLDivLoss()(q.log(), p)

                # Backpropagation
                optimizer.zero_grad()
                kl_loss.backward()
                optimizer.step()

                epoch_loss += kl_loss.item()

            print(f"Epoch {epoch+1}, Loss: {epoch_loss / len(train_loader)}")

        # Generate cluster labels
        dec_model.eval()
        with torch.no_grad():
            z, q = dec_model(features_tensor)
            cluster_labels = torch.argmax(q, dim=1).cpu().numpy() + 1
        
        # Save cluster results
        cluster_labels_df = pd.DataFrame({
            'barcode': feature_matrix.iloc[:, 0],
            'cluster': cluster_labels
        })
        
        tissue_position = pd.read_csv(tissue_positions_path, sep=',')
        labels_with_spatial_locations = tissue_position.merge(cluster_labels_df, how='outer', on=['barcode'])
        
        cluster_labels_filename = os.path.join(model_folder_path, f"cluster_labels_k_{k}.csv")
        labels_with_locations_filename = os.path.join(model_folder_path, f"labels_with_locations_k_{k}.csv")
        
        cluster_labels_df.to_csv(cluster_labels_filename, index=False)
        print(f"Cluster labels saved successfully to {cluster_labels_filename}")
        
        labels_with_spatial_locations.to_csv(labels_with_locations_filename, index=False, header=None)
        print(f"Labels with locations saved successfully to {labels_with_locations_filename}")

def main(path):
    HUGGINGFACE_HUB_TOKEN = "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"
    login(token=HUGGINGFACE_HUB_TOKEN)

    img_data_dir = os.path.join(path, 'image')
    tissue_positions_path = os.path.join(path, "spatial", "tissue_positions.csv")
    n_components = 10 

    print("----------GIGAPATH------------")
    normalized_vit_features = np.load('/home/nsong/Craig_SPA5_D/normalized_gigapath_features.npy')
    print(normalized_vit_features.shape)
    
    img_vit_w_barcode = add_barcode(normalized_vit_features, img_data_dir, tissue_positions_path)
    print("vit feature size:", img_vit_w_barcode.shape)

    print("------------DEC with Autoencoder Clustering----------------")
    run_dec_with_autoencoder(path, img_vit_w_barcode, tissue_positions_path, 4, 'dec_ae_gigapath')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run GigaPath DEC with Autoencoder clustering.")
    parser.add_argument("path", type=str, help="Path to the directory containing image and spatial data.")
    args = parser.parse_args()
    
    main(args.path)