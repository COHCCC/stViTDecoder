import os
from huggingface_hub import HfApi, login
from gigapath.pipeline import tile_one_slide, load_tile_slide_encoder, run_inference_with_tile_encoder
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tqdm import tqdm 

# Set Hugging Face Token
HF_TOKEN = "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"
os.environ["HF_TOKEN"] = HF_TOKEN
print("Hugging Face Token set successfully.")

# Login to Hugging Face
print("Logging into Hugging Face...")
login(token=HF_TOKEN)
print("Logged in successfully.")

# Tiling the slide
slide_path = "/coh_labs/dits/johnlee/DLS_gbm/HE_40X/DLS18_F00083122-2024-06-24-16.19.44.ndpi"
tmp_dir = '/coh_labs/dits/nsong/outputs/preprocessing/'
print("Tiling the slide...")
tile_one_slide(slide_path, save_dir=tmp_dir, level=1)
print("Slide tiling completed.")

# Load the tile images
slide_dir = "/coh_labs/dits/nsong/outputs/preprocessing/output/" + os.path.basename(slide_path) + "/"
image_paths = [os.path.join(slide_dir, img) for img in os.listdir(slide_dir) if img.endswith('.png')]
print(f"Found {len(image_paths)} image tiles")

# Load the Prov-GigaPath model (tile and slide encoder models)
print("Loading the Prov-GigaPath models...")
tile_encoder, slide_encoder_model = load_tile_slide_encoder(global_pool=True)
print("Models loaded successfully.")

# Run tile-level inference
print("Running tile-level inference...")
tile_encoder_outputs = run_inference_with_tile_encoder(image_paths, tile_encoder)
for k in tile_encoder_outputs.keys():
    print(f"tile_encoder_outputs[{k}].shape: {tile_encoder_outputs[k].shape}")

# Save the tile encoder outputs
with open(os.path.join(slide_dir, 'tile_encoder_outputs.pkl'), 'wb') as f:
    pickle.dump(tile_encoder_outputs, f)
print("Tile encoder outputs saved.")

# Load the tile encoder outputs
with open(os.path.join(slide_dir, 'tile_encoder_outputs.pkl'), 'rb') as f:
    loaded_tile_encoder_outputs = pickle.load(f)

embeddings = loaded_tile_encoder_outputs['tile_embeds']
coordinates = loaded_tile_encoder_outputs['coords']

# DEC Model Definition
class DEC(nn.Module):
    def __init__(self, input_dim, latent_dim, n_clusters):
        super(DEC, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim)
        )
        self.cluster_layer = nn.Parameter(torch.Tensor(n_clusters, latent_dim))
        nn.init.xavier_normal_(self.cluster_layer.data)

    def forward(self, x):
        z = self.encoder(x)
        q = self.soft_assignment(z)
        return z, q

    def soft_assignment(self, z):
        dist = torch.sum((z.unsqueeze(1) - self.cluster_layer)**2, dim=2)
        q = 1.0 / (1.0 + dist)
        q = q.pow((2 + 1) / 2)
        q = (q.t() / torch.sum(q, dim=1)).t()
        return q

def target_distribution(q):
    weight = q ** 2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()

# Initialize DEC Model
input_dim = embeddings.shape[1]
latent_dim = 512
n_clusters = 8
dec_model = DEC(input_dim=input_dim, latent_dim=latent_dim, n_clusters=n_clusters)

# Optimizer
optimizer = optim.Adam(dec_model.parameters(), lr=0.001)

# Train DEC Model with tqdm progress tracking
train_features = torch.tensor(embeddings, dtype=torch.float32)
num_epochs = 100

for epoch in tqdm(range(num_epochs), desc="Training DEC", unit="epoch"):
    dec_model.train()
    optimizer.zero_grad()
    z, q = dec_model(train_features)
    p = target_distribution(q).detach()
    kl_loss = nn.KLDivLoss()(q.log(), p)
    kl_loss.backward()
    optimizer.step()

    # Print loss every 10 epochs
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {kl_loss.item()}")

# Inference and argmax(q) to get hard labels
dec_model.eval()
with torch.no_grad():
    _, q = dec_model(train_features)
    cluster_labels = torch.argmax(q, dim=1).cpu().numpy()

# Plotting the DEC clustering results
plt.figure(figsize=(24, 24), dpi=300)

for i in range(n_clusters):
    cluster_coords = coordinates[cluster_labels == i]
    plt.scatter(cluster_coords[:, 0], cluster_coords[:, 1], marker='s', label=f'Cluster {i+1}', alpha=0.6, s=1.3)

plt.legend()
plt.title('DEC Clustering of Tiles')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
output_plot_path = "/home/nsong/DLmodels/prov-gigapath-main/demo/DLS18_F00083122_DEC_clustering.png"
plt.savefig(output_plot_path)
print(f"Plot saved as {output_plot_path}")
print("DEC clustering and plotting completed.")