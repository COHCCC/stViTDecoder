
#!/usr/bin/env python
# coding: utf-8

# Prov-GigaPath Demo
# This script provides a quick walkthrough of the Prov-GigaPath models.

import os
from huggingface_hub import HfApi, login
from gigapath.pipeline import tile_one_slide, load_tile_slide_encoder, run_inference_with_tile_encoder
import pickle
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

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
# Uncomment the following line to run the tiling
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

# KMeans clustering and plotting
print("Running KMeans clustering...")
with open(os.path.join(slide_dir, 'tile_encoder_outputs.pkl'), 'rb') as f:
    loaded_tile_encoder_outputs = pickle.load(f)

embeddings = loaded_tile_encoder_outputs['tile_embeds']
coordinates = loaded_tile_encoder_outputs['coords']

num_clusters = 8
kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(embeddings)
labels = kmeans.labels_
print("Finished KMeans labeling.")

plt.figure(figsize=(24, 24), dpi=300)  # Increased dpi for better resolution
#colors = plt.cm.get_cmap('tab10', num_clusters)  # Using a colormap

# for i in range(num_clusters):
    # cluster_coords = coordinates[labels == i]
    # plt.scatter(cluster_coords[:, 0], cluster_coords[:, 1], marker='s', label=f'Cluster {i+1}', alpha=0.8, s=1.5, color=colors(i))

for i in range(num_clusters):
    cluster_coords = coordinates[labels == i]
    plt.scatter(cluster_coords[:, 0], cluster_coords[:, 1], marker = 's', label=f'Cluster {i+1}', alpha=0.6, s=1.3)


plt.legend()
plt.title('KMeans Clustering of Tiles')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
output_plot_path = "/home/nsong/DLmodels/prov-gigapath-main/demo/DLS18_F00083122_kmeans_8_clustering.png"
plt.savefig(output_plot_path)
print(f"Plot saved as {output_plot_path}")
print("Clustering and plotting completed.")
