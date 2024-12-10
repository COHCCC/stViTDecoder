
# <span id="jump"><font color=#00297D>Integrating Spatial Transcriptomics and Histopathology with Deep Learning</font></div></span>

## <font color=#00297D>Index</font>

- [1. Introduction & Objectives](#1)
  - [1.1 Background](#1.1)
  - [1.2 Objectives](#1.2)
  - [1.3 Data](#1.3)
- [2. File Structure](#2)
- [3. Tasks & Workflows](#3)
  - [3.1 Spot-level Prediction](#3.1)
  - [3.2 Unsupervised Clustering](#3.2)
- [4. Results](#4)
- [6. References](#6)
- [7. Acknowledgements](#7)

## <h2 id="1"><font color=#00297D>1. Introduction & Objectives</font></h2>

### <h2 id="1.1"><font color=#00297D>1.1 Background</font></h2>

Spatial transcriptomics (ST) provides unprecedented insights into the spatial landscape of gene expression across tissues, crucial for understanding cancer heterogeneity. Many ST analysis methods, however, do not fully utilize the rich morphological information in histopathology images and are limited by small sample sizes and high costs. We developed an innovative multimodal framework combining spatial transcriptomics with histopathological features using advanced image-aware deep-learning models, aimed at identifying significant patterns missed by traditional clustering and leveraging these models to impute gene expression across tissue slides for large-scale biomarker discovery and mechanistic studies.

### <h2 id="1.2"><font color=#00297D>1.2 Objectives</font></h2>

1. Predict **cell types**, **tumor microenvironment (TME)** characteristics, and **gene expressions** from histopathology images using supervised models.
2. Uncover spatial patterns using unsupervised clustering on morphological and transcriptomic features.
3. Application on both **Visium** and **Visium HD** spatial transcriptomics datas.

### <h2 id="1.3"><font color=#00297D>1.3 Data</font></h2>
![10xGenomics](https://github.com/Nina-Song/stViTDecoder/blob/main/results/Visium_data_and_Visium_HD_data.jpg)
1.	**Visium ST**: Spatial transcriptomics data with **55 µm** spots, each capturing the expression of approximately 33,000 genes. Each spot corresponds to a region of about **200 pixels** in the associated H&E image. In this study focusing on glioblastoma (GBM), we analyzed **4,895 spots** linked to 10x resolution H&E images (data currently not publicly available).
2.	**Visium HD**: High-density spatial transcriptomics data with **8 µm** or **16 µm** bins, each capturing the expression of approximately 33,000 genes. Each bin corresponds to a region of about **55 pixels** in the associated H&E image. In this study focusing on colorectal cancer, we analyzed **137,051 bins** linked to 20x resolution H&E images. The dataset is publicly available for download [[link](https://www.10xgenomics.com/datasets/visium-hd-cytassist-gene-expression-libraries-of-human-crc)])

## <h2 id="2"><font color=#00297D>2. File Structure</font></h2>

The repository is structured as follows:

```plaintext
stViTDecoder/
├── data/                         # Raw and processed data
├── preprocessing/                
├── results/                      
├── scripts/                      # Shell scripts to run pipelines
│   ├── visium_hd/ 
│   │   ├── run_cell_type_pred.sh 
│   ├── visium_st/ 
│   │   ├── run_cell_type_pred.sh 
│   │   ├── run_gene_level_pred.sh
├── src/                          # Source code for tasks
│   ├── supervised_eval/          # Supervised learning modules
│   │   ├── visium_hd/            # Visium HD-specific pipelines
│   │   │   ├── cell_type_embed.py # Generate embeddings from ViT for each bin & create labels based on cell types (binery)
│   │   │   ├── cell_type_embed_tertiles.py # three classes set
│   │   │   ├── kmeans_cluster_embedding.py # Generate embeddings from ViT for each bin & create labels based on kmeans
│   │   │   ├── main.py           # linear prob classification
│   │   │   └── utils/            # Helper functions for HD processing
│   │   ├── visium_st/            # Visium ST-specific pipelines
│   │   │   ├── cell_type_embed.py
│   │   │   ├── cell_type_embed_tertiles.py
│   │   │   ├── gene_level_embed.py
│   │   │   ├── cell_type_embed_tertiles.py
│   │   │   └── main.py           # linear prob classification
│   ├── unsupervised_clustering/  # Clustering algorithms on different resolution input H&E images
│   │   ├── gigapath_kmeans_10xRes.py  # kmeans clustering
│   │   ├── gigapath_kmeans_40xRes.py
│   │   ├── gigapath_DEC_10xRes.py
│   │   └── gigapath_DEC_40xRes.py     # Deep embedding clustering
├── LICENSE                       # License file
└── README.md                     # Project documentation
```

---

### **Main Files Descriptions**

#### **Supervised Evaluation**
- **`supervised_eval/visium_hd/`**
  - `cell_type_embed.py`: Generates cell type-based labels (e.g., fibroblast vs. epithelial for colon cancer) for HD data using marker gene set expressions. Labels are determined using Gaussian Mixture Models (GMM) for binary (on/off) or tertile (high/mid/low) classification. Additionally, creates embeddings for each HD bin using Vision Transformer (ViT).
  - `main.py`: Implements a linear probe framework using PyTorch Lightning for HD data analysis and classification tasks.
  - `utils/`: Contains utility functions for preprocessing and image data loading, specifically designed for HD datasets.
- **`supervised_eval/visium_st/`**
  - `gene_level_embed.py`: Identifies top Moran’s I genes (indicating spatial heterogeneity) and generates labels based on GMM for binary or tertile classification. Produces embeddings for each Visium spot using ViT for non-HD datasets.
  - `cell_type_embed.py`: Creates cell type-based (oligodendrocytes vs microglia for GBM) labels for non-HD Visium data using marker gene set expressions.

#### **Unsupervised Clustering**
- **`unsupervised_clustering/`**
  - `gigapath_kmeans_10xRes.py`: KMeans clustering for 10x resolution.
  - `gigapath_DEC_40xRes.py`: Deep embedding clustering for 40x resolution.

#### **Scripts**
- Shell scripts to run pipelines for various tasks (e.g., supervised predictions, generate acc/auroc..).

#### **Preprocessing**
- JupyterNotebooks for pre-processing visium spatial transcriptomics and histology images.

## <h2 id="3"><font color=#00297D>3. Tasks & Workflows</font></h2>

### <h2 id="3.1"><font color=#00297D>3.1 Spot-level Prediction</font></h2>

1. **Generate Spot-Level Image Embeddings Based on Cell Types**
    ```bash
   python src/supervised_eval/visium_st/cell_type_embed.py
    ```
Adjust the gene list directly in the script: 
```gene_sets = {"fibroblast": ['COL3A1', 'THY1'], "epithelial": ['EPCAM'] }```
* Labels can be created based on GMM (see 'Generate clusters based on GMM') or based on user selected fold change thershold (see 'Generate clusters based on fold change thershold' in the script)

2. **Run Prediction**
   ```bash
   bash scripts/visium_hd/run_cell_type_pred.sh
   ```

3. **Generate Top Moran'i Genes Spot-Level Image Embeddings**
   ```bash
   python src/supervised_eval/visium_st/gene_level_embed.py
   ```

---

### <h2 id="3.2"><font color=#00297D>3.2 Unsupervised Clustering</font></h2>

1. **10x Resolution Clustering**
   ```bash
   python src/unsupervised_clustering/gigapath_kmeans_10xRes.py /path/to/data
   python src/unsupervised_clustering/gigapath_DEC_10xRes.py /path/to/data
   ```

2. **40x Resolution Clustering**
   ```bash
   python src/unsupervised_clustering/gigapath_kmeans_40xRes.py
   python src/unsupervised_clustering/gigapath_DEC_40xRes.py
   ```

## <h2 id="4"><font color=#00297D>4. Results</font></h2>
### <h3 id="4.1"><font color=#00297D>4.1 Visium ST</font></h2>
### **Cell-type Prediction (oligodendrocytes vs microglia) from GBM**
* Test Accuracy: 0.922
* f1: 0.921
* AUROC: 0.976
### **Gene Prediction (top 10 Moran's I) from GBM**
<img src="https://github.com/COHCCC/GigaPathCDT/blob/main/images/predicted_spatial_enrichment_genes.png" alt="morani" width="150"/>

---
### <h3 id="4.2"><font color=#00297D>4.2 Visium HD</font></h2>
### **Cell-type Prediction (oligodendrocytes vs microglia) from GBM**
* Test Accuracy: 0.730
* f1: 0.692
* AUROC: 0.774
**Fibroblast**
<img src="https://github.com/Nina-Song/stViTDecoder/blob/main/data/fibroblast.png" alt="fibro" width="400"/>

**Epithelial**
<img src="https://github.com/Nina-Song/stViTDecoder/blob/main/data/epithelial.png" alt="epi" width="400"/>
## <h2 id="5"><font color=#00297D>5. Usage Guide</font></h2>

### **Prerequisites**
Install the required Python dependencies:
```bash
pip install -r requirements.txt
```
## <h2 id="6"><font color=#00297D>6. References</font></h2>

1. Xu, H., Usuyama, N., Bagga, J. et al. A whole-slide foundation model for digital pathology from real-world data. Nature 630, 181–188 (2024). 
2. 10x Genomics Visium HD: [Link](https://www.10xgenomics.com/datasets?menu%5Bproducts.name%5D=Spatial%20Gene%20Expression&query=HD&page=1&configure%5BhitsPerPage%5D=50&configure%5BmaxValuesPerFacet%5D=1000)


## <h2 id="7"><font color=#00297D>7. Acknowledgements</font></h2>

Dr. David Craig  
John Lee  
Nina Song  

### <div align="center">[Back to Top](#jump)</div>