#!/bin/bash
# Define the base path for the gene folders
GENE_DIR_BASE="/mnt/nsong-data-disk/Craig_Spatial/Craig_SPA3_D/outs/Spatial_enrichment_tertiles_classifier"
OUTPUT_BASE="outputs/Craig_SPA3_D/spatial_enrichment_tertiles"

# Create the base output directory if it doesn't exist
mkdir -p $OUTPUT_BASE

# Loop through each gene folder in GENE_DIR_BASE
for GENE_DIR in $GENE_DIR_BASE/*; do
    # Check if it's a directory
    if [ -d "$GENE_DIR" ]; then
        echo "1"
        # Get the gene name
        GENE_NAME=$(basename "$GENE_DIR")
        echo "Processing gene: $GENE_NAME"

        # Set the input path to the embeddings zip file for this gene
        INPUTPATH="$GENE_DIR/${GENE_NAME}_embeddings.zip"

        # Set the dataset CSV path
        DATASETCSV="$GENE_DIR/meta_tertiles.csv"

        # Set the output directory for this gene
        OUTPUT="$OUTPUT_BASE/$GENE_NAME"

        # Create the output directory if it doesn't exist
        mkdir -p $OUTPUT

        # Run the Python training script
        python linear_probe/main.py --input_path $INPUTPATH \
                                    --dataset_csv $DATASETCSV \
                                    --output $OUTPUT \
                                    --batch_size 128 \
                                    --lr 0.02 \
                                    --min_lr 0.0 \
                                    --train_iters 4000 \
                                    --eval_interval 100 \
                                    --optim sgd \
                                    --weight_decay 0.01

        echo "Completed processing for gene: $GENE_NAME"
    fi
done
