python linear_probe.py \
    --metadata_csv /coh_labs/dits/nsong/Mayo_VisiumHD/789/metadata_3_types.csv \
    --embeddings_dir /coh_labs/dits/nsong/Mayo_VisiumHD/789/features_3_types \
    --output_dir ./logs \
    --input_dim 1536 \
    --num_classes 3 \
    --batch_size 128 \
    --max_epochs 10


# #!/bin/bash

# # Ensure the script stops on errors
# set -e

# # Define the parameters
# METADATA_CSV="/coh_labs/dits/nsong/Mayo_VisiumHD/789/metadata_3_types.csv"
# EMBEDDINGS_DIR="/coh_labs/dits/nsong/Mayo_VisiumHD/789/features_3_types"
# OUTPUT_DIR="./logs"
# INPUT_DIM=1536
# NUM_CLASSES=3
# BATCH_SIZE=128
# MAX_EPOCHS=20

# # Print the configurations for clarity
# echo "=== Running Linear Probe ==="
# echo "Metadata CSV: ${METADATA_CSV}"
# echo "Embeddings Dir: ${EMBEDDINGS_DIR}"
# echo "Output Dir: ${OUTPUT_DIR}"
# echo "Input Dim: ${INPUT_DIM}"
# echo "Num Classes: ${NUM_CLASSES}"
# echo "Batch Size: ${BATCH_SIZE}"
# echo "Max Epochs: ${MAX_EPOCHS}"

# # Activate the environment (if needed)
# # source activate my_environment

# # Run the linear probe script with parameters
# python linear_probe.py \
#     --metadata_csv "${METADATA_CSV}" \
#     --embeddings_dir "${EMBEDDINGS_DIR}" \
#     --output_dir "${OUTPUT_DIR}" \
#     --input_dim "${INPUT_DIM}" \
#     --num_classes "${NUM_CLASSES}" \
#     --batch_size "${BATCH_SIZE}" \
#     --max_epochs "${MAX_EPOCHS}" > output.log 2>&1 &

# # Notify the user where to monitor logs
# echo "Training started. Logs are being saved to 'output.log'."
# echo "Use 'tail -f output.log' to monitor progress in real-time."