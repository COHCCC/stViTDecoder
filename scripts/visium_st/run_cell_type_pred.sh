INPUTPATH=${1-:/mnt/nsong-data-disk/Craig_Spatial/Craig_SPA8_A/outs/GigaPath_oligo_microglia_hypoxia_embeddings.zip}
DATASETCSV=/mnt/nsong-data-disk/Craig_Spatial/Craig_SPA8_A/outs/oligo_microglia_hypoxia_meta.csv
OUTPUT=outputs/nsong-data-disk/Craig_SPA8_A/glia_oligo_hypoxia/

python linear_probe/main.py --input_path $INPUTPATH \
                --dataset_csv $DATASETCSV \
                --output $OUTPUT \
                --batch_size 128 \
                --lr 0.01 \
                --min_lr 0.0 \
                --train_iters 4000 \
                --eval_interval 100 \
                --optim adam \
                --weight_decay 0.001 \