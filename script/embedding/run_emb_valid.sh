set PYTHONPATH=..
python valid.py\
    --name emb_full_lr_1e-5 \
    --gpu_id 0 \
    --dataroot ./data/exp_code/ \
    --dataset NPFAEmbeddingDataset \
    --train_subjects embedding \
    --valid_subjects embedding \
    --no_obj \
    --vertice_dim 256 \
