set PYTHONPATH=..
python valid.py\
    --name emb_full_lr_1e-5_kl \
    --gpu_id 0 \
    --dataroot ./data/exp_code/ \
    --dataset NPFAEmbeddingDataset \
    --train_subjects embedding \
    --valid_subjects embedding \
    --no_obj \
    --feature_dim 64 \
    --vertice_dim 256 \
    --decoder_layer 1\
    --max_len  600\
