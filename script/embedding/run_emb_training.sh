set PYTHONPATH=/data/new_disk/pangbai/FaceFormer/FaceFormer/
python train.py\
    --name emb_full_lr_1e-7_kl_chnmale1 \
    --gpu_id 2 \
    --dataroot ./data/exp_code/ \
    --dataset NPFAEmbeddingDataset \
    --train_subjects embedding \
    --valid_subjects embedding \
    --lr 1e-7 \
    --feature_dim 64 \
    --vertice_dim 256 \
    --decoder_layer 1\
    --epoch_num 100000 \
    --max_len  600\
    --continue_train \
