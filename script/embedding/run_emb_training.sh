set PYTHONPATH=/data/new_disk/pangbai/FaceFormer/FaceFormer/
python train.py\
    --name emb_full_lr_1e-5 \
    --gpu_id 3 \
    --dataroot ./data/exp_code/ \
    --dataset NPFAEmbeddingDataset \
    --train_subjects embedding \
    --valid_subjects embedding \
    --lr 5e-7 \
    --vertice_dim 256 \
    --epoch_num 3000 \
    --continue_train \
