set PYTHONPATH=/data/new_disk/pangbai/FaceFormer/FaceFormer/
python train.py\
    --name debug_4_lr_1e-5 \
    --gpu_id 3 \
    --dataroot ./data/exp_code/ \
    --dataset NPFAEmbeddingDataset \
    --train_subjects embedding \
    --valid_subjects embedding \
    --lr 1e-5 \
    --vertice_dim 256 \
    --phase debug \
    --epoch_num 1000 \
