set PYTHONPATH=/data/new_disk/pangbai/FaceFormer/FaceFormer/
# export 'PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512'
python train.py\
    --name fusion_full_lr_1e-7 \
    --gpu_id 1 \
    --dataroot ./data/GNPFA/ \
    --dataset NPFAFusionDataset \
    --train_subjects "044blendshape" \
    --lr 1e-7 \
    --feature_dim 64 \
    --vertice_dim 256 \
    --decoder_layer 1\
    --epoch_num 100000 \
    --max_len  600\
    --continue_train \
