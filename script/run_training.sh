set PYTHONPATH=..
python train.py\
    --name scaled_nomask_lr_1e-4 \
    --dataset NPFAVerticeDataset \
    --gpu_id 2 \
    --dataroot ./data/GNPFA/ \
    --lr 1e-4 \
    --epoch_num 3000 \
    --continue_train \
