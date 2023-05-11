set PYTHONPATH=..
python train.py\
    --name ravdess_lr1e-4 \
    --dataset NPFARavdessDataset \
    --gpu_id 0 \
    --dataroot data/GNPFA_RAVDESS \
    --train_subjects TaylorSwift \
    --lr 1e-4 \
    --epoch_num 3000 \
