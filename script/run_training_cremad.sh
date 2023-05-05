set PYTHONPATH=..
python train.py\
    --name creamad_NEU_lr1e-4 \
    --dataset NPFAVerticeDataset \
    --gpu_id 2 \
    --dataroot data/GNPFA_CREMA-D \
    --train_subjects TaylorSwift \
    --lr 1e-4 \
    --epoch_num 3000 \
    --continue_train \
