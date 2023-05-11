set PYTHONPATH=..
python train.py\
    --name mead_lr1e-4_tf \
    --dataset NPFAVerticeDataset \
    --gpu_id 0 \
    --dataroot data/GNPFA_MEAD \
    --train_subjects TaylorSwift \
    --lr 1e-4 \
    --epoch_num 3000 \
    --continue_train \
    --teacher_forcing \
    