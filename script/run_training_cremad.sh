set PYTHONPATH=..
python train.py\
    --name creamad_lr1e-4 \
    --dataset NPFAVerticeDataset \
    --gpu_id 3 \
    --dataroot data/GNPFA_chn \
    --train_subjects TaylorSwift \
    --lr 1e-4 \
    --epoch_num 3000 \
    --continue_train \
    --teacher_forcing \
