set PYTHONPATH=..
python train.py\
    --name creamad_expcode_lr1e-4 \
    --dataset ExpcodeCremadDataset \
    --vertice_dim 256 \
    --gpu_id 2 \
    --dataroot data/Expcode_CREMA-D \
    --train_subjects TaylorSwift \
    --lr 1e-4 \
    --epoch_num 3000 \
    --teacher_forcing \
    # --continue_train \