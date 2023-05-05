set PYTHONPATH=..
python train.py\
    --name deca_wild_lr1e-4 \
    --dataset DecaDataset \
    --gpu_id 3 \
    --dataroot data/DECA_wild \
    --lr 1e-4 \
    --epoch_num 3000 \
    --vertice_dim 15069 \
    --train_subjects nostyle \
    --valid_subjects nostyle \
    --continue_train \
