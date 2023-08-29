set PYTHONPATH=..
python train.py\
    --name mead_neu_only \
    --dataset MixDataset \
    --gpu_id 0 \
    --dataroot data/GNPFA_mix \
    --train_subjects 000_generic_neutral_mesh_usc \
    --lr 1e-4 \
    --vertice_dim 28224 \
    --epoch_num 3000 \
    --mix_config data/config/mix_config_mead_neu.json \
    --teacher_forcing \
    # --continue_train \
    