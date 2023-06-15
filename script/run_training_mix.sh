set PYTHONPATH=..
python train.py\
    --name mix_neu_lr1e-4_tf_normalized_cremad_only \
    --dataset MixDataset \
    --gpu_id 1 \
    --dataroot data/GNPFA_mix \
    --train_subjects TaylorSwift \
    --lr 1e-4 \
    --epoch_num 3000 \
    --mix_config data/config/mix_config_neu_cremad.json \
    --teacher_forcing \
    # --continue_train \
    