set PYTHONPATH=..
python train.py\
    --name mixvoca_neu_lr1e-4_tf_cremad_neu+voca \
    --dataset MixVocaDataset \
    --gpu_id 1 \
    --dataroot data/GNPFA_mix \
    --vertice_dim 27687 \
    --mix_dataroot data/GNPFA_mix \
    --mix_train_subjects TaylorSwift \
    --voca_dataroot ./vocaset/ \
    --voca_train_subjects "FaceTalk_170728_03272_TA FaceTalk_170904_00128_TA FaceTalk_170725_00137_TA FaceTalk_170915_00223_TA FaceTalk_170811_03274_TA FaceTalk_170913_03279_TA FaceTalk_170904_03276_TA FaceTalk_170912_03278_TA FaceTalk_170811_03275_TA FaceTalk_170908_03277_TA" \
    --train_subjects TaylorSwift \
    --lr 1e-4 \
    --epoch_num 3000 \
    --mix_config data/config/mix_config_neu_cremad.json \
    --teacher_forcing \
    # --continue_train \
    