set PYTHONPATH=/data/new_disk/pangbai/FaceFormer/FaceFormer/
python train.py\
    --name uscvoca_lr_1e-4_tf \
    --gpu_id 0 \
    --dataroot ./vocaset/ \
    --dataset USCVocaDataset \
    --train_subjects "FaceTalk_170728_03272_TA FaceTalk_170904_00128_TA FaceTalk_170725_00137_TA FaceTalk_170915_00223_TA FaceTalk_170811_03274_TA FaceTalk_170913_03279_TA FaceTalk_170904_03276_TA FaceTalk_170912_03278_TA FaceTalk_170811_03275_TA FaceTalk_170908_03277_TA" \
    --lr 1e-4 \
    --vertice_dim 28224 \
    --epoch_num 1000 \
    --max_len 600 \
    --teacher_forcing \
