# set PYTHONPATH=/data/new_disk/pangbai/FaceFormer/FaceFormer/
# python valid.py\
#     --name voca_shuffle_lr_1e-4 \
#     --gpu_id 0 \
#     --dataroot ./vocaset/ \
#     --dataset VocaDataset \
#     --train_subjects "FaceTalk_170728_03272_TA FaceTalk_170904_00128_TA FaceTalk_170725_00137_TA FaceTalk_170915_00223_TA FaceTalk_170811_03274_TA FaceTalk_170913_03279_TA FaceTalk_170904_03276_TA FaceTalk_170912_03278_TA FaceTalk_170811_03275_TA FaceTalk_170908_03277_TA" \
#     --valid_subjects "FaceTalk_170809_00138_TA FaceTalk_170731_00024_TA" \
#     --vertice_dim 15069 \
#     --template_path vocaset/templates/FLAME_sample.ply \
#     --max_len 600 \
#     --how_many 1 \
set PYTHONPATH=/data/new_disk/pangbai/FaceFormer/FaceFormer/
python valid.py\
    --name voca_shuffle_lr_1e-4 \
    --gpu_id 3 \
    --dataroot ./vocaset/ \
    --dataset VocaNoStyleDataset \
    --train_subjects "FaceTalk_170728_03272_TA FaceTalk_170904_00128_TA FaceTalk_170725_00137_TA FaceTalk_170915_00223_TA FaceTalk_170811_03274_TA FaceTalk_170913_03279_TA FaceTalk_170904_03276_TA FaceTalk_170912_03278_TA FaceTalk_170811_03275_TA FaceTalk_170908_03277_TA" \
    --test_subjects "FaceTalk_170811_03275_TA FaceTalk_170908_03277_TA" \
    --vertice_dim 15069 \
    --max_len 600 \
    --how_many 1 \
    --template_path vocaset/templates/FLAME_sample.ply \
