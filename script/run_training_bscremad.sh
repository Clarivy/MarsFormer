set PYTHONPATH=..
python train.py\
    --name bscreamad_tf_lr1e-4_vert \
    --dataset NPFABSDataset \
    --vertice_dim 55 \
    --gpu_id 3 \
    --dataroot data/GNPFA_CREMA-D \
    --train_subjects "000_generic_neutral_mesh_usc" \
    --base_models_path "/data/new_disk/new_disk/pangbai/FaceFormer/FaceFormer/data/USC56" \
    --lr 1e-4 \
    --epoch_num 3000 \
    --teacher_forcing \
    # --continue_train \