MODEL_NAME=scaled_nomask_lr_1e-4
WHICH_EPOCH=latest
set PYTHONPATH=..
python valid.py\
    --name $MODEL_NAME \
    --gpu_id 0 \
    --dataroot ./data/GNPFA/ \
    --how_many 2

tar -cvf "${MODEL_NAME}_${WHICH_EPOCH}.tar" "data/results/${MODEL_NAME}_${WHICH_EPOCH}"