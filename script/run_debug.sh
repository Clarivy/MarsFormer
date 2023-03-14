set PYTHONPATH=..
python train.py\
    --name debug_1 \
    --gpu_id 0 \
    --dataroot ./data/GNPFA/ \
    --tf_log \
    --phase debug \
