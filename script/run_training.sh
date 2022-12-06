set PYTHONPATH=..
python main.py \
--dataset vocaset \
--vertice_dim 11793 \
--feature_dim 64 \
--base_model_path ./data/FLAME \
--base_template ./data/000_generic_neutral_mesh.obj \
--name performer_test_fast_attention_causal \
--max_epoch 200