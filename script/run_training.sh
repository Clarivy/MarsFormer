set PYTHONPATH=..
python main.py \
--dataset vocaset \
--vertice_dim 11793 \
--feature_dim 64 \
--base_model_path ./data/FLAME \
--base_template ./data/000_generic_neutral_mesh.obj \
--name performer_test_fast_attention_new_perself_nnmul \
--max_epoch 200 \
--neg_penalty 0 
#--load_model ./checkpoints/save/performer_test_fast_attention_neg0_causal_4_perself_nnmul/200_model.pth