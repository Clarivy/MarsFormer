set PYTHONPATH=..
python demo.py \
--model_name vocaset \
--wav_path "demo/wav/trim2.wav" \
--dataset vocaset \
--vertice_dim 11793 \
--feature_dim 64 \
--period 30 \
--fps 30 \
--base_model_path ./data/FLAME \
--base_template ./data/000_generic_neutral_mesh.obj \
--condition FaceTalk_170913_03279_TA \
--subject FaceTalk_170809_00138_TA \
--model_path ./vocaset/save/15_model.pth