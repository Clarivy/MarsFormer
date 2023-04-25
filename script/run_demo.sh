export PYOPENGL_PLATFORM=osmesa
export MUJOCO_GL=osmesa
python demo.py \
--model_name vocaset \
--wav_path demo/wav/en-ch_bgm.wav \
--dataset vocaset \
--vertice_dim 42186 \
--feature_dim 64 \
--period 30 \
--fps 30 \
--train_subjects "014_blendshape 044blendshape 045blendshape 046blendshape 047blendshape 064blendshape" \
--condition "014_blendshape" \
--model_template data/USC_neutral.obj \
--model_path checkpoints/scaled_nomask_lr_1e-4/latest_model.pth \
--render_template_path data/USC_neutral.obj \
--max_len 400 \
