## 
export PYOPENGL_PLATFORM=osmesa
export MUJOCO_GL=osmesa
python demo.py \
--model_name vocaset \
--wav_path demo/wav/maggie_clip.wav \
--output_file ./demo/output/maggie_clip_all.mp4 \
--dataset vocaset \
--vertice_dim 42186 \
--feature_dim 64 \
--period 30 \
--fps 30 \
--train_subjects "TaylorSwift" \
--condition "TaylorSwift" \
--model_template data/GNPFA_wild/identity/TaylorSwift.obj \
--model_path checkpoints/mix_lr1e-4_tf/latest_model.pth \
--render_template_path data/GNPFA_wild/identity/d2_OurUV.obj \
--max_len 400 \

# Female Audio: vocaset/wav/FaceTalk_170731_00024_TA_sentence01.wav
# Male Audio: vocaset/wav/FaceTalk_170725_00137_TA_sentence01.wav