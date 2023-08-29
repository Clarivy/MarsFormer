## 
export PYOPENGL_PLATFORM=osmesa
export MUJOCO_GL=osmesa
python demo.py \
    --model_name vocaset \
    --wav_path demo/wav/test_sentence.wav \
    --output_file demo/output/mead_neu_only_test_sentence.mp4 \
    --dataset vocaset \
    --vertice_dim 28224 \
    --feature_dim 64 \
    --period 30 \
    --fps 30 \
    --train_subjects "000_generic_neutral_mesh_usc" \
    --condition "000_generic_neutral_mesh_usc" \
    --model_template data/GNPFA_wild/identity/000_generic_neutral_mesh_usc.obj \
    --model_path checkpoints/mead_neu_only/latest_model.pth \
    --render_template_path data/GNPFA_wild/identity/000_generic_neutral_mesh_usc.obj \
    --max_len 400 \
    # --no_render \

# Female Audio: vocaset/wav/FaceTalk_170731_00024_TA_sentence01.wav
# Male Audio: vocaset/wav/FaceTalk_170725_00137_TA_sentence01.wav