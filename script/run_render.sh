export PYOPENGL_PLATFORM=osmesa
export MUJOCO_GL=osmesa
python render_npy.py \
    --output_file ./demo/output/pca_norm_ml500_vert_vel_datafull_000130000_test_sentence.mp4 \
    --wav_path /data/new_disk/new_disk/pangbai/FaceFormer/FaceFormer/demo/wav/test_sentence.wav \
    --npy_path /data/new_disk/new_disk/pangbai/FaceFormer/motion-diffusion-model/results/pca_norm_ml500_vert_vel_datafull_000130000_test_sentence_vert.npy \
    --render_template_path data/GNPFA_wild/identity/TaylorSwift.obj
