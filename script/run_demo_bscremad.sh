export PYOPENGL_PLATFORM=osmesa
export MUJOCO_GL=osmesa
conda activate Faceformer_old

audios=("test_sentence" "en_female" "maggie5")

function run_demo {
    python demo_bs.py \
        --model_name bscreamad_tf_lr1e-4_vert \
        --model_step latest \
        --model_path checkpoints/ \
        --wav_path /data/new_disk/new_disk/pangbai/FaceFormer/FaceFormer/demo/wav/$1.wav \
        --dataset vocaset \
        --vertice_dim 55 \
        --feature_dim 64 \
        --period 30 \
        --fps 30 \
        --train_subjects "000_generic_neutral_mesh_usc" \
        --condition "000_generic_neutral_mesh_usc" \
        --base_models /data/new_disk/new_disk/pangbai/FaceFormer/FaceFormer/data/USC56 \
        --render_template_path data/GNPFA_wild/identity/000_generic_neutral_mesh_usc.obj \
        --max_len 400
}

# Define your list of arguments

# Iterate through the arguments and call the function with each
for arg in "${audios[@]}"
do
    run_demo "$arg"
    if [ $? -ne 0 ]; then
        conda deactivate
        return 1
    fi
done


# Female Audio: vocaset/wav/FaceTalk_170731_00024_TA_sentence01.wav
# Male Audio: vocaset/wav/FaceTalk_170725_00137_TA_sentence01.wav

conda deactivate