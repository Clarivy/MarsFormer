set -e
source /root/anaconda3/etc/profile.d/conda.sh 
conda activate faceformer
export PYOPENGL_PLATFORM=osmesa
export MUJOCO_GL=osmesa
set PYTHONPATH=/data/new_disk/pangbai/FaceFormer/FaceFormer
python /data/new_disk/pangbai/FaceFormer/FaceFormer/demo.py \
--model_name vocaset \
--wav_path $FACEFORMER_WAV_PATH \
--dataset vocaset \
--vertice_dim 11793 \
--feature_dim 64 \
--period 30 \
--fps 30 \
--base_model_path ./data/FLAME \
--base_template ./data/000_generic_neutral_mesh.obj \
--model_template $FACEFORMER_TEMPLATE_PATH \
--condition FaceTalk_170913_03279_TA \
--model_path $FACEFORMER_MODEL_PATH \
--result_path $FACEFORMER_RESULT_PATH \
--no_render