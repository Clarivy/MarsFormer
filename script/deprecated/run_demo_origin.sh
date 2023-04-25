export PYOPENGL_PLATFORM=osmesa
export MUJOCO_GL=osmesa
python demo.py \
--model_name vocaset \
--wav_path demo/wav/en-ch_sln.wav \
--dataset vocaset \
--vertice_dim 15069 \
--feature_dim 64 \
--period 30 \
--fps 30 \
--train_subjects "FaceTalk_170728_03272_TA FaceTalk_170904_00128_TA FaceTalk_170725_00137_TA FaceTalk_170915_00223_TA FaceTalk_170811_03274_TA FaceTalk_170913_03279_TA FaceTalk_170904_03276_TA FaceTalk_170912_03278_TA FaceTalk_170811_03275_TA FaceTalk_170908_03277_TA" \
--test_subjects "FaceTalk_170811_03275_TA FaceTalk_170908_03277_TA" \
--condition FaceTalk_170913_03279_TA \
--subject FaceTalk_170809_00138_TA \
--model_path checkpoints/voca_shuffle_lr_1e-4/310_model.pth \
--max_len 600 \
