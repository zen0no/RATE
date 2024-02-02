#!/bin/bash

# DT + RATE
# cd TMaze_new/TMaze_new_checkpoints
# for dir in */; do
#     echo "${dir%/}"
#     cd
#     cd Name/REPOSITORIES/RATE
#     if [[ "$dir" == *_DT_* ]]; then
#         python3 TMaze_new/TMaze_new_src/inference_tmaze.py --model_mode 'DT' --max_n_final 9 --ckpt_name "${dir%/}" --ckpt_chooser 0
#     else
#         python3 TMaze_new/TMaze_new_src/inference_tmaze.py --model_mode 'RATE' --max_n_final 9 --ckpt_name "${dir%/}" --ckpt_chooser 0
#     fi
#     cd TMaze_new/TMaze_new_checkpoints
# done

cd TMaze_new/TMaze_new_checkpoints/old_version_december
for dir in */; do
    echo "${dir%/}"
    cd
    cd Name/REPOSITORIES/RATE
    
    if [[ "$dir" == *_RATE_min_1_max_3_* ]]; then
        python3 TMaze_new/TMaze_new_src/inference_tmaze.py --model_mode 'RATE' --max_n_final 9 --ckpt_name "${dir%/}" --ckpt_chooser 0
    fi

    cd TMaze_new/TMaze_new_checkpoints/old_version_december
done

# ~/Name/REPOSITORIES/RATE$ TMaze_new/TMaze_new_src/inference_tmaze.sh