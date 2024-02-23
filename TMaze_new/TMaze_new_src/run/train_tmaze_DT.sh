#!/bin/bash

python3 TMaze_new/TMaze_new_src/train_rate.py \
        --model_mode 'DT' \
        --max_n_final 1 \
        --end_seed 10 \
        --arch_mode 'TrXL' \
        --curr 'false'

python3 TMaze_new/TMaze_new_src/train_rate.py \
        --model_mode 'DT' \
        --max_n_final 3 \
        --end_seed 10 \
        --arch_mode 'TrXL' \
        --curr 'false'

python3 TMaze_new/TMaze_new_src/train_rate.py \
        --model_mode 'DT' \
        --max_n_final 5 \
        --end_seed 10 \
        --arch_mode 'TrXL' \
        --curr 'false'