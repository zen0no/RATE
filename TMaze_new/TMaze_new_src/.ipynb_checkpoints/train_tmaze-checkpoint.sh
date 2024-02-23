#!/bin/bash
# succes 10/10
#python3 TMaze_new/TMaze_new_src/train_RATE_new.py --model_mode 'RATE' --max_n_final 3
# error 5/10
# python3 TMaze_new/TMaze_new_src/train_RATE_new.py --model_mode 'RATE' --max_n_final 5 --start_seg 6 --end_seg 10
# python3 TMaze_new/TMaze_new_src/train_RATE_new.py --model_mode 'RATE' --max_n_final 7
# python3 TMaze_new/TMaze_new_src/train_RATE_new.py --model_mode 'RATE' --max_n_final 9

# python3 TMaze_new/TMaze_new_src/train_RATE_new.py --model_mode 'DT' --max_n_final 3
python3 TMaze_new/TMaze_new_src/train_RATE_new.py --model_mode 'DT' --max_n_final 5 --start_seg 8 --end_seg 10
python3 TMaze_new/TMaze_new_src/train_RATE_new.py --model_mode 'DT' --max_n_final 7
python3 TMaze_new/TMaze_new_src/train_RATE_new.py --model_mode 'DT' --max_n_final 9




# chmod +x TMaze_new/TMaze_new_src/train_tmaze.sh 
# TMaze_new/TMaze_new_src/train_tmaze.sh
