#!/bin/bash
# succes 10/10
#python3 TMaze_new/TMaze_new_src/train_rmdt_new.py --model_mode 'RMDT' --max_n_final 3
# error 5/10

# ! Classical curriculum training
# python3 TMaze_new/TMaze_new_src/train_rmdt_new.py --model_mode 'RMDT' --max_n_final 3
# python3 TMaze_new/TMaze_new_src/train_rmdt_new.py --model_mode 'RMDT' --max_n_final 5 --start_seg 6 --end_seg 10
# python3 TMaze_new/TMaze_new_src/train_rmdt_new.py --model_mode 'RMDT' --max_n_final 7
# python3 TMaze_new/TMaze_new_src/train_rmdt_new.py --model_mode 'RMDT' --max_n_final 9

# python3 TMaze_new/TMaze_new_src/train_rmdt_new.py --model_mode 'DT' --max_n_final 3
# python3 TMaze_new/TMaze_new_src/train_rmdt_new.py --model_mode 'DT' --max_n_final 5
# python3 TMaze_new/TMaze_new_src/train_rmdt_new.py --model_mode 'DT' --max_n_final 7
# python3 TMaze_new/TMaze_new_src/train_rmdt_new.py --model_mode 'DT' --max_n_final 9

# ! Train RMDT w/o curriculum
#  python3 TMaze_new/TMaze_new_src/train_rmdt_new.py --model_mode 'RMDT' --max_n_final 1
# #  python3 TMaze_new/TMaze_new_src/train_rmdt_new.py --model_mode 'RMDT' --max_n_final 2
#  python3 TMaze_new/TMaze_new_src/train_rmdt_new.py --model_mode 'RMDT' --max_n_final 3
# #  python3 TMaze_new/TMaze_new_src/train_rmdt_new.py --model_mode 'RMDT' --max_n_final 4
#  python3 TMaze_new/TMaze_new_src/train_rmdt_new.py --model_mode 'RMDT' --max_n_final 5
# #  python3 TMaze_new/TMaze_new_src/train_rmdt_new.py --model_mode 'RMDT' --max_n_final 6
#  python3 TMaze_new/TMaze_new_src/train_rmdt_new.py --model_mode 'RMDT' --max_n_final 7
# #  python3 TMaze_new/TMaze_new_src/train_rmdt_new.py --model_mode 'RMDT' --max_n_final 8
#  python3 TMaze_new/TMaze_new_src/train_rmdt_new.py --model_mode 'RMDT' --max_n_final 9

# ! Train DT with fixed training loop
# python3 TMaze_new/TMaze_new_src/train_rmdt_new.py --model_mode 'DT' --max_n_final 1
# python3 TMaze_new/TMaze_new_src/train_rmdt_new.py --model_mode 'DT' --max_n_final 3
# python3 TMaze_new/TMaze_new_src/train_rmdt_new.py --model_mode 'DT' --max_n_final 5
# python3 TMaze_new/TMaze_new_src/train_rmdt_new.py --model_mode 'DT' --max_n_final 7
python3 TMaze_new/TMaze_new_src/train_rmdt_new.py --model_mode 'DT' --max_n_final 9



# chmod +x TMaze_new/TMaze_new_src/train_tmaze.sh 
# TMaze_new/TMaze_new_src/train_tmaze.sh


#~/Egor_C/REPOSITORIES/RMDT$ python3 TMaze_new/TMaze_new_src/inference_tmaze.py --model_mode 'DT' --max_n_final 9 --ckpt_name 'loss_all_inf_on_9_segments_DT_min_1_max_9_RUN_10_2024_01_03_04_45_02' 
#--ckpt_chooser 0 