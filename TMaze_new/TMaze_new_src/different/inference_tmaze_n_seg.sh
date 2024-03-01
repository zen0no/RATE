#!/bin/bash

while getopts ":c:m:a:s:n:" opt; do
  case $opt in
    c)
      ckpt_folder="$OPTARG"
      ;;
    m)
      mode="$OPTARG"
      ;;
    a)
      arch_mode="$OPTARG"
      ;;
    s)
      max_segments="$OPTARG"
      ;;
    n)
      num_of_inf_segments="$OPTARG"
      ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      exit 1
      ;;
    :)
      echo "Option -$OPTARG requires an argument." >&2
      exit 1
      ;;
  esac
done
 
#ckpt_folder="$1"
ckpt_dir="TMaze_new/TMaze_new_checkpoints/${ckpt_folder##*/}"
cd "$ckpt_dir"
# # echo $(pwd)
echo "Selected mode: ${mode%/}"

for dir in */; do
    echo Current directory: "${dir%/}"
    cd $OLDPWD
    
    if [[ "$dir" == *_"${mode%/}"_* && "$dir" == *_"${max_segments%/}"_* && "$dir" == *_"${arch_mode%/}"_* ]]; then

        string="${dir%/}"
        string_no_data="${string%_*_*_*_*_*_*_*}"
        new_string=""
        for el in $string_no_data; do
            new_string+="$el"_
        done
        new_string="${new_string%_}"
        new_string_no_run="${new_string%_*}"

        python3 TMaze_new/TMaze_new_src/inference/inference_tmaze_n_seg.py --model_mode "${mode%/}" \
                                                                           --max_n_final 12 \
                                                                           --ckpt_name "${dir%/}" \
                                                                           --ckpt_chooser 0 \
                                                                           --ckpt_folder "${ckpt_folder%/}" \
                                                                           --arch_mode "${arch_mode%/}" \
                                                                           --inf_segments ${num_of_inf_segments}
    fi

    cd "$ckpt_dir"
done

# ~/Name/REPOSITORIES/RATE$ TMaze_new/TMaze_new_src/inference_tmaze.sh
# TMaze_new/TMaze_new_src/different/inference_tmaze_n_seg.sh -c 'DT_RATE_GRATE' -m 'RATE' -a 'TrXL' -s 'max_1' -n 20