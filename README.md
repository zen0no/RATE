# Recurrent Action Transformer with Memory (RATE) 
- [x] TMaze (done)
- [ ] VizDoom (coming soon)
- [ ] Memory Maze (TBD)

### Installing dependencies:
```bash
pip install -r requirements.txt
```

## TMaze 
### To reproduce experiments:
#### Train:
```bash
find . -type f -name "*\.sh" -exec chmod +x {} \;
TMaze_new/TMaze_new_src/run/train_tmaze_DT.sh
TMaze_new/TMaze_new_src/run/train_tmaze_RATE.sh
TMaze_new/TMaze_new_src/run/train_tmaze_GRATE.sh
```
#### Inference:
```bash
inf.sh
```

### To make individual run:
* `model_mode`: DT, RATE, DTXL or RATEM (RMT);
* `arch_mode`: TrXL, TrXL-I or GTrXL;
* `curr`: use of curriculum learning during training.
```bash
python3 TMaze_new/TMaze_new_src/train_tmaze.py --model_mode 'RATE' \
                                               --max_n_final 3 \
                                               --arch_mode 'TrXL' \
                                               --curr 'true' \
                                               --ckpt_folder 'RATE_max_3'

```



⠀⠀⠀⠀⠀⠀⠀⠀⠀⠠⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⠀⢀⣄⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠐⠀⣴⣤⡀⠀⢀⣀⣤⠤⠤⠶⠖⠒⠒⠒⠒⠒⠲⠶⠤⢤⣀⡀⣼⣛⣧⠀⢁⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣸⣏⢻⣍⠁⠀⢀⡀⠤⠄⠒⠒⠒⠒⠒⠒⠀⠤⠄⠀⠀⢸⡳⢾⢹⡀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⣠⠖⠋⠀⢯⡞⣎⡆⠁⠀⠀⠀⢀⡀⠀⠤⠤⠤⠤⠄⠀⡀⠀⠀⠻⣽⣻⡌⠹⣄⠀⠐⠀⠀⠀⠀
⠀⠀⠀⠀⠀⢀⡾⠁⠀⠀⢀⢾⣹⢿⣸⠀⣰⠎⠉⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠁⠆⠹⡿⣏⢆⠈⢷⡀⠀⠆⠀⠀
⠀⠀⠀⠀⣰⠏⠀⠀⢀⠔⠛⠄⠙⠫⠇⢀⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⢄⠠⠒⠒⠵⡈⢳⡀⠀⠀⠀
⠀⠄⠀⡰⠁⠀⠀⢠⠊⠄⠂⠁⠈⠁⠒⠊⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠐⡀⠀⠀⠀⠀⠈⠀⠀⠀⠀⠀⠄⢳⡀⠈⠀
⠈⠀⣸⠃⠀⠀⠀⠈⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠐⠀⠀⠐⠀⠀⠐⠀⢀⠀⠀⠀⠀⢷⠀⠀
⠀⢠⠇⠂⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠘⠀⠀⠀⠀⠀⠰⠀⠀⠀⠀⠀⠀⡄⠀⡀⠆⢰⠀⠀⠀⡄⠀⠀⠀⠸⡄⠀
⠀⣼⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡀⠉⠀⡄⠀⢀⠀⠀⡄⠂⠆⠀⠀⠀⠀⢁⠀⢁⠀⢸⠀⢇⠀⡇⠀⠀⠀⠀⣧⠀
⠀⡟⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠰⡃⠄⠈⡄⠀⡇⢀⢰⠀⠀⠀⠀⡼⠀⠸⢰⠀⣤⣅⣁⣴⠀⠀⠀⠀⢻⠀
⢠⡇⠀⠀⠀⠀⠀⠀⠀⠠⠀⠀⠀⠱⢀⣁⣤⣧⣴⣧⣄⡇⢸⣸⡄⠀⢀⣆⠀⣦⠊⢹⣿⣿⡛⠻⢿⠀⠀⠀⠀⢸⡇
⢸⡇⠀⠀⠀⠀⠀⠀⠀⠀⣃⠀⠀⢴⣿⠟⠉⢈⣿⣿⣿⡟⠇⠀⠀⠀⠀⠀⠀⢸⣶⣿⣿⡿⣧⠀⢸⡇⠀⢃⠀⢸⡇
⠈⡇⠀⠀⠀⠀⠀⠀⠀⡀⢉⡄⠀⢸⠁⠀⣷⣾⣿⣿⡟⣿⠀⠀⠀⠀⠀⠀⠀⠀⢧⠙⠋⢁⡟⢀⡦⢧⠀⠸⡇⢸⡇
⠀⣿⠀⠀⠀⠀⠀⠀⢀⠔⠪⡄⠀⠸⣁⠀⠹⣉⠉⠉⢠⠏⠀⠀⠀⠀⠀⠀⠀⠀⠈⠓⢲⠛⠆⢉⠀⢸⠀⢀⢇⢸⡇
⠀⢿⠀⠀⠀⠀⠀⢀⠃⡐⠐⣴⠀⠀⠏⠉⠖⠉⠋⡙⠁⠀⠀⠀⠀⠀⠀⢀⠀⠀⠀⢀⡠⠀⠊⠄⠌⢘⠀⠀⠸⢸⠀
⠀⢸⠀⠀⠀⠀⠀⠈⣆⢃⠘⠘⡀⠀⡸⡘⡐⡐⠠⠁⠀⡴⡖⣲⠒⠊⠉⠉⠉⠙⢿⣤⡇⠀⠀⠀⠈⢐⠀⠀⠁⣿⠀
⠀⠘⡇⠀⠀⠀⠀⠀⠈⢶⠬⣁⡇⠀⠀⠑⠐⠤⠐⠀⠀⡇⠉⠀⠀⠀⠀⠀⠀⠀⠀⢙⠇⠀⠀⠀⠀⣼⢀⠀⠀⣿⠀
⠀⠀⣇⠀⠀⠀⢰⠀⠀⠈⠀⠂⡇⠀⠃⢡⠀⠀⠀⠀⠀⠹⡄⠀⠀⠀⠀⠀⠀⠀⣠⠎⠀⠀⢀⡴⡞⡉⠈⠀⠀⣿⠀
⠀⠀⣹⠀⠀⠀⠀⠀⠀⠀⠀⡀⡇⠀⢰⠈⡷⡀⠀⠀⠀⠀⠸⢶⣀⠀⠀⢀⣰⠎⠁⢀⡶⠏⠁⣈⠆⠁⡀⠰⢸⡇⠀
⠀⠀⢸⡀⢸⠀⠀⠆⠀⠀⠀⠀⡇⠀⠀⠀⢡⡄⡏⢆⠒⠢⠤⠤⠤⢨⠥⡴⠒⠚⠉⠉⠀⠀⡠⠁⡘⢠⠁⢀⠆⡇⠀
⠀⠀⢸⡇⠀⡀⠀⠀⠀⠀⢠⢠⠁⠀⠘⡀⠠⣷⠃⠀⠀⠀⠀⠉⢰⠈⢱⠄⡀⡄⠀⢸⠀⠐⠀⠰⠁⠀⠀⡞⠄⣷⠀
⠀⠀⠀⣷⠀⡇⠀⠀⠀⠸⠀⡈⠀⠀⢂⠃⠀⡄⠇⠀⠀⠀⠀⠀⢔⠳⠀⠀⠣⠍⠒⠤⣰⠁⢠⠃⢠⠀⠀⠅⠀⢻⡀
⠀⠀⠀⠉⠀⠁⠀⠀⠀⠀⠀⠁⠀⠀⠈⠀⠀⠁⠈⠀⠀⠀⠀⠀⠀⠉⠁⠀⠀⠈⠁⠀⠈⠀⠁⠀⠈⠀⠀⠁⠀⠈⠁
