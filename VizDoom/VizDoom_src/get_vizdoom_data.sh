#!/bin/bash

file="VizDoom/VizDoom_data/train_VizDoom_Two_Colors_Column_disappear_delay_45_no_walls_agent_p1_01.pickle"
url="https://sc.link/ppwRI"

if [ ! -f "$file" ]; then
    echo "$file does not exist. Downloading..."
    wget "$url" -O "$file"
else
    echo "$file already exists."
fi
