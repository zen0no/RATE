# RATE
We propose the Recurrent Action Transformer with Memory (RATE) – a model that incorporates recurrent memory. To evaluate our model, we conducted extensive experiments on
both memory-intensive environments (VizDoom-Two-Color, T-Maze) and classic Atari games and MuJoco control environments. The results show that the use of memory can significantly improve performance in memory-intensive environments while maintaining or improving results in classic environments. We hope that our findings will
stimulate research on memory mechanisms for transformers applicable to offline reinforcement learning.

Out code based on [this RMT repository](https://github.com/booydar/LM-RMT). The frame of code for working with Atari and Mujoco based on [official DT repository](https://github.com/kzl/decision-transformer).

This repository remains a work in progress, as our paper is currently undergoing the review process. We intend to provide additional details regarding citation and accompanying figures illustrating the model architecture soon.

To reproduce the experiments on the TMaze and VizDoom-Two-Colors environments, navigate to the `/Vizdoom` branch
