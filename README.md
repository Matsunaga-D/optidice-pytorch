# OptiDICE

<b> IN PROGRESS</b>
This repository is the PyTorch implementation of [OptiDICE: Offline Policy Optimization via Stationary Distribution Correction Estimation](https://arxiv.org/abs/2106.10783) (presented at ICML 2021).

## Installation Guide

### Environment Variables
- Insert the following commands in `~/.bashrc`.
    ```
    export MUJOCO_PY_MJKEY_PATH=$HOME/.mujoco/mjkey.txt
    export MUJOCO_PY_MJPRO_PATH=$HOME/.mujoco/mjpro150/
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mjpro150/bin
    ```

### Mujoco
1. Download MuJoCo. Save `mjkey.txt` to `$HOME/.mujoco` after the following commands:
     ``` 
     mkdir -p $HOME/.mujoco
     cd $HOME/.mujoco
     wget https://www.roboti.us/download/mjpro150_linux.zip
     unzip mjpro150_linux.zip
     rm mjpro150_linux.zip
     wget https://www.roboti.us/download/mujoco200_linux.zip
     unzip mujoco200_linux.zip
     rm mujoco200_linux.zip
     ```

### Conda Environment
1. Create conda environment and activate it:
     ```
     conda env create -f environment.yml
     conda activate optidice
     ```

2. Install `d4rl`:
     ```
     pip install git+https://github.com/rail-berkeley/d4rl@master#egg=d4rl
     ```

### How to Run
1. Random MDP experiments:
     ```
     python finite_run.py
     ```

2. D4RL Benchmarks
     ```
     python neural_dice_rl.py \
       --env_name=maze2d-umaze-v1 \
       --policy_extraction=iproj \
       --e_loss_type=mse \
       --alpha=0.001
     ```


### Bibtex

If you use this code, please cite our paper:
```
@inproceedings{lee2021optidice,
  author    = {Jongmin Lee and Wonseok Jeon and Byung-Jun Lee and Joelle Pineau and Kee-Eung Kim},
  title     = {OptiDICE: Offline Policy Optimization via Stationary Distribution Correction Estimation},
  booktitle = {Proceedings of the 38th International Conference on Machine Learning (ICML)},
  year      = {2021}
}
```

