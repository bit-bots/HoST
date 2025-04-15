# HoST


[![arXiv](https://img.shields.io/badge/arXiv-2502.08378-brown)](https://arxiv.org/abs/2502.08378)
[![](https://img.shields.io/badge/Website-%F0%9F%9A%80-yellow)](https://taohuang13.github.io/humanoid-standingup.github.io/)
[![](https://img.shields.io/badge/Youtube-🎬-red)](https://www.youtube.com/watch?v=Yruh-3CFwE4)
[![](https://img.shields.io/badge/bilibili-📹-blue)](https://www.bilibili.com/video/BV1o2KPeUEob/?spm_id_from=333.337.search-card.all.click&vd_source=ef6a9a20816968cc19099a3f662afd86)

This is the official PyTorch implementation of the paper "[**Learning Humanoid Standing-up Control across Diverse Postures**]()" by 

[Tao Huang](https://taohuang13.github.io/), [Junli Ren](https://renjunli99.github.io/), [Huayi Wang](https://why618188.github.io/), [Zirui Wang](https://scholar.google.com/citations?user=Vc3DCUIAAAAJ&hl=zh-TW), [Qingwei Ben](https://www.qingweiben.com/), [Muning Wen]((https://www.qingweiben.com/)), [Xiao Chen](https://xiao-chen.tech/), [Jianan Li](https://github.com/OpenRobotLab/HoST), [Jiangmiao Pang](https://oceanpang.github.io/)

<p align="left">
  <img width="75%" src="docs/teaser.png">
</p>

# 🛠️ Installation Instructions
Clone this repository:
```bash
git clone https://github.com/OpenRobotLab/HoST.git
cd HoST
```
Create a conda environment:
```bash
conda env create -f conda_env.yml 
conda activate host
```
Install pytorch 1.10 with cuda-11.3:
```bash
pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
```

Download and install [Isaac Gym](https://developer.nvidia.com/isaac-gym):
```bash
cd isaacgym/python && pip install -e .
```
Install rsl_rl (PPO implementation) and legged gym:
```bash
cd rsl_rl && pip install -e . && cd .. 
cd legged_gym &&  pip install -e . && cd .. 
```

# 💻 Reproducing Experimental Results
## Download Video Demonstrations 
|Domain        | Tasks | Episodes| Size | Collection | Link  |       
|:------------- |:-------------:|:-----:|:----:|:-----:|:-----:|
| Adroit | 3 | 150 | 23.8M | [VRL3](https://github.com/microsoft/VRL3) | [Download](https://huggingface.co/datasets/tauhuang/diffusion_reward/tree/main/adroit) 
| MetaWorld | 7 | 140 | 38.8M | [Scripts](https://github.com/Farama-Foundation/Metaworld/blob/master/scripts/demo_sawyer.py) | [Download](https://huggingface.co/datasets/tauhuang/diffusion_reward/tree/main/metaworld)|

You can download the datasets and place them to  `/video_dataset` to reproduce the results in this paper. 

## Pretrain Reward Models
Train VQGAN encoder. 
```bash
bash scripts/run/codec_model/vqgan_${domain}.sh    # [adroit, metaworld]
```
Train video models.
```bash
bash scripts/run/video_model/${video_model}_${domain}.sh    # [vqdiffusion, videogpt]_[adroit, metaworld]
```

### (Optinal) Download Pre-trained Models
We also provide the pre-trained reward models (including Diffusion Reward and VIPER) used in this paper for result reproduction. You may download the models with configuration files [here](https://huggingface.co/tauhuang/diffusion_reward/tree/main), and place the folders in `/exp_local`.

## Train RL with Pre-trained Rewards 
Train DrQv2 with different rewards.
```bash
bash scripts/run/rl/drqv2_${domain}_${reward}.sh ${task}    # [adroit, metaworld]_[diffusion_reward, viper, viper_std, amp, rnd, raw_sparse_reward]
```
Notice that you should login [wandb](https://wandb.ai/site) for logging experiments online. Turn it off, if you aim to log locally, in configuration file [here](diffusion_reward/configs/rl//default.yaml#L24).

# 🧭 Code Navigation

```
diffusion_reward
  |- configs               # experiment configs 
  |    |- models           # configs of codec models and video models
  |    |- rl               # configs of rl 
  |
  |- envs                  # envrionments, wrappers, env maker
  |    |- adroit.py        # Adroit env
  |    |- metaworld.py     # MetaWorld env
  |    |- wrapper.py       # env wrapper and utils
  |
  |- models                # implements core codec models and video models
  |    |- codec_models     # image encoder, e.g., VQGAN
  |    |- video_models     # video prediction models, e.g., VQDiffusion and VideoGPT
  |    |- reward_models    # reward models, e.g., Diffusion Reward and VIPER
  |
  |- rl                    # implements core rl algorithms
```

# ✉️ Contact
For any questions, please feel free to email taou.cs13@gmail.com.


<!-- # 🙏 Acknowledgement
Our code is built upon [VQGAN](https://github.com/dome272/VQGAN-pytorch), [VQ-Diffusion](https://github.com/microsoft/VQ-Diffusion), [VIPER](https://github.com/Alescontrela/viper_rl), [AMP](https://github.com/med-air/DEX), [RND](https://github.com/jcwleo/random-network-distillation-pytorch), and [DrQv2](https://github.com/facebookresearch/drqv2). We thank all these authors for their nicely open sourced code and their great contributions to the community. -->

# 🏷️ License
This repository is released under the MIT license. See [LICENSE](LICENSE) for additional details.

# 📝 Citation

If you find our work useful, please consider citing:
```
@article{huang2025host,
  title={Learning Humanoid Standing-up Control across Diverse Postures},
  author={Huang, Tao and Ren, Junli and Wang, Huayi and Wang, Zirui and Ben, Qingwei and Wen, Muning and Chen, Xiao and Li, Jianan and Pang, Jiangmiao},
  journal={arXiv preprint arXiv:2502.08378},
  year={2025},
}
```