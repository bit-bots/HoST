# HoST


[![arXiv](https://img.shields.io/badge/arXiv-2502.08378-brown)](https://arxiv.org/abs/2502.08378)
[![](https://img.shields.io/badge/Website-%F0%9F%9A%80-yellow)](https://taohuang13.github.io/humanoid-standingup.github.io/)
[![](https://img.shields.io/badge/Youtube-🎬-red)](https://www.youtube.com/watch?v=Yruh-3CFwE4)
[![](https://img.shields.io/badge/bilibili-📹-blue)](https://www.bilibili.com/video/BV1o2KPeUEob/?spm_id_from=333.337.search-card.all.click&vd_source=ef6a9a20816968cc19099a3f662afd86)

This is the official PyTorch implementation of the paper "[**Learning Humanoid Standing-up Control across Diverse Postures**]()" by 

[Tao Huang](https://taohuang13.github.io/), [Junli Ren](https://renjunli99.github.io/), [Huayi Wang](https://why618188.github.io/), [Zirui Wang](https://scholar.google.com/citations?user=Vc3DCUIAAAAJ&hl=zh-TW), [Qingwei Ben](https://www.qingweiben.com/), [Muning Wen]([Muning Wen](https://scholar.google.com/citations?user=Zt1WFtQAAAAJ&hl=en)), [Xiao Chen](https://xiao-chen.tech/), [Jianan Li](https://github.com/OpenRobotLab/HoST), [Jiangmiao Pang](https://oceanpang.github.io/)

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


# ✉️ Contact
For any questions, please feel free to email taou.cs13@gmail.com.


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