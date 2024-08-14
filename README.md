# radarODE-MTL
[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Famazing0844%2FradarODE-MTL&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=vistors&edge_flat=false)](https://hits.seeyoufarm.com)

Code for Paper: 

1. [radarODE: An ODE-embedded deep learning model for contactless ECG reconstruction from millimeter-wave radar](https://arxiv.org/abs/2408.01672)
2. radarODE-MTL: A Multi-Task Learning Framework with Eccentric Gradient Alignment for Robust Radar-Based ECG Reconstruction

``radarODE-MTL`` is an open-source library built on [PyTorch](https://pytorch.org/) and Multi-Task Learning (MTL) framework [LibMTL](https://github.com/median-research-group/LibMTL)

:partying_face: Any problem please send them in Issues or eamil [:email:](yuanyuan.zhang16@student.xjtlu.edu.cn).

## Dataset Download
Please refer to [MMECG Dataset](https://github.com/jinbochen0823/RCG2ECG) for the Dataset downloading.

### Run a Model

The complete training code for the NYUv2 dataset is provided in [examples/nyu](./examples/nyu). The file [main.py](./examples/nyu/main.py) is the main file for training on the NYUv2 dataset.

You can find the arguments and settings in.

```shell
radarODE-MTL/Projects/radarODE_plus/main.py
```

More details on the available MTL Architectures, Optimization Strategies, Dataset please refer to [LibMTL](https://github.com/median-research-group/LibMTL).
<!-- ## Contributor

``LibMTL`` is developed and maintained by [Baijiong Lin](https://baijiong-lin.github.io). -->


## Quick Introduction
### Overall Framework for radarODE
<img src='image/radarODE.jpg' width=700>

### Overall Framework for radarODE-MTL
<img src='image/radarODE_MTL.jpg' width=700>

### Intuition behind ECG recovery based on signle cardiac cycle
<img src='image/scg_ecg.jpg' width=350>

### Result from our methods in the presence body movement
<img src='image/result.jpg' width=700>

## Citation

If you find our work helpful for your research, please cite our paper:
```
@article{zhang2024radarODE,
  title={{radarODE: An ODE-embedded deep learning model for contactless ECG reconstruction from millimeter-wave radar}}, 
  author={Yuanyuan Zhang and Runwei Guan and Lingxiao Li and Rui Yang and Yutao Yue and Eng Gee Lim},
  year={2024},
  month={Aug.},
  journal={arXiv preprint arXiv:2408.01672 [eess]},
  month={Aug.},
}
```

## License

``radarODE-MTL`` is released under the [MIT](./LICENSE) license.