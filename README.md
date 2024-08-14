# radarODE-MTL
[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Famazing0844%2FradarODE-MTL&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=vistors&edge_flat=false)](https://hits.seeyoufarm.com)

Code for Paper: 

1. [radarODE: An ODE-embedded deep learning model for contactless ECG reconstruction from millimeter-wave radar](https://arxiv.org/abs/2408.01672)
2. radarODE-MTL: A Multi-Task Learning Framework with Eccentric Gradient Alignment for Robust Radar-Based ECG Reconstruction


``radarODE-MTL`` is an open-source library built on [PyTorch](https://pytorch.org/) and Multi-Task Learning (MTL) framework [LibMTL](https://github.com/median-research-group/LibMTL)

:partying_face: Any problem please send them in Issues or [:email:](yuanyuan.zhang16@student.xjtlu.edu.cn).

## Dataset Download
Please refer to [MMECG Dataset](https://github.com/jinbochen0823/RCG2ECG) for the Dataset downloading.


## Features

- **Unified**:  ``LibMTL`` provides a unified code base to implement and a consistent evaluation procedure including data processing, metric objectives, and hyper-parameters on several representative MTL benchmark datasets, which allows quantitative, fair, and consistent comparisons between different MTL algorithms.
- **Comprehensive**: ``LibMTL`` supports many state-of-the-art MTL methods including 8 architectures and 16 optimization strategies. Meanwhile, ``LibMTL`` provides a fair comparison of several benchmark datasets covering different fields.
- **Extensible**:  ``LibMTL`` follows the modular design principles, which allows users to flexibly and conveniently add customized components or make personalized modifications. Therefore, users can easily and fast develop novel optimization strategies and architectures or apply the existing MTL algorithms to new application scenarios with the support of ``LibMTL``.

## Overall Framework

 ![framework](./docs/docs/images/framework.png)

Each module is introduced in [Docs](https://libmtl.readthedocs.io/en/latest/docs/user_guide/framework.html).



### Run a Model

The complete training code for the NYUv2 dataset is provided in [examples/nyu](./examples/nyu). The file [main.py](./examples/nyu/main.py) is the main file for training on the NYUv2 dataset.

You can find the command-line arguments by running the following command.

```shell
python main.py -h
```

For instance, running the following command will train an MTL model with EW and HPS on NYUv2 dataset.

```shell
python main.py --weighting EW --arch HPS --dataset_path /path/to/nyuv2 --gpu_id 0 --scheduler step --mode train --save_path PATH
```

More details is represented in [Docs](https://libmtl.readthedocs.io/en/latest/docs/getting_started/quick_start.html).

<!-- ## Contributor

``LibMTL`` is developed and maintained by [Baijiong Lin](https://baijiong-lin.github.io). -->


## Quick Introduction
<img src='image/radarODE.pdf' width=700>
<img src='image/radarODE_MTL.pdf' width=700>
<img src='image/scg_ecg.pdf' width=700>
<img src='image/result.pdf' width=700>

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