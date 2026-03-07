# FLASH3DGS: ALGORITHM AND SYSTEM CO-OPTIMIZATION FOR FAST 3D GAUSSIAN SPLATTING ON GPUS

FLASH3DGS is an optimized system and algorithm co-design for accelerating 3D Gaussian Splatting on GPUs. This work introduces novel techniques including **early sorting** and **axis-shared rasterization** to significantly improve rendering performance while maintaining high-quality output.

## Installation

From the FLASH3DGS root directory, run:

```bash
source ./setup_flash3dgs.sh
```

This will install all dependencies, create the `flash3dgs` conda environment, build FLASH3DGS from source, and download the benchmark dataset.

> Use `source` (not `bash`) so the conda environment stays active in your current shell.

## Usage

### Training

To train models on benchmark scenes:

```bash
cd examples
bash benchmarks/Training.sh
```

### Evaluation

To evaluate trained models:

```bash
bash benchmarks/Evaluation.sh
```

## Acknowledgements

This work is based on the [gsplat library](https://github.com/nerfstudio-project/gsplat) developed by the Nerfstudio team. We are grateful for their excellent open-source contribution, which provided the foundation for our optimizations.

If you find the original gsplat library useful in your projects or papers, please consider citing:

```
@article{ye2024gsplatopensourcelibrarygaussian,
    title={gsplat: An Open-Source Library for {Gaussian} Splatting}, 
    author={Vickie Ye and Ruilong Li and Justin Kerr and Matias Turkulainen and Brent Yi and Zhuoyang Pan and Otto Seiskari and Jianbo Ye and Jeffrey Hu and Matthew Tancik and Angjoo Kanazawa},
    year={2024},
    eprint={2409.06765},
    journal={arXiv preprint arXiv:2409.06765},
    archivePrefix={arXiv},
    primaryClass={cs.CV},
    url={https://arxXiv.org/abs/2409.06765}, 
}
```
