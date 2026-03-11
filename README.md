# FLASH3DGS: ALGORITHM AND SYSTEM CO-OPTIMIZATION FOR FAST 3D GAUSSIAN SPLATTING ON GPUS

FLASH3DGS is an optimized system and algorithm co-design for accelerating 3D Gaussian Splatting on GPUs. This work introduces novel techniques including **early sorting** and **axis-shared rasterization** to significantly improve rendering performance while maintaining high-quality output.

## Installation

### Install FLASH3DGS from Source

Clone and install the FLASH3DGS repository:

```bash
git clone --recurse-submodules https://github.com/LingjunGao/flash3dgs
cd flash3dgs/
```

### One-command setup (recommended)

From the FLASH3DGS root directory, run:

```bash
source ./setup_flash3dgs.sh
```

This command will:
- install required system packages,
- create and activate the `flash3dgs` conda environment,
- install FLASH3DGS and Python dependencies,
- download the benchmark dataset.

> If you run `bash ./setup_flash3dgs.sh`, setup will still run, but conda activation will not stay active in your current shell.

## Usage

### Training (Optional)

Training is optional. Pre-trained checkpoints for the benchmark scenes are downloaded automatically by `setup_flash3dgs.sh`.

If you prefer to train your own models, run:

```bash
cd examples
bash benchmarks/Training.sh
```

Results will be saved to `examples/results/New_Training/`.

### Evaluation

To evaluate, run:

```bash
bash benchmarks/Evaluation.sh
```

The script automatically selects the checkpoint root in the following order:
1. `results/New_Training/` — used if you ran `Training.sh` yourself.
2. `results/Benchmark_Training/` — the pre-trained checkpoints downloaded during setup.

If neither folder is found, the script will print instructions on how to train or download the checkpoints.

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
