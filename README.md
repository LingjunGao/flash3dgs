# FLASH3DGS: ALGORITHM AND SYSTEM CO-OPTIMIZATION FOR FAST 3D GAUSSIAN SPLATTING ON GPUS

FLASH3DGS is an optimized system and algorithm co-design for accelerating 3D Gaussian Splatting on GPUs. This work introduces novel techniques including **early sorting** and **axis-shared rasterization** to significantly improve rendering performance while maintaining high-quality output.

## Installation

### Environment Setup

First, create and activate a conda environment with the required dependencies:

```bash
# create conda environment
conda create --name gsplat -y python=3.10
conda activate gsplat
pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit
```

### Install FLASH3DGS from Source

Clone and install the FLASH3DGS repository:

```bashFLASH3DGS: ALGORITHM AND SYSTEM CO-OPTIMIZATION FOR
FAST 3D GAUSSIAN SPLATTING ON GPUS
git clone --recurse-submodules https://github.com/LingjunGao/flash3dgs
cd flash3dgs/
pip install -e .
```

### Download Dataset

Navigate to the examples directory and download the benchmark dataset:

```bash
cd examples
pip install -r requirements.txt
python datasets/download_dataset.py
```

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
