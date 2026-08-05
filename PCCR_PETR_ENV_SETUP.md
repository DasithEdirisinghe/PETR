# PETR Environment Setup for PCCR

This document records the working PETR environment used for training PETR on
the PCCR R1 dataset.

## Software versions

- Linux
- Python 3.6.8
- PyTorch 1.9.0
- TorchVision 0.10.0
- CUDA toolkit 11.1 packaged with PyTorch
- CUDA 11.2-compatible host driver/toolkit
- MMCV-Full 1.4.0
- MMDetection 2.24.1
- MMSegmentation 0.20.2
- MMDetection3D 0.17.1

## 1. Clone the repositories

```bash
cd /path/to/your/dev_ws

git clone https://github.com/megvii-research/PETR.git

git clone https://github.com/open-mmlab/mmdetection3d.git
cd mmdetection3d
git checkout v0.17.1
cd ..

git clone https://github.com/BaderTim/plentiful-carla-camera-rigs.git
```

## 2. Create the Conda environment

```bash
conda create -n petr python=3.6.8 -y
conda activate petr
```

Install Python 3.6-compatible packaging tools:

```bash
python -m pip install \
    pip==21.3.1 \
    setuptools==59.6.0 \
    wheel==0.37.1
```

## 3. Install PyTorch

```bash
conda install -y \
    pytorch=1.9.0 \
    torchvision=0.10.0 \
    cudatoolkit=11.1 \
    -c pytorch

conda install -y mkl=2024.0
```

Verify the installation:

```bash
python -c "import torch; print('torch:', torch.__version__); print('CUDA:', torch.version.cuda); print('available:', torch.cuda.is_available())"
```

## 4. Install compatible base dependencies

```bash
python -m pip install \
    numpy==1.19.5 \
    opencv-python==4.5.5.64

conda install -y -c conda-forge geos shapely
```

## 5. Install MMCV-Full

```bash
python -m pip install \
    mmcv-full==1.4.0 \
    -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html
```

Verify MMCV:

```bash
python -c "import mmcv; print('mmcv:', mmcv.__version__)"
```

## 6. Install MMDetection and MMSegmentation

```bash
python -m pip install \
    mmdet==2.24.1 \
    mmsegmentation==0.20.2
```

Verify the installations:

```bash
python -c "import mmdet, mmseg; print('mmdet:', mmdet.__version__); print('mmseg:', mmseg.__version__)"
```

## 7. Install MMDetection3D in editable mode

```bash
cd /path/to/your/dev_ws/mmdetection3d

python -m pip install -r requirements/build.txt
python -m pip install -v -e .
```

This installs MMDetection3D inside the active `petr` Conda environment while
keeping the source checkout editable.

Verify it:

```bash
python -c "import mmdet3d; print('mmdet3d:', mmdet3d.__version__); print('location:', mmdet3d.__file__)"
```

## 8. Install the additional PETR dependency

The PETR plugin requires `einops`:

```bash
python -m pip install einops==0.4.1
```

The PETR repository's `requirements.txt` was not installed as a separate step
in this environment. Its relevant runtime packages were already installed by
the preceding MMDetection3D installation and the explicit dependency commands.

## 9. Add PCCR support to PETR

```bash
cd /path/to/your/dev_ws/PETR

cp \
    ../plentiful-carla-camera-rigs/models/PETR/pccr_dataset.py \
    projects/mmdet3d_plugin/datasets/pccr_dataset.py

cp \
    ../plentiful-carla-camera-rigs/models/PETR/petr_r50dcn_gridmask_p4_800x320_pccr.py \
    projects/configs/petr/petr_r50dcn_gridmask_p4_800x320_pccr.py
```

Register `PCCRDataset` in
`projects/mmdet3d_plugin/datasets/__init__.py`:

```python
from .pccr_dataset import PCCRDataset
```

Also expose `PCCRDataset` from
`projects/mmdet3d_plugin/__init__.py`.

## 10. Download the pretrained ResNet-50 backbone

```bash
cd /path/to/your/dev_ws/PETR

mkdir -p ckpts

wget \
    https://download.openmmlab.com/pretrain/third_party/resnet50_msra-5891d200.pth \
    -O ckpts/resnet50_msra-5891d200.pth
```

Verify the file:

```bash
ls -lh ckpts/resnet50_msra-5891d200.pth
sha256sum ckpts/resnet50_msra-5891d200.pth
```

Expected SHA-256:

```text
5891d200ef31febc610b915837775354fa880be47283cd0b14648a95aa13465e
```

## 11. Link the PCCR R1 dataset

```bash
cd /path/to/your/dev_ws/PETR

mkdir -p data/pccr

ln -s \
    /path/to/your/dev_ws/plentiful-carla-camera-rigs/pccr-dataset/pccr/data/R1 \
    data/pccr/R1
```

The PCCR PETR configuration should use:

```python
data_root = 'data/pccr/R1/'
```

Verify the annotation files:

```bash
ls data/pccr/R1/R1_infos_train.pkl
ls data/pccr/R1/R1_infos_val.pkl
ls data/pccr/R1/R1_infos_test.pkl
```

## 12. Configure `PYTHONPATH`

When running PETR directly through `tools/train.py`, add the PETR repository
root to `PYTHONPATH`:

```bash
cd /path/to/your/dev_ws/PETR
export PYTHONPATH="$(pwd):${PYTHONPATH}"
```

PETR's `tools/dist_train.sh` sets the repository path automatically for a
distributed training run.

## 13. Verify the complete environment

```bash
python -c "
import torch
import mmcv
import mmdet
import mmseg
import mmdet3d
import einops
import projects.mmdet3d_plugin
from mmdet.datasets import DATASETS

print('torch:', torch.__version__)
print('torch CUDA:', torch.version.cuda)
print('CUDA available:', torch.cuda.is_available())
print('mmcv:', mmcv.__version__)
print('mmdet:', mmdet.__version__)
print('mmseg:', mmseg.__version__)
print('mmdet3d:', mmdet3d.__version__)
print('einops:', einops.__version__)
print('PCCRDataset:', DATASETS.get('PCCRDataset'))

assert DATASETS.get('PCCRDataset') is not None
"
```

The environment is ready when all imports succeed and `PCCRDataset` is present
in the MMDetection registry.
