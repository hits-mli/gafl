# Geometric Algebra Flow Matching (GAFL) for Protein Backbone Generation

Source code for "Generating Highly Designable Proteins with Geometric Algebra Flow Matching" (https://arxiv.org/abs/2411.05238)

If you use this code, please cite:

```
@inproceedings{wagnerseute2024gafl,
  title={Generating Highly Designable Proteins with Geometric Algebra Flow Matching},
  author={Wagner, Simon and Seute, Leif and Viliuga, Vsevolod and Wolf, Nicolas and Gr{\"a}ter, Frauke and St{\"u}hmer, Jan},
  booktitle={Thirty-eighth Conference on Neural Information Processing Systems},
  year={2024}
}
```

This repository is based on FrameFlow (https://github.com/microsoft/protein-frame-flow).

The datasets and weights of the models reported in the paper will be made available in the future

# Installation

### TLDR

```bash
conda env create -f environment.yaml
conda activate gafl
bash install_gatr.sh # Apply patches to gatr

# Install package:
pip install -e .
```

For customizing the environment to a specific torch or CUDA version, it might be easier to modify the `minimal_env.yaml` file provided instead of `environment.yaml`.

### Geometric Algebra Transformer
Geometric Algebra Transformer (gatr) in version 1.2.0 requires the xformers package that resulted in conflicting package dependencies. We therefore require to install gatr from source and apply patches to remove the dependency on xformers. Please note that gatr is distributed under its own license, which you can find in LICENSE.

To install gatr with the required patches please run
```
conda activate gafl
bash install_gatr.sh
```

### Install package
After installing the requirements from `environment.yaml` or `minimal_env.yaml` and applying the patches to gatr, you can install the `gafl` package by running
```bash
pip install -e .
```

# Usage

## Inference

To sample backbone structures using the model (without the re-folding procedure, which is implemented e.g. in [FrameDiff](https://github.com/jasonkyuyim/se3_diffusion)), run
```bash
python experiments/inference.py inference.ckpt_path=<path/to/ckpt>
```

You can specify the inference settings like the number of samples and timesteps via a config file like `configs/inference.yaml`.

### Download model weights

The weights of the models reported in the paper are published at `https://github.com/hits-mli/gafl/releases/download/v1.0.0/gafl-pdb.zip`. Download the zip file, extract it and specify the path to the checkpoint in the inference command:
```bash
mkdir -p outputs
wget https://github.com/hits-mli/gafl/releases/download/v1.0.0/gafl-pdb.zip
unzip gafl-pdb.zip -d outputs
python experiments/inference.py inference.ckpt_path=outputs/gafl-pdb/gafl321.ckpt
```



## Training

To train the model on the scope dataset, paste the path to your metadata csv file in configs/data/default.yaml and run

```bash
python experiments/train.py model=gafl
```

For training on pdb, set paths to the metadata csv file and to the cluster-defining file (as in FrameDiff) in configs/data/pdb.yaml and run

```bash
python experiments/train.py model=gafl data=pdb
```

### Download dataset

GAFL was trained on the [PDB dataset from FrameDiff](https://github.com/jasonkyuyim/se3_diffusion?tab=readme-ov-file#downloading-the-pdb-for-training). For ablations, the [SCOPe dataset from FrameFlow](https://github.com/microsoft/protein-frame-flow/tree/legacy?tab=readme-ov-file#data) was used.
