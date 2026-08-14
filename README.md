# Nonlinear Sufficient Dimension Reduction with Deep Learning

This repository contains the simulation code for nonlinear sufficient
dimension reduction with a neural-network regression component.

## Repository contents

```text
data_generator_all_settings.py   Data-generating mechanisms
main.py                           Main simulation entry point
Step1_initialization.py           Network initialization
Step2_model_training.py           Neural-network training
Step3_dimension_estimation.py     Dimension estimation
article_models_demo.ipynb         Browser-based Jupyter demonstration
```

The article simulation models are labelled A--J. Models A--E have structural
dimension two, and Models F--J have structural dimension three.

## Installation

Create or activate a Python environment, then install the required packages:

```bash
python -m pip install numpy scipy scikit-learn pandas torch jupyter
```

The exact PyTorch installation command may depend on whether the experiment
uses a CPU or CUDA-enabled GPU.

## Run from the command line

Run commands from the repository root, where `main.py` is located. For
example:

```bash
python main.py --model A --scenario I --p 20 --n 500 --m 500 \
  --h 0 --d_max 10 --lambda_c 0.005 --lambda_n 0.649 \
  --lr 0.001 --L 3 --hidden_dim 256 --patience 20 \
  --val_split 0.30 --stop_threshold 0.30 --max_epochs 800 \
  --seed_start 1 --n_seeds 1 --output_dir results/model_A \
  --save_ghat
```

For `n=1000`, use the corresponding value of `lambda_n` selected in the
simulation configuration.

## Run in a browser with Jupyter

Start Jupyter from the repository root:

```bash
jupyter notebook article_models_demo.ipynb
```

The notebook uses `Path.cwd()` and relative repository paths. It does not
depend on the original author's computer path. Run the cells from top to
bottom, choose the simulation setting, and run one model or all models A--J.
Results are written to the local `jupyter_results/` folder.

## Reproducibility

Each experiment specifies a random seed. The output directory contains the
JSON summary, CSV summary, and, when requested, the saved estimated
representation (`ghat` files).

## License and citation

Please add the license and paper citation information before making the
repository public.
