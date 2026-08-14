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

## Real-data experiments

The real-data scripts are intended to be run from the repository root. Put
the downloaded data files in a local `data/` directory; do not hard-code a
personal computer path in the scripts.

### Energy Efficiency

Download the dataset from the UCI Machine Learning Repository:

<https://archive.ics.uci.edu/dataset/242/energy+efficiency>

Use the file `ENB2012_data.xlsx`. It contains 768 observations, 8 predictor
variables, and two continuous responses, `Y1` and `Y2`. The usual analysis
uses a random 70/30 train-test split, standardizes predictors using the
training data only, and reports the test MSE averaged over the two responses.

Example command:

```bash
python real_data/run_energy_efficiency_realdata.py \
  --data data/ENB2012_data.xlsx \
  --out_dir results/energy_efficiency \
  --method aks_nnr --seed 1 \
  --d_max 10 --m 0 --h 0 --lambda_c 0.005 \
  --L 3 --hidden_dim 256 --lr 0.001 \
  --patience 20 --val_split 0.30 --stop_threshold 0.30 \
  --max_epochs 800
```

The `--method` option can be set to `aks_nnr`, `full_dnn`, `linear`,
`gmddnet_s`, or `gmddnet_f`. For GMDDNet-S/F, the learned representation is
followed by the downstream regression model specified by the script.

### Student Performance

Download the Student Performance dataset from UCI:

<https://archive.ics.uci.edu/dataset/320/student+performance>

Use `student-mat.csv` or `student-por.csv`. The three continuous responses
are `G1`, `G2`, and `G3`; these columns must be excluded from the predictors.
The remaining categorical variables are encoded before fitting. The default
analysis uses a random 70/30 train-test split, training-split standardization,
and test MSE averaged over the three responses.

Example command:

```bash
python real_data/run_student_performance_realdata.py \
  --data data/student-por.csv \
  --out_dir results/student_performance \
  --method aks_nnr --seed 1 \
  --d_max 10 --m 0 --h 0 --lambda_c 0.005 \
  --L 3 --hidden_dim 256 --lr 0.001 \
  --patience 20 --val_split 0.30 --stop_threshold 0.30 \
  --max_epochs 800
```

To repeat an experiment, change `--seed`. For example, a 200-run study can
use seeds 1 through 200 and store the corresponding JSON files in the same
output directory.

### Summarizing real-data results

After the runs finish, summarize the JSON files with the corresponding
summary script:

```bash
python summarize_energy_efficiency_results.py \
  --base_dir results/energy_efficiency \
  --out_dir results/energy_efficiency/analysis

python summarize_student_performance_results.py \
  --base_dir results/student_performance \
  --out_dir results/student_performance/analysis
```

The analysis directory contains the method-level summary and the individual
run table. Large result folders, logs, checkpoints, and `ghat` files are not
required for the basic repository demonstration and should normally be
excluded from GitHub.

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
