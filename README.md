# REUCL_MOE

# Theory on Mixture-of-Experts in Continual Learning

## Codebase Structure

```bash
.
├── main                            # main codebase folder
│   ├── config                      # global config file
│   │   ├── exp_config.yaml
│   │   └── exp_id.yaml
│   ├── models
│   │   └── resnet18_ocm.py
│   ├── pipe_plain.py               # main entry point
│   ├── utils
│   │   ├── args_config.py
│   │   ├── datasets_config.py
│   │   ├── exp_config.py
│   │   ├── expdata_config.py
│   │   ├── exp_stats.py
│   │   └── stats.py
│   └── utils_cl
│       ├── metric.py
│       └── train.py
├── Makefile                        # script for running small-scale experiments
├── plots
│   ├── example.ipynb
│   └── plot.py                     # util code for analyzing experiment results
├── r1_syn                          # synthetic data simulation code
│   ├── fig2_forgetting_error.py
│   └── fig3_NN_MoE.py
└── Readme.md
```


## Environment Setup
Step 1. Prepare temp folder
```bash
mkdir .data
mkdir .exp_result
```
Step 2. Prepare dataset: manually download a prepared dataset file from https://drive.google.com/file/d/1jUf8ff62dGXOsFtonSXU7yMy20zq6Rtn/view, and put it under `.data/`. Alternatively, you may use `gdown` as suggested by [this link](https://stackoverflow.com/questions/25010369/wget-curl-large-file-from-google-drive). Then, unpack it:
```bash
cd .data && tar -xzf cifar-100.tgz && cd ..
```

Step 3. Prepare conda environment: you may start with the environment created during week 2 (please check [this link](https://github.com/williamqwu/ml-tutorials-suite/blob/main/notes/env.md)). Additionally:
```bash
pip install colorlog pyyaml
```

Step 4. Activate your conda environment, and start a minimal test:
```bash
# NOTE: make sure you are at your project root directory
# conda activate your_env_name
make min_test
```

This project explores Mixture-of-Experts (MoE) models for continual learning, focusing on how to effectively learn from a sequence of tasks without forgetting previously acquired knowledge.

## Core Idea

In traditional neural networks, learning new tasks often leads to catastrophic forgetting, where performance on earlier tasks degrades. This project addresses that problem using a Mixture-of-Experts (MoE) approach:

* Multiple expert networks are trained

* A gating mechanism decides which expert(s) to use for each input

* Knowledge is distributed across experts instead of being overwritten

Continual Learning Setting

* The project simulates a continual (incremental) learning scenario, where:

* Data arrives in sequential tasks

* The model must learn each task in order

* Past data is limited or unavailable

* Performance is evaluated on all previously seen tasks

## Key Components
* Model (resnet18_ocm.py) - A ResNet-18–based architecture adapted for continual learning
  
* Supports expert-style modularization

* Training Pipeline (pipe_plain.py)

* Main entry point for running experiments

* Handles task sequencing, training, and evaluation

* Continual Learning Utilities (utils_cl/)

* Training loops tailored for sequential tasks

Metrics for:
* Accuracy

* Forgetting

* Task performance over time

Experiment Configuration (config/)
* YAML-based configs for reproducible experiments

* Controls datasets, model settings, and training parameters

Synthetic Experiments (r1_syn/)

* Simulated setups to analyze:
* Forgetting behavior
* MoE vs standard neural networks

## Visualization (plots/)
Tools for analyzing results

* Jupyter notebook for generating figures

## What You Can Do With It
* Run continual learning experiments on CIFAR-100

* Compare MoE vs standard neural networks

* Measure catastrophic forgetting

* Visualize how knowledge evolves across tasks

* Reproduce figures from research experiments

## Goal

The main goal is to demonstrate that Mixture-of-Experts architectures can improve knowledge retention in continual learning by:

* Isolating task-specific knowledge

* Reducing interference between tasks

* Enabling scalable, modular learning systems
