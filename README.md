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
