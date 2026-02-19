# GenModeling
Repo for a variety of generative models for unconditional and conditional generation

## Usage

### 1. Setup Environment

First, create and activate the conda environment using the provided `environment.yml` file:

```bash
conda env create -f environment.yml
conda activate gen-modeling
```

### 2. Configure Environment Variables

Create a `.env` file in the root of the project to set necessary environment variables. This file should extend your `PYTHONPATH` to include this repository and set `CUDA_VISIBLE_DEVICES`.

Here is an example `.env` file:

```bash
# .env
PYTHONPATH=${PYTHONPATH}:/path/to/your/GenModeling
CUDA_VISIBLE_DEVICES=0
```

**Note:** Replace `/path/to/your/GenModeling` with the absolute path to this repository on your machine.

Load these environment variables into your shell session by running:

```bash
set -o allexport && source .env && set +o allexport
```

### 3. Running Scripts

To run the train scripts, you need to provide a path to a configuration file.

For example, to run `sample.py`:
```bash
python scripts/train.py path/to/your/config.yml
```

Similarly, for training on multiple gpus:
```bash
python scripts/train_distributed.py path/to/your/config.yml
```

To run the inference scripts, you need to provide a path to a configuration file.

For example, to run `sample.py`:
```bash
python scripts/sample.py path/to/your/config.yml
```

Similarly, to run `sample_edm.py`:
```bash
python scripts/sample_edm.py path/to/your/config.yml
```

Make sure your configuration file specifies the correct model checkpoint and inference parameters.
