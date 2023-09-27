# time-series

## Requirements

To install conda on your remote Linux server, use the following commands:

```sh
cd /tmp
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
sh Miniconda3-latest-Linux-x86_64.sh
```

To set up the environment with conda, use the following commands:

```sh
conda create --name time-series python=3.9
conda activate time-series
python -m pip install -r requirements.txt
```