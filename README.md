# treespec

![pypi-image](https://badge.fury.io/py/treespec.svg)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE) 
[![CI](https://github.com/vogelingmar/treespec/actions/workflows/main.yml/badge.svg)](https://github.com/vogelingmar/treespec/actions/workflows/main.yml)
[![coverage](https://codecov.io/gh/vogelingmar/treespec/branch/main/graph/badge.svg)](https://codecov.io/github/vogelingmar/treespec?branch=main)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/treespec)


> A deeplearning pipeline to classify tree species in terrestrial panorama pictures.

## About
treespec is a PyTorch-lightning based deep learning pipeline equipped with tools helpfull for creating datasets from images, 3D point clouds and shapefile data containing inventory information.

**Features:**
- match tree inventories
- create masked datasets
- train standard torchvision models

## Installation

### Method 1: Docker Container
Download the current [treespec container](https://hub.docker.com/repository/docker/vogelingmar/treespec/general) on Docker Hub ([Docker](https://docs.docker.com/engine/install/) required).
```BibTeX
docker pull vogelingmar/treespec:latest
```
Run the container with docker run and mount folders you want to work with.
```BibTeX
docker run -it --gpus all -v *local_path*:/workspace/data vogelingmar/treespec
```

### Method 2: GitHub Repository
Clone the [treespec repository](https://github.com/vogelingmar/treespec) from GitHub ([Git](https://github.com/git-guides/install-git) required).

When first setting up treespec you have to have [Python3](https://www.python.org/downloads/) installed on your system. To create a virtual environment
and install all the required dependecies to run the treespec pipeline follow these steps:
1. Navigate into your local treespec repo.
   
```BibTeX
cd treespec/
```

2. Run the setup script.

```BibTeX
bash setup.sh
```


## Usage

1. Activate the virtual environment created by setup.sh.
   
```BibTeX
. venv/bin/activate
```

2. Run the pytest tests to check if everything works.
```BibTeX
pip install -e .[dev]; pytest test
```
  
3. Configure the settings of the scripts (src/treespec/scripts) in the src/conf/config.yaml file (see config.py/ config_parser.py for available options).
   
```BibTeX
nano src/conf/config.yaml
``` 

4. Run any script (example: train.py).
   
```BibTeX
python src/scripts/train.py
```

Now you should see the training progress in your terminal, along with some metrics. 
In the end you can see some statistics and the trained model is saved to src/io/models.


If you want to look further into the training statistics run this command and follow its instructions.

```BibTeX
tensorboard --logdir=lightning_logs/
```

5. For further help you can build the documentation.
```BibTeX
pip install -e .[docs]; cd docs; make html
```
You can now find the generated html files in docs/_build/html.

