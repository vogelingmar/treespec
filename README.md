# treespec
A deeplearning pipeline to classify tree species in terrestrial panorama pictures. Automatically create datasets for model training and prediction and link the results spatially by using 3D point clouds and existing inventory data.

# Setup
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

3. Activate the virtual environment created by setup.sh.
   
```BibTeX
. venv/bin/activate
```

4. Run the pytest tests to check if everything works.
```BibTeX
pip install -e .[dev]; pytest test
```
  
5. Configure the settings of the scripts (src/treespec/scripts) in the src/conf/config.yaml file (see config.py/ config_parser.py for available options).
   
```BibTeX
nano src/conf/config.yaml
``` 

6. Run any script (example: train.py).
   
```BibTeX
python src/scripts/train.py
```

Now you should see the training progress in your terminal, along with some metrics. 
In the end you can see some statistics and the trained model is saved to src/io/models.


If you want to look further into the training statistics run this command and follow its instructions.

```BibTeX
tensorboard --logdir=lightning_logs/
```

7. For further help you can build the documentation.
```BibTeX
pip install -e .[docs]; cd docs; make html
```
You can now find the generated html files in docs/_build/html.

