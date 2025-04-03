# treespec
A deeplearning pipeline to classify tree species in terrestrial panorama pictures and map the results on a 3D point cloud.

# Setup
When first setting up treespec you have to have Python3 installed on your system. To create a virtual environment
and install all the required dependecies to run the treespec pipeline follow these steps:
1. Navigate into your local treespec repo.
```BibTeX
cd treespec/
```

2. Run the setup script.

```BibTeX
bash setup.sh
```

3. To test if the classification model works you can run the run.py file.
```BibTeX
python src/run.py
```

Now you should see the training progress in your terminal. In the end you can see some statistics and the prediction
result of three test images (note: they could be part of the training set).

If you want to look further into the training statistics run this command and follow its instructions.
```BibTeX
tensorboard --logdir=lightning_logs/
```