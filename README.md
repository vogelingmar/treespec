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

3. Configure the settings of both the model, that should be trained and the tree extraction from the video in the
src/conf/config.yaml file.
```BibTeX
nano src/conf/config.yaml
``` 

4. To test if the classification model works you can run the train.py script.
```BibTeX
python src/scripts/train.py
```
Now you should see the training progress in your terminal, along with some metrics. 
In the end you can see some statistics and the finetuned model is saved to src/io/models.


If you want to look further into the training statistics run this command and follow its instructions.
```BibTeX
tensorboard --logdir=lightning_logs/
```

5. To test if the tree extraction and prediction works run the extract_and_predict.py script.
```BibTeX
python src/scripts/extract_and_predict.py
```
After this script has finished you can find the tree images extracted from the given video at src/io/pictures 
sorted into the correct folder according to the prediction results.

