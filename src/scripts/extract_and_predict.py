import os
import torch
import shutil
from torchvision.models import ResNet50_Weights
from src.models.lumberjack import Lumberjack
from src.models.classification_model import ClassificationModel

# Initialize Lumberjack and ClassificationModel
lumberjack = Lumberjack(visualize=False)
classification_model = ClassificationModel(
    model_weights=ResNet50_Weights.DEFAULT, #Replace with appropriate weights
    num_classes=3  # Adjust based on your dataset
)
# Load the trained model weights
trained_model_path = "/home/ingmar/Documents/repos/treespec/src/io/models/resnet50_finetuned.pth"  # Path to the saved model
classification_model.model.load_state_dict(torch.load(trained_model_path))
classification_model.eval()  # Set the model to evaluation mode

# Process video to extract tree images
#lumberjack.process_video(corrected=True)

# Directory containing extracted tree images
output_trees_dir = lumberjack.output_trees_dir

# Define output directories for each class
class_names = ["beech", "chestnut", "pine"]  # Replace with your class names
output_dirs = {class_name: os.path.join(output_trees_dir, class_name) for class_name in class_names}

# Create directories for each class
for class_dir in output_dirs.values():
    os.makedirs(class_dir, exist_ok=True)

# Predict and organize images
for image_name in os.listdir(output_trees_dir):
    image_path = os.path.join(output_trees_dir, image_name)

    # Skip directories
    if os.path.isdir(image_path):
        continue

    # Predict the class of the image
    prediction = classification_model.predict(image_path)
    predicted_class = prediction["category"]

    # Move the image to the corresponding class folder
    target_dir = output_dirs[predicted_class]
    shutil.move(image_path, os.path.join(target_dir, image_name))