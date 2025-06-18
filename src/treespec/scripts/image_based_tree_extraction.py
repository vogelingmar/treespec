"""Extract and predict tree images from a video and organize them into class folders according to the predictions."""

import os
import shutil
import torch

from treespec.models.lumberjack import Lumberjack
from treespec.models.classification_model import ClassificationModel
from treespec.conf.config_parser import image_based_extract_config_values as config_values
from treespec.conf.config_parser import train_config_values

if __name__ == "__main__":
    r"""
    Script that extracts tree images from a video and organizes them into class folders according to the predictions.
    """
    prediction_video_dir = None
    if config_values("predict_video_dest_dir") is not None:
        prediction_video_dir = config_values("predict_video_dest_dir")

    # Initialize Lumberjack and ClassificationModel
    lumberjack = Lumberjack(
        model=config_values("model"),
        output_trees_dir=config_values("output_trees_dir"),
        predict_video_dest_dir=prediction_video_dir,
        visualize=config_values("visualize"),
    )

    classification_model = ClassificationModel(
        model=train_config_values("model"),
        model_weights=train_config_values("model_weights"),
        num_classes=train_config_values("num_classes"),
        loss_function=train_config_values("loss_function")(),
        learning_rate=train_config_values("learning_rate"),
    )

    # Load the trained model weights
    trained_model_path = train_config_values("trained_model_dir") + train_config_values("model") + "_finetuned" + ".pth"
    classification_model.model.load_state_dict(torch.load(trained_model_path))
    classification_model.eval()  # Set the model to evaluation mode

    # Process video to extract tree images
    if config_values("video") is not None and config_values("corrected") is not None:
        lumberjack.process_video(
            video_path=config_values("video"),
            corrected=config_values("corrected"),
            mask=config_values("mask"),
        )
    if config_values("image_dir") is not None and config_values("cameras") is not None and config_values("image_filetype") is not None:
        lumberjack.process_images(
            image_dir=config_values("image_dir"),
            cameras=config_values("cameras"),
            filetype=config_values("image_filetype"),
            mask=config_values("mask"),
        )

    if config_values("predict") == True:
        # Directory containing extracted tree images
        output_trees_dir = lumberjack.output_trees_dir

        dataset = train_config_values("dataset")(
            data_dir=train_config_values("dataset_dir"),
            batch_size=train_config_values("batch_size"),
            num_workers=train_config_values("num_workers"),
        )
        # Define output directories for each class
        class_names = dataset.classes
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
            predicted_class_id = prediction["category"]

            # Move the image to the corresponding class folder
            target_dir = output_dirs[class_names[predicted_class_id]]
            shutil.move(image_path, os.path.join(target_dir, image_name))
