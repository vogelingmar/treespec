"""Extract and predict tree images from a video and organize them into class folders according to the predictions."""

import os
import shutil
import torch
import hydra
from hydra.core.config_store import ConfigStore

from treespec.models.lumberjack import Lumberjack
from treespec.models.classification_model import ClassificationModel
from treespec.scripts.train import model_dict, model_weights_dict, loss_function_dict

from treespec.conf.config import TreespecConfig

cs = ConfigStore.instance()
cs.store(name="treespec_config", node=TreespecConfig)


@hydra.main(config_path="../conf", config_name="config")
def main(cfg: TreespecConfig):
    r"""
    Script that extracts tree images from a video and organizes them into class folders according to the predictions.
    """

    # Initialize Lumberjack and ClassificationModel
    lumberjack = Lumberjack(
        model=cfg.ExtractParams.model,
        output_trees_dir=cfg.ExtractParams.output_trees_dir,
        predict_video_dest_dir=cfg.ExtractParams.predict_video_dest_dir,
        visualize=cfg.ExtractParams.visualize,
    )
    print(type(loss_function_dict[cfg.TrainParams.loss_function]))

    classification_model = ClassificationModel(
        model=model_dict[cfg.TrainParams.model],
        model_weights=model_weights_dict[cfg.TrainParams.model_weights],
        num_classes=cfg.TrainParams.num_classes,
        loss_function=loss_function_dict[cfg.TrainParams.loss_function],
        learning_rate=cfg.TrainParams.learning_rate,
    )

    # Load the trained model weights
    trained_model_path = cfg.TrainParams.trained_model_dir + cfg.TrainParams.model + "_finetuned" + ".pth"
    classification_model.model.load_state_dict(torch.load(trained_model_path))
    classification_model.eval()  # Set the model to evaluation mode

    # Process video to extract tree images
    lumberjack.process_video(
        video_path=cfg.ExtractParams.video,
        corrected=cfg.ExtractParams.corrected,
    )

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
        predicted_class_id = prediction["category"]

        class_dict = {0: "beech", 1: "chestnut", 2: "pine"}

        # Move the image to the corresponding class folder
        target_dir = output_dirs[class_dict[predicted_class_id]]
        shutil.move(image_path, os.path.join(target_dir, image_name))


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
