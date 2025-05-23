"""Definition of the config parameters"""

from dataclasses import dataclass


@dataclass
class TrainParams:  # pylint: disable=too-many-instance-attributes
    """Configuration of parameters for the training process"""

    model: str
    model_weights: str
    dataset: str
    dataset_dir: str
    num_classes: int
    epoch_count: int
    batch_size: int
    num_workers: int
    learning_rate: float
    loss_function: str
    use_augmentations: bool
    trained_model_dir: str
    train_augmentations: list


@dataclass
class ExtractParams:  # pylint: disable=too-many-instance-attributes
    """Configuration of parameters fot the extraction process"""

    model: str
    output_trees_dir: str
    predict_video_dest_dir: str
    visualize: bool
    video: str
    corrected: bool
    image_dir: str
    cameras: list
    image_filetype: str


@dataclass
class TreespecConfig:
    """Configuration of the configs going into the treespec config"""

    train: TrainParams
    extract: ExtractParams
