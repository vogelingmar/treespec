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
    use_ids: bool
    epoch_count: int
    batch_size: int
    num_workers: int
    learning_rate: float
    loss_function: str
    use_augmentations: bool
    trained_model_dir: str
    train_augmentations: list


@dataclass
class EssenDatasetParams:  # pylint: disable=too-many-instance-attributes
    """Configuration of parameters for the create_essen_dataset script"""

    original_color_images_path: str
    color_images_path: str
    color_type: str
    color_output_type: str
    original_id_images_path: str
    segmentid_images_path: str
    seg_type: str
    seg_output_type: str
    original_sem_images_path: str
    semantic_images_path: str
    sem_type: str
    sem_output_type: str
    run: str
    output_trees_dir: str
    attribute_path: str
    mask: str
    filter_id: str
    filter_semantic: str
    crop: bool
    pictures_extracted: bool


@dataclass
class PredictEssen:
    """Configuration of parameters for the predict_essen script"""

    tree_images_dir: str
    input_inventory_path: str
    output_inventory_path: str
    trained_model_path: str


@dataclass
class Matching:
    """Configuration of parameters for the matching script"""

    predicted_cadastre_path: str
    cadastre_path: str
    output_path: str
    use_dbh_filter: bool


@dataclass
class TreespecConfig:
    """Configuration of the configs going into the treespec config"""

    train: TrainParams
    essen_dataset: EssenDatasetParams
    predict_essen: PredictEssen
    matching: Matching
