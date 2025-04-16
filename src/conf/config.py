from dataclasses import dataclass

@dataclass
class TrainParams:
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

@dataclass
class ExtractParams:
    model: str
    output_trees_dir: str
    predict_video_dest_dir: str
    visualize: bool
    video: str
    corrected: bool


# TODO: create extra dataset config



@dataclass
class TreespecConfig:
    train: TrainParams
    extract: ExtractParams
