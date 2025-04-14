from dataclasses import dataclass


@dataclass
class Directories:
    bark_dir: str
    output_dir: str


@dataclass
class Files:
    model: str
    video: str
    corrected_video: str
    predict_video: str


@dataclass
class TrainParams:
    model: str
    model_weights: str
    dataset: str
    num_classes: int
    epoch_count: int
    batch_size: int
    num_workers: int
    learning_rate: float
    loss_function: str
    use_augmentations: bool


# TODO: create extra dataset config

# TODO: change video recognition class structure


@dataclass
class TreespecConfig:
    paths: Directories
    files: Files
    params: TrainParams
