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
    num_classes: int
    epoch_count: int
    batch_size: int
    lr: float
    loss_function: str

@dataclass
class TreespecConfig:
    paths: Directories
    files: Files
    params: TrainParams