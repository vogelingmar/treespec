import hydra
from hydra.core.config_store import ConfigStore
from treespec.conf.config import TreespecConfig

cs = ConfigStore.instance()
cs.store(name="treespec_config", node=TreespecConfig)

@hydra.main(config_path="./", config_name="config")
def train_config_values(cfg: TreespecConfig, param: str):

    match param:
        case "model":
            match cfg.train.model:
                case "resnet50":
                    from torchvision.models import resnet50
                    return resnet50
                case "resnet152":
                    from torchvision.models import resnet152
                    return resnet152
                case "swin_transformer":
                    from torchvision.models import swin_v2_b
                    return swin_v2_b
                case "efficientnet":
                    from torchvision.models import efficientnet_v2_m
                    return efficientnet_v2_m
                case "googlenet":
                    from torchvision.models import googlenet
                    return googlenet
                case "mobilenet":
                    from torchvision.models import mobilenet_v3_large
                    return mobilenet_v3_large
                case "wide_resnet":
                    from torchvision.models import wide_resnet101_2
                    return wide_resnet101_2
                case _:
                    raise ValueError(f"Unknown model: {cfg.train.model}")
        case "model_weights":
            match cfg.train.model_weights:
                case "resnet50_default":
                    from torchvision.models import ResNet50_Weights
                    return ResNet50_Weights.DEFAULT
                case "resnet152_default":
                    from torchvision.models import ResNet152_Weights
                    return ResNet152_Weights.DEFAULT
                case "swin_default":
                    from torchvision.models import Swin_V2_B_Weights
                    return Swin_V2_B_Weights.DEFAULT
                case "efficientnet_default":
                    from torchvision.models import EfficientNet_V2_M_Weights
                    return EfficientNet_V2_M_Weights.DEFAULT
                case "googlenet_default":
                    from torchvision.models import GoogLeNet_Weights
                    return GoogLeNet_Weights.DEFAULT
                case "mobilenet_default":
                    from torchvision.models import MobileNet_V3_Large_Weights
                    return MobileNet_V3_Large_Weights.DEFAULT
                case "wide_resnet_default":
                    from torchvision.models import Wide_ResNet101_2_Weights
                    return Wide_ResNet101_2_Weights.DEFAULT
                case _:
                    raise ValueError(f"Unknown model weights: {cfg.train.model_weights}")
        case "dataset":
            match cfg.train.dataset:
                case "folder":
                    from treespec.datasets.image_dataset import ImageDataset
                    return ImageDataset
                case _:
                    raise ValueError(f"Unknown dataset: {cfg.train.dataset}")
        case "loss_function":
            match cfg.train.loss_function:
                case "cross_entropy":
                    import torch.nn as nn
                    return nn.CrossEntropyLoss
                case _:
                    raise ValueError(f"Unknown loss function: {cfg.train.loss_function}")
        case "train_augmentations":
            default_transforms = train_config_values("model_weights").transforms()

            train_augmentations = default_transforms

            for entry in cfg.train.train_augmentations:
                augmentation_class = None
                match entry["name"]:
                    case "RandomHorizontalFlip":
                        from torchvision.transforms import v2
                        augmentation_class = v2.RandomHorizontalFlip
                    case "RandomVerticalFlip":
                        from torchvision.transforms import v2
                        augmentation_class = v2.RandomVerticalFlip
                    case "RandomRotation":
                        from torchvision.transforms import v2
                        augmentation_class = v2.RandomRotation
                    case "RandomPerspective":
                        from torchvision.transforms import v2
                        augmentation_class = v2.RandomPerspective
                    case "ColorJitter":
                        from torchvision.transforms import v2
                        augmentation_class = v2.ColorJitter
                    case "RandomResizedCrop":
                        from torchvision.transforms import v2
                        augmentation_class = v2.RandomResizedCrop
                    case _:
                        raise ValueError(f"Unknown augmentation: {cfg.train.augmentations}")
                    
                params = {k: v for k, v in entry.items() if k != "name"}
                augmentation = augmentation_class(**params)
                train_augmentations = v2.Compose(
                    [
                        train_augmentations,
                        augmentation,
                    ]
                )
            return train_augmentations
        case "dataset_dir":
            return cfg.train.dataset_dir
        case "num_classes":
            return cfg.train.num_classes
        case "epoch_count":
            return cfg.train.epoch_count
        case "batch_size":
            return cfg.train.batch_size
        case "num_workers":
            return cfg.train.num_workers
        case "learning_rate":
            return cfg.train.learning_rate
        case "use_augmentations":
            return cfg.train.use_augmentations
        case "trained_model_dir":
            return cfg.train.trained_model_dir
        case _:
            raise ValueError(f"Unknown parameter: {param}")
        
@hydra.main(config_path="./", config_name="config") 
def image_based_extract_config_values(cfg: TreespecConfig, param: str):
    match param:
        case "model":
            return cfg.extract.model
        case "output_trees_dir":
            return cfg.extract.output_trees_dir
        case "predict_video_dest_dir":
            return cfg.extract.predict_video_dest_dir
        case "visualize":
            return cfg.extract.visualize
        case "video":
            return cfg.extract.video
        case "corrected":
            return cfg.extract.corrected
        case "image_dir":
            return cfg.extract.image_dir
        case "cameras":
            return cfg.extract.cameras
        case "image_filetype":
            return cfg.extract.image_filetype
        case "predict":
            return cfg.extract.predict
        case "mask":
            return cfg.extract.mask
        case _:
            raise ValueError(f"Unknown parameter: {param}")

@hydra.main(config_path="./", config_name="config")        
def create_essen_dataset_config_values(cfg: TreespecConfig, param: str):
    match param:
        case "original_color_images_path":
            return cfg.essen_dataset.original_color_images_path
        case "color_images_path":
            return cfg.essen_dataset.color_images_path
        case "color_type":
            return cfg.essen_dataset.color_image_type
        case "original_seg_images_path":
            return cfg.essen_dataset.original_segmentation_images_path
        case "segmentid_images_path":
            return cfg.essen_dataset.segmentation_images_path
        case "seg_type":
            return cfg.essen_dataset.segmentation_image_type
        case "seg_output_type":
            return cfg.essen_dataset.segmentation_output_type
        case "run":
            return cfg.essen_dataset.run
        case "output_trees_dir":
            return cfg.essen_dataset.output_trees_dir
        case "attribute_path":
            return cfg.essen_dataset.attribute_path
        case "mask":
            return cfg.essen_dataset.mask
        case "filter":
            return cfg.essen_dataset.filter
        case _:
            raise ValueError(f"Unknown parameter: {param}")