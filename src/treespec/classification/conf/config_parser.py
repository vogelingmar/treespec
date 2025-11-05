"""The config parser for the treespec classification module. It extracts values or objects based on the provided parameter and configuration."""

from typing import Optional
from pathlib import Path
# pylint: disable=import-outside-toplevel, too-many-return-statements
from treespec.classification.conf.config import ClassificationConfig


def train_config_values(  # pylint: disable=too-many-locals
    param: str,
    cfg: ClassificationConfig,
) -> Optional[any]:
    r"""Extracts the value or object corresponding to a training parameter from the configuration.

    Args:
        param: The training parameter to extract.
        cfg: The configuration object containing training settings.

    Returns:
        Any: The value or object corresponding to the specified training parameter.

    Raises:
        ValueError: If the parameter is unknown or not supported.
    """

    match param:
        case "model":
            match cfg.train.model:
                case "resnet18":
                    from torchvision.models import resnet18  # type: ignore

                    return resnet18
                case "resnet50":
                    from torchvision.models import resnet50  # type: ignore

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
                case "mobilenet_v2":
                    from torchvision.models import mobilenet_v2

                    return mobilenet_v2
                case "mobilenet_v3":
                    from torchvision.models import mobilenet_v3_large

                    return mobilenet_v3_large
                case "wide_resnet":
                    from torchvision.models import wide_resnet101_2

                    return wide_resnet101_2
                case "convnext":
                    from torchvision.models import convnext_base

                    return convnext_base
                case "alexnet":
                    from torchvision.models import alexnet

                    return alexnet
                case "vgg16":
                    from torchvision.models import vgg16

                    return vgg16
                case "densenet":
                    from torchvision.models import densenet121

                    return densenet121
                case _:
                    raise ValueError(f"Unknown model: {cfg.train.model}")
        case "model_weights":
            match cfg.train.model_weights:
                case "resnet18_default":
                    from torchvision.models import ResNet18_Weights  # type: ignore

                    return ResNet18_Weights.DEFAULT
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
                case "mobilenet_v2_default":
                    from torchvision.models import MobileNet_V2_Weights

                    return MobileNet_V2_Weights.DEFAULT
                case "mobilenet_v3_default":
                    from torchvision.models import MobileNet_V3_Large_Weights

                    return MobileNet_V3_Large_Weights.DEFAULT
                case "wide_resnet_default":
                    from torchvision.models import Wide_ResNet101_2_Weights

                    return Wide_ResNet101_2_Weights.DEFAULT
                case "convnext_default":
                    from torchvision.models import ConvNeXt_Base_Weights

                    return ConvNeXt_Base_Weights.DEFAULT
                case "alexnet_default":
                    from torchvision.models import AlexNet_Weights

                    return AlexNet_Weights.DEFAULT
                case "vgg16_default":
                    from torchvision.models import VGG16_Weights

                    return VGG16_Weights.DEFAULT
                case "densenet_default":
                    from torchvision.models import DenseNet121_Weights

                    return DenseNet121_Weights.DEFAULT
                case _:
                    raise ValueError(f"Unknown model weights: {cfg.train.model_weights}")
        case "dataset":
            match cfg.train.dataset:
                case "folder":
                    from treespec.classification.image_dataset import ImageDataset

                    return ImageDataset
                case _:
                    raise ValueError(f"Unknown dataset: {cfg.train.dataset}")
        case "loss_function":
            match cfg.train.loss_function:
                case "cross_entropy":
                    from torch import nn

                    return nn.CrossEntropyLoss
                case _:
                    raise ValueError(f"Unknown loss function: {cfg.train.loss_function}")
        case "train_augmentations":
            default_transforms = train_config_values("model_weights", cfg).transforms()

            train_augmentations = default_transforms

            for entry in cfg.train.train_augmentations:
                augmentation_class = None
                match entry["name"]:
                    case "RandomHorizontalFlip":
                        from torchvision.transforms import v2  # type: ignore

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
                    case "ElasticTransform":
                        from torchvision.transforms import v2

                        augmentation_class = v2.ElasticTransform
                    case _:
                        raise ValueError(f"Unknown augmentation: {entry['name']}")

                params = {k: v for k, v in entry.items() if k != "name"}
                augmentation = augmentation_class(**params)
                train_augmentations = v2.Compose(
                    [
                        train_augmentations,
                        augmentation,
                    ]
                )
            return train_augmentations
        case "dataset_dir_path":
            return cfg.train.dataset_dir_path
        case "num_classes":
            return cfg.train.num_classes
        case "use_ids":
            return cfg.train.use_ids
        case "epoch_count":
            return cfg.train.epoch_count
        case "batch_size":
            return cfg.train.batch_size
        case "num_workers":
            return cfg.train.num_workers
        case "learning_rate":
            return cfg.train.learning_rate
        case "trained_model_dir_path":
            return cfg.train.trained_model_dir_path
        case "trained_model_path":
            return cfg.train.trained_model_path
        case _:
            raise ValueError(f"Unknown parameter: {param}")


def predict_config_values(param: str, cfg: ClassificationConfig) -> Optional[Path]:
    r"""Extracts the value or object corresponding to a prediction parameter from the configuration.

    Args:
        param (str): The prediction parameter to extract.
        cfg (ClassificationConfig): The configuration object containing prediction settings.

    Returns:
        Any: The value or object corresponding to the specified prediction parameter.

    Raises:
        ValueError: If the parameter is unknown or not supported.
    """
    match param:
        case "model":
            return train_config_values("model", cfg)
        case "model_weights":
            return train_config_values("model_weights", cfg)
        case "dataset_dir_path":
            return train_config_values("dataset_dir_path", cfg)
        case "tree_images_dir_path":
            return cfg.predict.tree_images_dir_path
        case "input_inventory_path":
            return cfg.predict.input_inventory_path
        case "output_inventory_path":
            return cfg.predict.output_inventory_path
        case "trained_model_path":
            return cfg.predict.trained_model_path
        case _:
            raise ValueError(f"Unknown parameter: {param}")
