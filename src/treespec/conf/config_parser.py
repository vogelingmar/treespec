"""The config parser for Treespec. It takes the value and the config and returns the translated object or value."""

# pylint: disable=import-outside-toplevel, too-many-return-statements
from treespec.conf.config import TreespecConfig


def train_config_values(  # pylint: disable=too-many-locals
    param: str,
    cfg: TreespecConfig,
):
    r"""Takes a parameter and the config and returns the corresponding value or object.

    Args:
        param: The parameter to extract.
        cfg: The TreespecConfig object containing the configuration.

    Returns:
        The value or object corresponding to the parameter.

    Raises:
        ValueError: If the parameter is unknown or not supported.
    """

    match param:
        case "model":
            match cfg.train.model:
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
        case "dataset_dir":
            return cfg.train.dataset_dir
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
        case "use_augmentations":
            return cfg.train.use_augmentations
        case "trained_model_dir":
            return cfg.train.trained_model_dir
        case _:
            raise ValueError(f"Unknown parameter: {param}")


def create_essen_dataset_config_values(param: str, cfg: TreespecConfig):
    r"""Takes a parameter and the config and returns the corresponding value or object.

    Args:
        param: The parameter to extract.
        cfg: The TreespecConfig object containing the configuration.

    Returns:
        The value or object corresponding to the parameter.

    Raises:
        ValueError: If the parameter is unknown or not supported.
    """
    match param:
        case "original_color_images_path":
            return cfg.essen_dataset.original_color_images_path
        case "color_images_path":
            return cfg.essen_dataset.color_images_path
        case "color_type":
            return cfg.essen_dataset.color_type
        case "color_output_type":
            return cfg.essen_dataset.color_output_type
        case "original_id_images_path":
            return cfg.essen_dataset.original_id_images_path
        case "segmentid_images_path":
            return cfg.essen_dataset.segmentid_images_path
        case "seg_type":
            return cfg.essen_dataset.seg_type
        case "seg_output_type":
            return cfg.essen_dataset.seg_output_type
        case "original_sem_images_path":
            return cfg.essen_dataset.original_sem_images_path
        case "semantic_images_path":
            return cfg.essen_dataset.semantic_images_path
        case "sem_type":
            return cfg.essen_dataset.sem_type
        case "sem_output_type":
            return cfg.essen_dataset.sem_output_type
        case "run":
            return cfg.essen_dataset.run
        case "output_trees_dir":
            return cfg.essen_dataset.output_trees_dir
        case "attribute_path":
            return cfg.essen_dataset.attribute_path
        case "mask":
            return cfg.essen_dataset.mask
        case "filter_id":
            return cfg.essen_dataset.filter_id
        case "filter_semantic":
            return cfg.essen_dataset.filter_semantic
        case "apply_center_crop":
            return cfg.essen_dataset.crop
        case "pictures_extracted":
            return cfg.essen_dataset.pictures_extracted
        case _:
            raise ValueError(f"Unknown parameter: {param}")


def predict_essen_config_values(param: str, cfg: TreespecConfig):
    r"""Takes a parameter and the config and returns the corresponding value or object.

    Args:
        param: The parameter to extract.
        cfg: The TreespecConfig object containing the configuration.

    Returns:
        The value or object corresponding to the parameter.

    Raises:
        ValueError: If the parameter is unknown or not supported.
    """
    match param:
        case "tree_images_dir":
            return cfg.predict_essen.tree_images_dir
        case "input_inventory_path":
            return cfg.predict_essen.input_inventory_path
        case "output_inventory_path":
            return cfg.predict_essen.output_inventory_path
        case "trained_model_path":
            return cfg.predict_essen.trained_model_path
        case _:
            raise ValueError(f"Unknown parameter: {param}")


def matching_config_values(
    param: str,
    cfg: TreespecConfig,
):
    r"""Takes a parameter and the config and returns the corresponding value or object.

    Args:
        param: The parameter to extract.
        cfg: The TreespecConfig object containing the configuration.

    Returns:
        The value or object corresponding to the parameter.

    Raises:
        ValueError: If the parameter is unknown or not supported.
    """
    match param:
        case "predicted_cadastre_path":
            return cfg.matching.predicted_cadastre_path
        case "cadastre_path":
            return cfg.matching.cadastre_path
        case "output_path":
            return cfg.matching.output_path
        case "use_dbh_filter":
            return cfg.matching.use_dbh_filter
        case _:
            raise ValueError(f"Unknown parameter: {param}")
