"""Training Script of the Treespec Pipeline"""

import torch
from torch import nn
from torchvision.transforms import v2  # type: ignore
from torchvision.models import (  # type: ignore
    resnet50,
    ResNet50_Weights,
    resnet152,
    ResNet152_Weights,
    swin_v2_b,
    Swin_V2_B_Weights,
    efficientnet_v2_m,
    EfficientNet_V2_M_Weights,
    googlenet,
    GoogLeNet_Weights,
    mobilenet_v3_large,
    MobileNet_V3_Large_Weights,
    wide_resnet101_2,
    Wide_ResNet101_2_Weights,
)
import pytorch_lightning as L
import hydra
from hydra.core.config_store import ConfigStore

from treespec.models.classification_model import ClassificationModel
from treespec.datasets.sauen.sauen_dataset import SauenDataset
from treespec.conf.config import TreespecConfig

cs = ConfigStore.instance()
cs.store(name="treespec_config", node=TreespecConfig)

model_dict = {
    "resnet50": resnet50,
    "resnet152": resnet152,
    "swin_transformer": swin_v2_b,
    "efficientnet": efficientnet_v2_m,
    "googlenet": googlenet,
    "mobilenet": mobilenet_v3_large,
    "wide_resnet": wide_resnet101_2,
}
model_weights_dict = {
    "resnet50_default": ResNet50_Weights.DEFAULT,
    "resnet152_default": ResNet152_Weights.DEFAULT,
    "swin_default": Swin_V2_B_Weights.DEFAULT,
    "efficientnet_default": EfficientNet_V2_M_Weights.DEFAULT,
    "googlenet_default": GoogLeNet_Weights.DEFAULT,
    "mobilenet_default": MobileNet_V3_Large_Weights.DEFAULT,
    "wide_resnet_default": Wide_ResNet101_2_Weights.DEFAULT,

}
dataset_dict = {
    "sauen": SauenDataset,
}
loss_function_dict = {
    "cross_entropy": nn.CrossEntropyLoss,
}
augmentations_dict = {
    "RandomHorizontalFlip": v2.RandomHorizontalFlip,
    "RandomVerticalFlip": v2.RandomVerticalFlip,
    "RandomRotation": v2.RandomRotation,
    "RandomPerspective": v2.RandomPerspective,
    "ColorJitter": v2.ColorJitter,
    "RandomResizedCrop": v2.RandomResizedCrop,
}


@hydra.main(config_path="../conf", config_name="config")
def main(cfg: TreespecConfig):
    """Training Script of the Treespec Pipeline"""

    default_transforms = model_weights_dict[cfg.train.model_weights].transforms()

    train_augmentations = default_transforms

    if cfg.train.use_augmentations is True:

        for entry in cfg.train.train_augmentations:
            augmentation_class = augmentations_dict[entry["name"]]
            params = {k: v for k, v in entry.items() if k != "name"}
            augmentation = augmentation_class(**params)
            train_augmentations = v2.Compose(
                [
                    train_augmentations,
                    augmentation,
                ]
            )

    dataset = dataset_dict[cfg.train.dataset](
        data_dir=cfg.train.dataset_dir,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
    )
    dataset.prepare_data()
    dataset.setup(transform=default_transforms)

    loss_function = loss_function_dict[cfg.train.loss_function](label_smoothing=0.1, weight=dataset.loss_weights())

    model = ClassificationModel(
        model=model_dict[cfg.train.model],
        model_weights=model_weights_dict[cfg.train.model_weights],
        num_classes=cfg.train.num_classes,
        loss_function=loss_function,
        learning_rate=cfg.train.learning_rate,
    )

    trainer = L.Trainer(max_epochs=cfg.train.epoch_count, log_every_n_steps=10)

    trainer.fit(
        model=model,
        train_dataloaders=dataset.train_dataloader(augmentation=train_augmentations),
        val_dataloaders=dataset.val_dataloader(),
    )

    trainer.test(model=model, dataloaders=dataset.test_dataloader())
    torch.save(
        model.model.state_dict(),
        (cfg.train.trained_model_dir + cfg.train.model + "_finetuned" + ".pth"),
    )


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
