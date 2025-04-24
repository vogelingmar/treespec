"""Training Script of the Treespec Pipeline"""

import torch
from torch import nn
from torchvision.transforms import v2
from torchvision.models import (
    resnet50,
    resnet152,
    ResNet50_Weights,
    ResNet152_Weights,
    swin_v2_b,
    Swin_V2_B_Weights,
)
import pytorch_lightning as L
import hydra
from hydra.core.config_store import ConfigStore

from src.treespec.models.classification_model import ClassificationModel
from src.treespec.datasets.sauen.sauen_dataset import SauenDataset
from src.treespec.conf.config import TreespecConfig

cs = ConfigStore.instance()
cs.store(name="treespec_config", node=TreespecConfig)

model_dict = {
    "resnet50": resnet50,
    "resnet152": resnet152,
    "swin_transformer": swin_v2_b,
}
model_weights_dict = {
    "resnet50_default": ResNet50_Weights.DEFAULT,
    "resnet152_default": ResNet152_Weights.DEFAULT,
    "swin_default": Swin_V2_B_Weights.DEFAULT,
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

    default_transforms = model_weights_dict[cfg.TrainParams.model_weights].transforms()
    # TODO: experiment with data augmentations

    train_augmentations = default_transforms

    if cfg.TrainParams.use_augmentations is True:

        for entry in cfg.TrainParams.train_augmentations:
            augmentation_class = augmentations_dict[entry['name']]
            params = {k: v for k, v in entry.items() if k != 'name'}
            augmentation = augmentation_class(**params)
            train_augmentations = v2.Compose(
                [
                train_augmentations,
                augmentation,
                ]
            )

    dataset = dataset_dict[cfg.TrainParams.dataset](
        data_dir=cfg.TrainParams.dataset_dir,
        batch_size=cfg.TrainParams.batch_size,
        num_workers=cfg.TrainParams.num_workers,
    )
    dataset.prepare_data()
    dataset.setup(transform=default_transforms)

    model = ClassificationModel(
        model=model_dict[cfg.TrainParams.model],
        model_weights=model_weights_dict[cfg.TrainParams.model_weights],
        num_classes=cfg.TrainParams.num_classes,
        loss_function=loss_function_dict[cfg.TrainParams.loss_function](
            label_smoothing=0.1, weight=dataset.loss_weights()
        ),
        learning_rate=cfg.TrainParams.learning_rate,
    )

    trainer = L.Trainer(max_epochs=cfg.TrainParams.epoch_count, log_every_n_steps=10)

    trainer.fit(
        model=model,
        train_dataloaders=dataset.train_dataloader(augmentation=train_augmentations),
        val_dataloaders=dataset.val_dataloader(),
    )

    trainer.test(model=model, dataloaders=dataset.test_dataloader())
    torch.save(
        model.model.state_dict(),
        (cfg.TrainParams.trained_model_dir + cfg.TrainParams.model + "_finetuned" + ".pth"),
    )


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
