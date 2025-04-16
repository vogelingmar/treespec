"""Training Script of the Treespec Pipeline"""

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

from src.models.classification_model import ClassificationModel
from src.datasets.sauen.sauen_dataset import SauenDataset
from src.conf.config import TreespecConfig

cs = ConfigStore.instance()
cs.store(name="treespec_config", node=TreespecConfig)


@hydra.main(config_path="../conf", config_name="config")
def main(cfg: TreespecConfig):

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

    default_transforms = model_weights_dict[cfg.TrainParams.model_weights].transforms()
    # TODO: experiment with data augmentations
    if cfg.TrainParams.use_augmentations is True:
        train_augmentations = v2.Compose(
            [
                v2.RandomResizedCrop(size=(224, 224)),
                v2.RandomHorizontalFlip(0.2),
                v2.RandomVerticalFlip(0.4),
                v2.RandomRotation(15),
                v2.RandomPerspective(distortion_scale=0.3),
                v2.ColorJitter(),
                default_transforms,
            ]
        )
    else:
        train_augmentations = default_transforms

    dataset = dataset_dict[cfg.TrainParams.dataset](
        batch_size=cfg.TrainParams.batch_size, num_workers=cfg.TrainParams.num_workers
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
    # torch.save(self.model.state_dict(), "io/models/resnet50_finetuned.pth")


if __name__ == "__main__":
    main()
