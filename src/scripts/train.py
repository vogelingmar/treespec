from torch import nn
from torchvision.transforms import v2
from torchvision.models import resnet50, ResNet50_Weights
import pytorch_lightning as L
import hydra
from hydra.core.config_store import ConfigStore

from src.models.classification_model import ClassificationModel
from src.datasets.sauen.sauen_dataset import SauenDataset
from src.conf.config import TreespecConfig

cs = ConfigStore.instance()
cs.store(name="treespec_config", node=TreespecConfig)

model_dict = {
    "resnet50": resnet50
}
model_weights_dict = {
    "resnet50_default": ResNet50_Weights.DEFAULT
}
dataset_dict = {
    "sauen": SauenDataset
}
loss_function_dict = {
    "cross_entropy": nn.CrossEntropyLoss
}

@hydra.main(config_path="../conf", config_name="config")
def main(cfg: TreespecConfig):

    default_transforms = model_weights_dict[cfg.TrainParams.model_weights].transforms()
    dataset = dataset_dict[cfg.TrainParams.dataset](batch_size=cfg.TrainParams.batch_size, num_workers=cfg.TrainParams.num_workers)
    dataset.prepare_data()
    dataset.setup(transform=default_transforms)

    model = ClassificationModel(model = model_dict[cfg.TrainParams.model], 
                                model_weights = model_weights_dict[cfg.TrainParams.model_weights], 
                                num_classes = cfg.TrainParams.num_classes,
                                loss_function = loss_function_dict[cfg.TrainParams.loss_function](label_smoothing=0.1, weight=dataset.loss_weights()),
                                learning_rate = cfg.TrainParams.learning_rate)


    trainer = L.Trainer(max_epochs=cfg.TrainParams.epoch_count, log_every_n_steps=10)

    train_augmentations = v2.Compose(
        [
            # v2.RandomResizedCrop(size=(224, 224)),
            # v2.RandomHorizontalFlip(0.2),
            # v2.RandomVerticalFlip(0.4),
            # v2.RandomRotation(15),
            # v2.RandomPerspective(distortion_scale=0.3),
            # v2.ColorJitter(),
            default_transforms,
        ]
    )

    trainer.fit(
        model=model,
        train_dataloaders=dataset.train_dataloader(augmentation=train_augmentations),
        val_dataloaders=dataset.val_dataloader(),
    )

    trainer.test(model=model, dataloaders=dataset.test_dataloader())
    # torch.save(self.model.state_dict(), "io/models/resnet50_finetuned.pth")

    # print(trainer.predict(model=model, dataloaders=dataset.predict_dataloader()))

    model.predict(
        "/home/ingmar/Documents/repos/treespec/src/datasets/sauen/images/sauen_v2/beech/bark_4068_box_00_angle_-5.34.jpg"
    )
    model.predict(
        "/home/ingmar/Documents/repos/treespec/src/datasets/sauen/images/sauen_v2/chestnut/bark_2628_box_00_angle_-3.95.jpg"
    )
    model.predict(
        "/home/ingmar/Documents/repos/treespec/src/datasets/sauen/images/sauen_v2/pine/bark_4308_box_01_angle_9.19.jpg"
    )

if __name__ == "__main__":
    main()