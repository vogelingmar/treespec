import pytorch_lightning as L
from torchvision.transforms import v2
from torch import nn

from torchvision.models import resnet50, ResNet50_Weights

from classification_model import ClassificationModel
from datasets.sauen.sauen_dataset import Sauen_Dataset
import config

default_transforms = ResNet50_Weights.DEFAULT.transforms()
dataset = Sauen_Dataset()
dataset.prepare_data()
dataset.setup(transform=default_transforms)

model = ClassificationModel(loss_function=nn.CrossEntropyLoss(label_smoothing=0.1, weight=dataset.loss_weights()))


trainer = L.Trainer(max_epochs=15, log_every_n_steps=10)

train_augmentations = v2.Compose(
    [
        # v2.RandomResizedCrop(size=(224, 224)),
        # v2.RandomHorizontalFlip(0.2),
        # v2.RandomVerticalFlip(0.4),
        # v2.RandomRotation(15),
        # v2.RandomPerspective(distortion_scale=0.3),
        # v2.ColorJitter(),
        v2.ToTensor(),
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
