import pytorch_lightning as L
from torchvision.transforms import v2
from torch import nn

from torchvision.models import resnet50, ResNet50_Weights

from classification_model import ClassificationModel
from datasets.sauen.sauen_dataset import Sauen_Dataset

transforms = ResNet50_Weights.DEFAULT.transforms()
dataset = Sauen_Dataset()
dataset.prepare_data()
dataset.setup(transforms=transforms)

model = ClassificationModel(
    loss_function=nn.CrossEntropyLoss(
        label_smoothing=0.1, weight=dataset.loss_weights()
    )
)


trainer = L.Trainer(max_epochs=15, log_every_n_steps=10)

train_transforms = v2.Compose(
    [
        v2.RandomResizedCrop(size=(224, 224)),
        v2.RandomHorizontalFlip(0.2),
        v2.RandomVerticalFlip(0.4),
        v2.RandomRotation(15),
        v2.RandomPerspective(distortion_scale=0.3),
        v2.ColorJitter(),
    ]
)

trainer.fit(
    model=model,
    train_dataloaders=dataset.train_dataloader(),
    val_dataloaders=dataset.val_dataloader(),
)

trainer.test(model=model, dataloaders=dataset.test_dataloader())
# torch.save(self.model.state_dict(), "io/models/resnet50_finetuned.pth")

model.predict(
    "/home/ingmar/Documents/repos/treespec/src/datasets/sauen/images/sauen_v2/beech/bark_4068_box_00_angle_-5.34.jpg"
)
model.predict(
    "/home/ingmar/Documents/repos/treespec/src/datasets/sauen/images/sauen_v2//chestnut/bark_2628_box_00_angle_-3.95.jpg"
)
model.predict(
    "/home/ingmar/Documents/repos/treespec/src/datasets/sauen/images/sauen_v2/pine/bark_4308_box_01_angle_9.19.jpg"
)
