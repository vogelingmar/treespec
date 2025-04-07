""" Classification Model using ResNet50 to classify the barks from the Sauen Dataset. """

from torchvision.io import decode_image
from torchvision.models import resnet50, ResNet50_Weights
import torch
import torch.nn as nn
import pytorch_lightning as L

from datasets.sauen_dataset import Sauen_Dataset

class ClassificationModel(L.LightningModule):

    def __init__(self, num_classes: int = 3, dataset: L.LightningDataModule = Sauen_Dataset()):
        super(ClassificationModel, self).__init__()
        self.weights = ResNet50_Weights.DEFAULT
        self.model = resnet50(weights=self.weights)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        self.dataset = dataset
        self.dataset.setup(transforms=self.weights.transforms())
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self.model(inputs)
        loss = self.criterion(outputs, labels)
        self.log("train_loss", loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self.model(inputs)
        val_loss = self.criterion(outputs, labels)
        self.log("val_loss", val_loss)

    def test_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self.model(inputs)
        test_loss = self.criterion(outputs, labels)
        self.log("test_loss", test_loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)
    
    # TODO: has to be changed
    def pumpen(self):
        trainer = L.Trainer(max_epochs=15)

        trainer.fit(model = self, train_dataloaders=self.dataset.train_dataloader(), val_dataloaders=self.dataset.val_dataloader())

        trainer.test(model=self, dataloaders=self.dataset.test_dataloader())
        #torch.save(self.model.state_dict(), "io/models/resnet50_finetuned.pth")

    def predict(self, img: str = "/home/ingmar/Documents/repos/treespec/src/io/datasets/sauen/beech/bark_4116_box_00_angle_-6.11.jpg"):

        picture = decode_image(img)
        batch = self.weights.transforms()(picture).unsqueeze(0)

        self.model.eval()
        with torch.no_grad():
            prediction = self.forward(batch).squeeze(0).softmax(0)
            class_id = prediction.argmax().item()
            score = prediction[class_id].item()

        custom_categories = ["beech", "chestnut", "pine"]
        category_name = custom_categories[class_id]

        print(f"{category_name}: {100 * score:.1f}%")

