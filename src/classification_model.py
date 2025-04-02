from torchvision.io import decode_image
from torchvision.models import resnet50, ResNet50_Weights
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as L

from sauen_dataset import Sauen_Dataset

class ClassificationModel(L.LightningModule):

    def __init__(self, num_classes: int = 3, dataset: L.LightningDataModule = Sauen_Dataset()):
        super(ClassificationModel, self).__init__()
        self.weights = ResNet50_Weights.DEFAULT
        self.model = resnet50(weights=self.weights)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        self.dataset = dataset
        self.dataset.setup(transforms=self.weights.transforms())

    def train(self):
        self.model.train()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        num_epochs = 10
        for epoch in range(num_epochs):
            self.model.train()
            running_loss = 0.0
            for inputs, labels in self.dataset.train_dataloader():
            
                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)

                # Backward pass and optimization
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(self.dataset.train_dataloader())}")

            # Validation loop
            self.model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for inputs, labels in self.dataset.val_dataloader():
                    outputs = self.model(inputs)
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            print(f"Validation Accuracy: {100 * correct / total:.2f}%")

        # make one final test
        # Save the fine-tuned model
        torch.save(self.model.state_dict(), "resnet50_finetuned.pth")

    def predict(self, img: str = "/home/ingmar/Documents/repos/PercepTree/PercepTreeV1/output/bark_screenshots/beech/bark_4116_box_00_angle_-6.11.jpg"):

        picture = decode_image(img)
        batch = self.weights.transforms()(picture).unsqueeze(0)

        self.model.eval()
        with torch.no_grad():
            prediction = self.model(batch).squeeze(0).softmax(0)
            class_id = prediction.argmax().item()
            score = prediction[class_id].item()

        custom_categories = ["beech", "chestnut", "pine"]
        category_name = custom_categories[class_id]

        print(f"{category_name}: {100 * score:.1f}%")