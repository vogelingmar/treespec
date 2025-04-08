"""Classification Model using ResNet50 to classify the barks from the Sauen Dataset."""

from torchvision.io import decode_image
from torchvision.models import resnet50, ResNet50_Weights
import torch
from torch import nn
import pytorch_lightning as L

from datasets.sauen.sauen_dataset import Sauen_Dataset


class ClassificationModel(L.LightningModule):
    r"""
    The tree species classification model of the treespec pipeline.

    Args:
        model_weights: The weights to be used to initialize the model.
        model: The model to be used for classification.
        num_classes: The number of classes to be differentiated by the model.
        dataset: The dataset to be used for the model.
        loss_function: The loss function to be used for training.
        learning_rate: The learning rate to be used for training.
        batch_size: The batch size to be used for training.
    """

    def __init__(
        self,
        model_weights: int = ResNet50_Weights.DEFAULT,  # change type
        model: int = resnet50,  # change type
        num_classes: int = 3,
        dataset: L.LightningDataModule = Sauen_Dataset,
        loss_function: int = nn.CrossEntropyLoss,
        learning_rate: float = 0.001,
        batch_size: int = 5,
    ):

        super(ClassificationModel, self).__init__()
        self.model_weights = model_weights
        self.model = model(weights=self.model_weights)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        #self.dataset = dataset()
        #self.dataset.setup(
        #    transforms=self.model_weights.transforms(), batch_size=batch_size
        #)
        #self.criterion = loss_function(
        #    label_smoothing=0.1, weight=self.dataset.loss_weights()
        #)
        self.learning_rate = learning_rate

    def forward(self, x):
        r"""
        The forward method of the classification model.

        Args:
            x: Input tensor

        Returns:
            Output tensor

        Shape:
            - :code:`x`: idk
            - Output: idk
        """
        return self.model(x)

    def training_step(self, batch, batch_idx):
        r"""
        The function describing the training step of the classification model.

        Args: 
            batch: The batch of data to be used for training.
            batch_idx: The index of the batch.
        
        Returns:
            The loss of the model from the training step.

        Shape:
            - :code:`batch`: :math:`(I_k, L_k)`
            - :code:`batch_idx`: b
            - Output: idk

            | where
            |
            | :math:`I_k = \text{ k-th input image of the batch encoded as tensor}`
            | :math:`L_k = \text{ k-th class index of the k-th input index}`

        """
        inputs, labels = batch
        outputs = self.model(inputs)
        loss = self.criterion(outputs, labels)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        r"""
        The function describing the validation step of the classification model.

        Args: 
            batch: The batch of data to be used for training.
            batch_idx: The index of the batch.

        Shape:
            - :code:`batch`: :math:`(I_k, L_k)`
            - :code:`batch_idx`: b

            | where
            |
            | :math:`I_k = \text{ k-th input image of the batch encoded as tensor}`
            | :math:`L_k = \text{ k-th class index of the k-th input index}`

        """
        inputs, labels = batch
        outputs = self.model(inputs)
        val_loss = self.criterion(outputs, labels)
        self.log("val_loss", val_loss)

    def test_step(self, batch, batch_idx):
        r"""
        The function describing the validation step of the classification model.

        Args: 
            batch: The batch of data to be used for training.
            batch_idx: The index of the batch.

        Shape:
            - :code:`batch`: :math:`(I_k, L_k)`
            - :code:`batch_idx`: b

            | where
            |
            | :math:`I_k = \text{ k-th input image of the batch encoded as tensor}`
            | :math:`L_k = \text{ k-th class index of the k-th input index}`

        """
        inputs, labels = batch
        outputs = self.model(inputs)
        test_loss = self.criterion(outputs, labels)
        self.log("test_loss", test_loss)

    def configure_optimizers(self):
        r"""
        The function describing the optimizer of the classification model.
        Returns:
            The optimizer to be used for training.
        """
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    # TODO: has to be changed
    def pumpen(self):
        trainer = L.Trainer(max_epochs=15)

        trainer.fit(
            model=self,
            train_dataloaders=self.dataset.train_dataloader(),
            val_dataloaders=self.dataset.val_dataloader(),
        )

        trainer.test(model=self, dataloaders=self.dataset.test_dataloader())
        # torch.save(self.model.state_dict(), "io/models/resnet50_finetuned.pth")

#= "/home/ingmar/Documents/repos/treespec/src/io/datasets/sauen
# /beech/bark_4116_box_00_angle_-6.11.jpg"
    def predict(
        self,
        img: str,
    ):

        picture = decode_image(img)
        batch = self.model_weights.transforms()(picture).unsqueeze(0)

        self.model.eval()
        with torch.no_grad():
            prediction = self.forward(batch).squeeze(0).softmax(0)
            class_id = prediction.argmax().item()
            score = prediction[class_id].item()

        custom_categories = ["beech", "chestnut", "pine"]
        category_name = custom_categories[class_id]

        print(f"{category_name}: {100 * score:.1f}%")
