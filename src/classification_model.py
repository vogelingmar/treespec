"""Classification Model using ResNet50 to classify the barks from the Sauen Dataset."""

from torchvision.io import decode_image
from torchvision.models import resnet50, ResNet50_Weights
import torch
from torch import nn
import pytorch_lightning as L
import torchmetrics

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
        loss_function: int = nn.CrossEntropyLoss,
        learning_rate: float = 0.001,
    ):
        super(ClassificationModel, self).__init__()
        self.model_weights = model_weights
        self.model = model(weights=self.model_weights)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        self.learning_rate = learning_rate
        self.loss_function = loss_function

        self.accuracy = torchmetrics.Accuracy(
            num_classes=num_classes, task="multiclass"
        )
        self.f1 = torchmetrics.F1Score(num_classes=num_classes, task="multiclass")
        self.precision = torchmetrics.Precision(
            num_classes=num_classes, task="multiclass"
        )
        self.recall = torchmetrics.Recall(num_classes=num_classes, task="multiclass")

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

    def calculate_metrics(self, outputs, labels):
        r"""
        The function calculating the metrics of the classification model.

        Args:
            outputs: The output of the model.
            labels: The labels of the data.

        Returns:
            accuracy, f1, precision, recall

        Shape:
            - :code:`outputs`: idk
            - :code:`labels`: idk
            - Output: idk

        """
        accuracy = self.accuracy(outputs, labels)
        f1 = self.f1(outputs, labels)
        precision = self.precision(outputs, labels)
        recall = self.recall(outputs, labels)

        return accuracy, f1, precision, recall

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
        outputs = self.forward(inputs)
        loss = self.loss_function(outputs, labels)

        accuracy, f1, precision, recall = self.calculate_metrics(outputs, labels)
        self.log_dict(
            {
                "train_loss": loss,
                "train_accuracy": accuracy,
                "train_f1": f1,
                "train_precision": precision,
                "train_recall": recall,
            },
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )

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
        outputs = self.forward(inputs)
        val_loss = self.loss_function(outputs, labels)
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
        outputs = self.forward(inputs)
        test_loss = self.loss_function(outputs, labels)
        self.log("test_loss", test_loss)

    def configure_optimizers(self):
        r"""
        The function describing the optimizer of the classification model.
        Returns:
            The optimizer to be used for training.
        """
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

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
