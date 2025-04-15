"""Classification Model to classify the barks from the Sauen Dataset."""

from typing import Callable, Optional

import torch
from torch import nn
from torch.nn.modules.loss import _Loss
from torchvision.io import decode_image  # type: ignore
from torchvision.models import resnet50  # ignore: import-untyped
from torchvision.models._api import WeightsEnum  # ignore: import-untyped
import torchmetrics

import pytorch_lightning as L


class ClassificationModel(L.LightningModule):  # pylint: disable=too-many-instance-attributes
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

    def __init__(  # pylint: disable=too-many-arguments, too-many-positional-arguments
        self,
        model_weights: Optional[WeightsEnum],
        model: Callable = resnet50,
        num_classes: int = 3,
        loss_function: _Loss = nn.CrossEntropyLoss,
        learning_rate: float = 0.001,
    ):
        super().__init__()
        self.model_weights = model_weights
        self.model = model(weights=self.model_weights)

        # Modify classification head based on model architecture
        if hasattr(self.model, "fc"):  # For ResNet-like models
            self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        elif hasattr(self.model, "head"):  # For Swin Transformer-like models
            self.model.head = nn.Linear(self.model.head.in_features, num_classes)
        else:
            raise AttributeError("The model does not have a recognized classification head.")

        self.loss_function = loss_function
        self.learning_rate = learning_rate

        self.accuracy = torchmetrics.Accuracy(num_classes=num_classes, task="multiclass")
        self.f1 = torchmetrics.F1Score(num_classes=num_classes, task="multiclass")
        self.precision = torchmetrics.Precision(num_classes=num_classes, task="multiclass")
        self.recall = torchmetrics.Recall(num_classes=num_classes, task="multiclass")

    def forward(self, x: torch.Tensor):
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
    
    def calculate_per_class_metrics(self, outputs: torch.Tensor, labels: torch.Tensor):
        r"""
        Calculate TP, FP, FN, TN, precision, recall, and F1-score per class.

        Args:
            outputs: The output predictions of the model.
            labels: The ground truth labels.

        Returns:
            A dictionary containing per-class metrics.
        """
        # Compute confusion matrix
        confusion_matrix = self.confusion_matrix(outputs, labels)

        # Extract TP, FP, FN, TN per class
        tp = torch.diag(confusion_matrix)
        fp = confusion_matrix.sum(dim=0) - tp
        fn = confusion_matrix.sum(dim=1) - tp
        tn = confusion_matrix.sum() - (tp + fp + fn)

        # Compute precision, recall, and F1-score per class
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1_score = 2 * (precision * recall) / (precision + recall + 1e-8)

        return {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
        }

    def calculate_metrics(self, outputs: torch.Tensor, labels: torch.Tensor):
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

    def training_step(self, batch: list, batch_idx: int):
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

        per_class_metrics = self.calculate_per_class_metrics(outputs, labels)

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

        # Log per-class metrics
        for i, (precision, recall, f1) in enumerate(
            zip(
                per_class_metrics["precision"],
                per_class_metrics["recall"],
                per_class_metrics["f1_score"],
            )
        ):
            self.log(f"val_precision_class_{i}", precision)
            self.log(f"val_recall_class_{i}", recall)
            self.log(f"val_f1_class_{i}", f1)

        return loss

    def validation_step(self, batch: list, batch_idx: int):
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

        per_class_metrics = self.calculate_per_class_metrics(outputs, labels)

        accuracy, f1, precision, recall = self.calculate_metrics(outputs, labels)
        self.log_dict(
            {
                "val_loss": val_loss,
                "val_accuracy": accuracy,
                "val_f1": f1,
                "val_precision": precision,
                "val_recall": recall,
            },
        )

         # Log per-class metrics
        for i, (precision, recall, f1) in enumerate(
            zip(
                per_class_metrics["precision"],
                per_class_metrics["recall"],
                per_class_metrics["f1_score"],
            )
        ):
            self.log(f"val_precision_class_{i}", precision)
            self.log(f"val_recall_class_{i}", recall)
            self.log(f"val_f1_class_{i}", f1)

    def test_step(self, batch: list, batch_idx: int):
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

        per_class_metrics = self.calculate_per_class_metrics(outputs, labels)

        accuracy, f1, precision, recall = self.calculate_metrics(outputs, labels)
        self.log_dict(
            {
                "test_loss": test_loss,
                "test_accuracy": accuracy,
                "test_f1": f1,
                "test_precision": precision,
                "test_recall": recall,
            },
        )

         # Log per-class metrics
        for i, (precision, recall, f1) in enumerate(
            zip(
                per_class_metrics["precision"],
                per_class_metrics["recall"],
                per_class_metrics["f1_score"],
            )
        ):
            self.log(f"val_precision_class_{i}", precision)
            self.log(f"val_recall_class_{i}", recall)
            self.log(f"val_f1_class_{i}", f1)

    def predict_step(self, batch: list, batch_idx: int):

        pass

    def configure_optimizers(self):
        r"""
        The function describing the optimizer of the classification model.
        Returns:
            The optimizer to be used for training.
        """

        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def predict(
        self,
        img_path: str,
    ):
        r"""
        The predict function of the classification model. 
        Input is a path to an image thats class should be predicted.
        """

        picture = decode_image(img_path)
        batch = self.model_weights.transforms()(picture).unsqueeze(0)

        self.model.eval()
        with torch.no_grad():
            prediction = self.forward(batch).squeeze(0).softmax(0)
            class_id = prediction.argmax().item()
            score = prediction[class_id].item()

        custom_categories = ["beech", "chestnut", "pine"]
        category_name = custom_categories[class_id]

        print(f"{category_name}: {100 * score:.1f}%")
