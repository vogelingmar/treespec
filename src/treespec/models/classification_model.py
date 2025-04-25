"""Classification Model to classify the barks from the Sauen Dataset."""

from typing import Callable

import torch
from torch import nn
from torch.nn.modules.loss import _Loss
from torchvision.io import decode_image  # type: ignore
from torchvision.models._api import WeightsEnum  # type: ignore
import torchmetrics
from torchmetrics import ConfusionMatrix

import pytorch_lightning as L


class ClassificationModel(L.LightningModule):  # pylint: disable=too-many-instance-attributes
    r"""
    The tree species classification model of the treespec pipeline.

    Args:
        model_weights: The weights to be used to initialize the model.
        model: The model to be used for classification.
        num_classes: The number of classes to be differentiated by the model.
        loss_function: The loss function to be used for training.
        learning_rate: The learning rate to be used for training.
    """

    def __init__(  # pylint: disable=too-many-arguments, too-many-positional-arguments
        self,
        model_weights: WeightsEnum,
        model: Callable,
        num_classes: int,
        loss_function: _Loss,
        learning_rate: float,
    ):
        super().__init__()
        self.model_weights = model_weights
        self.model = model(weights=self.model_weights)

        if hasattr(self.model, "fc"):
            self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        elif hasattr(self.model, "head"):
            self.model.head = nn.Linear(self.model.head.in_features, num_classes)
        else:
            raise AttributeError("The model does not have a recognized classification head.")

        self.loss_function = loss_function
        self.learning_rate = learning_rate

        self.avg_accuracy = torchmetrics.Accuracy(num_classes=num_classes, task="multiclass")
        self.avg_f1 = torchmetrics.F1Score(num_classes=num_classes, task="multiclass")
        self.avg_precision = torchmetrics.Precision(num_classes=num_classes, task="multiclass")
        self.avg_recall = torchmetrics.Recall(num_classes=num_classes, task="multiclass")

        self.confusion_matrix = ConfusionMatrix(num_classes=num_classes, task="multiclass")

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pylint: disable=arguments-differ
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

    def calculate_per_class_metrics(self, predictions: torch.Tensor, labels: torch.Tensor) -> dict:
        r"""
        Calculate TP, FP, TN, FN, precision, recall, and F1-score for each class.

        Args:
            predictions: The output predictions of the model.
            labels: The ground truth labels of the input.

        Returns:
            A dictionary containing per-class metrics.
        """

        # Compute confusion matrix
        confusion_matrix = self.confusion_matrix(predictions, labels)

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
            "tn": tn,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
        }

    def _common_steps(  # pylint: disable=too-many-locals
        self,
        batch: torch.Tensor,
        batch_idx: int, # pylint: disable=unused-argument
        stage: str,  # pylint: disable=unused-argument
    ) -> torch.Tensor:
        r"""
        The function describing the common steps of the training step,
        validations step and test step of the classification model.

        Args:
            batch: The batch of data to be used for training.
            batch_idx: The index of the batch.
            stage: The stage of the model (train, val, test).

        Returns:
            The loss of the model during the step.

        Shape:
            - :code:`batch`: :math:`(I_k, L_k)`
            - :code:`batch_idx`: b
            - :code:`stage`: str
            - Output: idk

            | where
            |
            | :math:`I_k = \text{ k-th input image of the batch encoded as tensor}`
            | :math:`L_k = \text{ k-th class index of the k-th input index}`
        """

        inputs, labels = batch
        predictions = self.forward(inputs)

        loss = self.loss_function(predictions, labels)

        self.log_dict(
            {
                f"{stage}_loss": loss,
                f"{stage}_accuracy": self.avg_accuracy(predictions, labels),
                f"{stage}_f1": self.avg_f1(predictions, labels),
                f"{stage}_precision": self.avg_precision(predictions, labels),
                f"{stage}_recall": self.avg_recall(predictions, labels),
            },
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )

        per_class_metrics = self.calculate_per_class_metrics(predictions, labels)

        for i, (precision, recall, f1, tp, fp, tn, fn) in enumerate(
            zip(
                per_class_metrics["f1_score"],
                per_class_metrics["precision"],
                per_class_metrics["recall"],
                per_class_metrics["tp"],
                per_class_metrics["fp"],
                per_class_metrics["tn"],
                per_class_metrics["fn"],
            )
        ):
            self.log_dict(
                {
                    f"test_precision_class_{i}": precision.float(),
                    f"test_recall_class_{i}": recall.float(),
                    f"test_f1_score_class_{i}": f1.float(),
                    f"test_tp_class_{i}": tp.float(),
                    f"test_fp_class_{i}": fp.float(),
                    f"test_tn_class_{i}": tn.float(),
                    f"test_fn_class_{i}": fn.float(),
                }
            )

        return loss

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:  # pylint: disable=arguments-differ
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

        return self._common_steps(batch, batch_idx, "train")

    def validation_step(self, batch: torch.Tensor, batch_idx: int):  # pylint: disable=arguments-differ
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

        self._common_steps(batch, batch_idx, "val")

    def test_step(self, batch: torch.Tensor, batch_idx: int):  # pylint: disable=arguments-differ
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

        self._common_steps(batch, batch_idx, "test")

    def predict_step(  # pylint: disable=arguments-differ
        self, batch: torch.Tensor, batch_idx: int  # pylint: disable=unused-argument
    ) -> tuple:  # pylint: disable=arguments-differ
        r"""
        The predict step of the classification model.

        Args:
            batch: The batch of data to be used for training.
            batch_idx: The index of the batch.

        Returns:
            The class id and the score of the prediction.

        Shape:
            - :code:`batch`: :math:`(I_k, L_k)`
            - :code:`batch_idx`: b
            - Output: (class_id, score)

            | where
            |
            | :math:`I_k = \text{ k-th input image of the batch encoded as tensor}`
            | :math:`L_k = \text{ k-th class index of the k-th input index}`
            | :math:`class_id = \text{ id of the predicted class}`
            | :math:`score = \text{ score of the predicted class}`
        """

        predictions = self.forward(batch).squeeze(0).softmax(0)
        class_id = int(predictions.argmax().item())
        score = predictions[class_id].item()

        return class_id, score

    def configure_optimizers(self) -> torch.optim.Optimizer:
        r"""
        The function describing the optimizer of the classification model.

        Returns:
            The optimizer to be used for training.
        """

        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def predict(self, img_path: str) -> dict:
        r"""
        The predict function of the classification model.

        Args:
            img_path: The path to the image to be predicted.

        Returns:
            A dictionary containing the predicted category and confidence score.
        """
        # Decode the image and apply the model's transforms
        picture = decode_image(img_path)
        batch = self.model_weights.transforms()(picture).unsqueeze(0)

        class_id, score = self.predict_step(batch, 0)

        # Return the prediction as a dictionary
        return {"category": class_id, "score": score}
