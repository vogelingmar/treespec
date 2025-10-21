"""Classification Model to classify tree images."""  # pylint: disable=duplicate-code

from typing import Callable, Optional
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

import torch
from pathlib import Path
from torch import nn
from torch.nn.modules.loss import _Loss
from torchvision.io import read_image, decode_image  # instead of decode_image
from torchvision.models._api import WeightsEnum  # type: ignore
import torchmetrics
from torchmetrics import ConfusionMatrix

import pytorch_lightning as L


class ClassificationModel(L.LightningModule):  # pylint: disable=too-many-instance-attributes
    r"""
    The tree species classification model of the treespec pipeline.

    Args:
        model: The model to be used for classification.
        model_weights: The weights to be used to initialize the model.
        num_classes: The number of classes to be differentiated by the model.
        loss_function: The loss function to be used for training.
        learning_rate: The learning rate to be used for training.
    """

    def __init__(  # pylint: disable=too-many-arguments, too-many-positional-arguments
        self,
        model: Callable,
        model_weights: WeightsEnum,
        num_classes: int,
        loss_function: _Loss,
        learning_rate: float,
        class_labels: Optional[list[str]] = None,
    ) -> None:
        super().__init__()
        self.model_weights = model_weights
        self.model = model(weights=self.model_weights)

        if hasattr(self.model, "fc"):
            self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        elif hasattr(self.model, "head"):
            self.model.head = nn.Linear(self.model.head.in_features, num_classes)
        elif hasattr(self.model, "classifier"):
            if isinstance(self.model.classifier, nn.Sequential):
                self.model.classifier[-1] = nn.Linear(self.model.classifier[-1].in_features, num_classes)
            else:
                self.model.classifier = nn.Linear(self.model.classifier.in_features, num_classes)
        else:
            raise AttributeError("The model does not have a recognized classification head.")

        self.loss_function = loss_function
        self.learning_rate = learning_rate

        self.avg_accuracy = torchmetrics.Accuracy(num_classes=num_classes, task="multiclass")
        self.avg_f1 = torchmetrics.F1Score(num_classes=num_classes, task="multiclass")
        self.avg_precision = torchmetrics.Precision(num_classes=num_classes, task="multiclass")
        self.avg_recall = torchmetrics.Recall(num_classes=num_classes, task="multiclass")

        self.confusion_matrix = ConfusionMatrix(num_classes=num_classes, task="multiclass")

        self.test_confmat = ConfusionMatrix(task="multiclass", num_classes=num_classes)
        self.class_labels = class_labels

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pylint: disable=arguments-differ
        r"""
        The forward method of the classification model.

        Args:
            x: Input tensor

        Returns:
            Output tensor

        Shape:
            - :code:`x`: :math:`(B, C, H, W)`
            - Output: :math:`(B, N)`

            | where
            |
            | :math:`B = \text{ batch size}`
            | :math:`C = \text{ number of channels}`
            | :math:`H = \text{ height of the input image}`
            | :math:`W = \text{ width of the input image}`
            | :math:`N = \text{ number of classes to be differentiated by the model}`
        """
        return self.model(x)

    def calculate_per_class_metrics(self, predictions: torch.Tensor, labels: torch.Tensor) -> dict:
        r"""
        Calculate true positives, false positives, true negatives, false negatives, precision, recall, and F1-score for each class.

        Args:
            predictions: The output predictions of the model.
            labels: The ground truth labels of the input.

        Returns:
            A dictionary containing per-class metrics.
        """

        confusion_matrix = self.confusion_matrix(predictions, labels)

        true_positive = torch.diag(confusion_matrix)
        false_positive = confusion_matrix.sum(dim=0) - true_positive
        false_negative = confusion_matrix.sum(dim=1) - true_positive
        true_negative = confusion_matrix.sum() - (true_positive + false_positive + false_negative)

        precision = true_positive / (true_positive + false_positive + 1e-8)
        recall = true_positive / (true_positive + false_negative + 1e-8)
        f1_score = 2 * (precision * recall) / (precision + recall + 1e-8)

        return {
            "true_positive": true_positive,
            "false_positive": false_positive,
            "true_negative": true_negative,
            "false_negative": false_negative,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
        }

    def _common_steps(  # pylint: disable=too-many-locals
        self,
        batch: torch.Tensor,
        batch_idx: int,  # pylint: disable=unused-argument
        stage: str,  # pylint: disable=unused-argument
        logging: bool,
    ) -> torch.Tensor:
        r"""
        The function describing the common steps of the training step,
        validations step and test step of the classification model.

        Args:
            batch: The batch of data to be used for training.
            batch_idx: The index of the batch.
            stage: The stage of the model (train, val, test).
            logging: Whether to log the metrics or not.

        Returns:
            The loss of the model during the step.

        Shape:
            - :code:`batch`: :math:`(I_k, L_k)`

            | where
            |
            | :math:`I_k = \text{ k-th input image of the batch encoded as tensor}`
            | :math:`L_k = \text{ k-th class index of the k-th input index}`
        """

        inputs, labels = batch
        predictions = self.forward(inputs)

        loss = self.loss_function(predictions, labels)

        if logging:
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

            for i, (precision, recall, f1, true_positive, false_positive, true_negative, false_negative) in enumerate(
                zip(
                    per_class_metrics["f1_score"],
                    per_class_metrics["precision"],
                    per_class_metrics["recall"],
                    per_class_metrics["true_positive"],
                    per_class_metrics["false_positive"],
                    per_class_metrics["true_negative"],
                    per_class_metrics["false_negative"],
                )
            ):
                self.log_dict(
                    {
                        f"{stage}_precision_class_{i}": precision.float(),
                        f"{stage}_recall_class_{i}": recall.float(),
                        f"{stage}_f1_score_class_{i}": f1.float(),
                        f"{stage}_true_positives_class_{i}": true_positive.float(),
                        f"{stage}_false_positives_class_{i}": false_positive.float(),
                        f"{stage}_true_negatives_class_{i}": true_negative.float(),
                        f"{stage}_false_negatives_class_{i}": false_negative.float(),
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

            | where
            |
            | :math:`I_k = \text{ k-th input image of the batch encoded as tensor}`
            | :math:`L_k = \text{ k-th class index of the k-th input index}`
        """

        return self._common_steps(batch, batch_idx, "train", True)

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> None:  # pylint: disable=arguments-differ
        r"""
        The function describing the validation step of the classification model.

        Args:
            batch: The batch of data to be used for training.
            batch_idx: The index of the batch.

        Shape:
            - :code:`batch`: :math:`(I_k, L_k)`

            | where
            |
            | :math:`I_k = \text{ k-th input image of the batch encoded as tensor}`
            | :math:`L_k = \text{ k-th class index of the k-th input index}`
        """

        self._common_steps(batch, batch_idx, "val", True)

    def test_step(self, batch: torch.Tensor, batch_idx: int) -> None:  # pylint: disable=arguments-differ
        r"""
        The function describing the validation step of the classification model.

        Args:
            batch: The batch of data to be used for training.
            batch_idx: The index of the batch.

        Shape:
            - :code:`batch`: :math:`(I_k, L_k)`

            | where
            |
            | :math:`I_k = \text{ k-th input image of the batch encoded as tensor}`
            | :math:`L_k = \text{ k-th class index of the k-th input index}`
        """

        inputs, labels = batch
        outputs = self.forward(inputs)
        preds = torch.argmax(outputs, dim=1)

        self.test_confmat.update(preds, labels)

        self._common_steps(batch, batch_idx, "test", True)

    def predict_step(  # pylint: disable=arguments-differ
        self, batch: torch.Tensor, batch_idx: int  # pylint: disable=unused-argument
    ) -> tuple[int, int]:  # pylint: disable=arguments-differ
        r"""
        The predict step of the classification model.

        Args:
            batch: The batch of data to be used for training.
            batch_idx: The index of the batch.

        Returns:
            The class id and the score of the prediction.

        Shape:
            - :code:`batch`: :math:`(I_k, L_k)`

            | where
            |
            | :math:`I_k = \text{ k-th input image of the batch encoded as tensor}`
            | :math:`L_k = \text{ k-th class index of the k-th input index}`
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

    def predict(self, img_path: Path) -> dict:
        #TODO: remove this from the classification model
        r"""
        The predict function of the classification model.

        Args:
            img_path: The path to the image to be predicted.

        Returns:
            A dictionary containing the predicted category and confidence score.
        """
        try:
            picture = decode_image(img_path)
        except Exception as e:
            raise ValueError(f"Could not read image: {e}")
        batch = self.model_weights.transforms()(picture).unsqueeze(0)
        device = next(self.model.parameters()).device
        batch = batch.to(device)
        class_id, score = self.predict_step(batch, 0)
        return {"category": class_id, "score": score}

    def on_test_epoch_end(self):
        """
        Called at the end of the test epoch to compute and visualize the confusion matrix.
        """
        confmat = self.test_confmat.compute().cpu().numpy()

        print("✅ Confusion Matrix:")
        print(confmat)

        # ✅ Optional: add class labels if available
        # You can define them in your LightningModule init, or pass from outside
        # Example:
        class_labels = getattr(self, "class_labels", None)

        disp = ConfusionMatrixDisplay(confusion_matrix=confmat, display_labels=class_labels)
        disp.plot(cmap="Blues", xticks_rotation=45)
        plt.title("Test Confusion Matrix")
        plt.show()

        # Reset confusion matrix for safety (especially if testing multiple times)
        self.test_confmat.reset()