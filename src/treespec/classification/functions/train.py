"""Training function for tree species classification"""

from torch import nn
import torch
import pytorch_lightning as L
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pathlib import Path
import os
from torchvision.models._api import WeightsEnum  # type: ignore
from torch.nn.modules.loss import _Loss
from typing import Callable, Optional
from torchvision.transforms.v2 import Transform  # type: ignore

from treespec.classification.classification_model import ClassificationModel


def train(
    model: Callable,
    model_weights: WeightsEnum,
    input_dataset: L.LightningDataModule,
    dataset_dir_path: Path,
    num_classes: int,
    use_ids: bool,
    epoch_count: int,
    batch_size: int,
    num_workers: int,
    learning_rate: float,
    input_loss_function: _Loss,
    trained_model_dir: Path,
    trained_model_path: Optional[Path] = None,
    train_augmentations: Optional[Transform] = None,
) -> None:
    """
    Trains a tree species classification model using the Treespec pipeline.

    This function sets up the dataset, initializes the model, and trains it using PyTorch Lightning.
    It includes early stopping and model checkpointing to save the best-performing model.

    Args:
        model: The model architecture to be used for training.
        model_weights: Pre-trained weights for the model.
        image_dataset: Dataset class to be used for loading training, validation, and test data.
        dataset_dir_path: Directory containing the dataset.
        num_classes: Number of classes to destinguish.
        use_ids: Whether to use IDs for dataset loading to prevent dataset leakage.
        epoch_count: Maximum number of epochs to train the model.
        batch_size: Batch size for data loading.
        num_workers: Number of workers for data loading.
        learning_rate: Learning rate for the model.
        input_loss_function: Loss function to be used during training.
        trained_model_dir: Directory to save the trained model checkpoints. If pre_trained is True, this is the path to the pre-trained model.
        train_augmentations: Optional augmentations to apply to the training data.
        pre_trained: Whether to use pre-trained weights from the trained model directory for training.
    """
    default_transforms = model_weights.transforms()

    dataset = input_dataset(
        dataset_dir_path=dataset_dir_path,
        batch_size=batch_size,
        num_workers=num_workers,
        use_ids=use_ids,
    )
    dataset.prepare_data()
    dataset.setup(transform=default_transforms)

    loss_function = input_loss_function(label_smoothing=0.1, weight=dataset.loss_weights())

    classification_model = ClassificationModel(
            model=model,
            model_weights=model_weights,
            num_classes=num_classes,
            loss_function=loss_function,
            learning_rate=learning_rate,
        )
    
    if trained_model_path is not None:

        checkpoint = torch.load(trained_model_path, map_location="cpu")

        for key in [
            "model.classifier.6.weight",
            "model.classifier.6.bias",
            "loss_function.weight",
        ]:
            if key in checkpoint["state_dict"]:
                del checkpoint["state_dict"][key]

        classification_model.model.load_state_dict(checkpoint["state_dict"], strict=False)

    early_stop_callback = EarlyStopping(
        monitor="train_loss",  # exchange for any metric (adjust mode accordingly)
        patience=10,
        verbose=True,
        mode="min",
    )

    filename = f"{type(classification_model.model).__name__}_{Path(dataset_dir_path).stem}_{num_classes}_checkpoint"
    final_name = f"{type(classification_model.model).__name__}_{Path(dataset_dir_path).stem}_{num_classes}_finetuned"

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=trained_model_dir,
        filename=filename,
        save_top_k=1,
        mode="min",
    )

    trainer = L.Trainer(
        max_epochs=epoch_count,
        log_every_n_steps=10,
        callbacks=[early_stop_callback, checkpoint_callback],
    )

    trainer.fit(
        model=classification_model,
        train_dataloaders=dataset.train_dataloader(augmentation=train_augmentations),
        val_dataloaders=dataset.val_dataloader(),
    )

    best_model_path = checkpoint_callback.best_model_path
    if best_model_path:
        classification_model = ClassificationModel.load_from_checkpoint(
            best_model_path,
            model=model,
            model_weights=model_weights,
            num_classes=num_classes,
            loss_function=loss_function,
            learning_rate=learning_rate,
        )
    trainer.test(model=classification_model, dataloaders=dataset.test_dataloader())

    final_model_path = os.path.join(trained_model_dir, final_name)
    #torch.save(
    #    classification_model.model.state_dict(),
    #    final_model_path,
    #)
    trainer.save_checkpoint(final_model_path)
