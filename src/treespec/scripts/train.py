"""Training Script of the Treespec Pipeline"""

import torch
from torch import nn
import pytorch_lightning as L
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from treespec.models.classification_model import ClassificationModel
from treespec.datasets.image_dataset import ImageDataset
from treespec.conf.config import TreespecConfig
from treespec.conf.config_parser import train_config_values as train_config_values

if __name__ == "__main__":
    """Training Script of the Treespec Pipeline"""

    default_transforms = train_config_values("model_weights").transforms()

    dataset = train_config_values("dataset")(
        data_dir=train_config_values("dataset_dir"),
        batch_size=train_config_values("batch_size"),
        num_workers=train_config_values("num_workers"),
    )
    dataset.prepare_data()
    dataset.setup(transform=default_transforms)

    loss_function = train_config_values("loss_function")(label_smoothing=0.1, weight=dataset.loss_weights())

    model = ClassificationModel(
        model=train_config_values("model"),
        model_weights=train_config_values("model_weights"),
        num_classes=train_config_values("num_classes"),
        loss_function=loss_function,
        learning_rate=train_config_values("learning_rate"),
    )

    early_stop_callback = EarlyStopping(
        monitor="val_loss",           # or another metric, e.g. "val_acc"
        patience=5,                   # number of epochs with no improvement after which training will be stopped
        verbose=True,
        mode="min"                    # "min" for loss, "max" for accuracy
    )
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=train_config_values("trained_model_dir"),
        filename=train_config_values("model") + "_best",
        save_top_k=1,
        mode="min"
    )

    trainer = L.Trainer(
        max_epochs=train_config_values("epoch_count"),
        log_every_n_steps=10,
        callbacks=[early_stop_callback, checkpoint_callback],
    )

    trainer.fit(
        model=model,
        train_dataloaders=dataset.train_dataloader(augmentation=train_config_values["train_augmentations"]),
        val_dataloaders=dataset.val_dataloader(),
    )

    # Optionally, test using the best checkpoint
    best_model_path = checkpoint_callback.best_model_path
    if best_model_path:
        model.model.load_state_dict(torch.load(best_model_path))
    trainer.test(model=model, dataloaders=dataset.test_dataloader())

    # Save the best model weights
    if best_model_path:
        torch.save(
            model.model.state_dict(),
            best_model_path,
        )
    else:
        torch.save(
            model.model.state_dict(),
            (train_config_values("trained_model_dir") + train_config_values("model") + "_finetuned" + ".pth"),
        )