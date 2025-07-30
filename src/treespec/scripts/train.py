"""Training Script of the Treespec Pipeline"""

import torch
import pytorch_lightning as L
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import pathlib
import hydra
from hydra.core.config_store import ConfigStore

from treespec.models.classification_model import ClassificationModel
from treespec.conf.config import TreespecConfig
from treespec.conf.config_parser import train_config_values

cs = ConfigStore.instance()
cs.store(name="treespec_config", node=TreespecConfig)


@hydra.main(config_path="../conf", config_name="config")
def main(cfg: TreespecConfig):
    """Training Script of the Treespec Pipeline"""

    default_transforms = train_config_values("model_weights", cfg).transforms()

    dataset = train_config_values("dataset", cfg)(
        data_dir=train_config_values("dataset_dir", cfg),
        batch_size=train_config_values("batch_size", cfg),
        num_workers=train_config_values("num_workers", cfg),
        use_ids=train_config_values("use_ids", cfg),
    )
    dataset.prepare_data()
    dataset.setup(transform=default_transforms)

    loss_function = train_config_values("loss_function", cfg)(label_smoothing=0.1, weight=dataset.loss_weights())

    model = ClassificationModel(
        model=train_config_values("model", cfg),
        model_weights=train_config_values("model_weights", cfg),
        num_classes=train_config_values("num_classes", cfg),
        loss_function=loss_function,
        learning_rate=train_config_values("learning_rate", cfg),
    )

    early_stop_callback = EarlyStopping(
        monitor="train_loss",  # exchange for any metric (adjust mode accordingly)
        patience=10,
        verbose=True,
        mode="min",
    )

    filename = f"{cfg.train.model}_{pathlib.Path(train_config_values("dataset_dir", cfg)).name}_best"
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=train_config_values("trained_model_dir", cfg),
        filename=filename,
        save_top_k=1,
        mode="min",
    )

    trainer = L.Trainer(
        max_epochs=train_config_values("epoch_count", cfg),
        log_every_n_steps=10,
        callbacks=[early_stop_callback, checkpoint_callback],
    )

    trainer.fit(
        model=model,
        train_dataloaders=dataset.train_dataloader(augmentation=train_config_values("train_augmentations", cfg)),
        val_dataloaders=dataset.val_dataloader(),
    )

    best_model_path = checkpoint_callback.best_model_path
    if best_model_path:
        model = ClassificationModel.load_from_checkpoint(
            best_model_path,
            model=train_config_values("model", cfg),
            model_weights=train_config_values("model_weights", cfg),
            num_classes=train_config_values("num_classes", cfg),
            loss_function=loss_function,
            learning_rate=train_config_values("learning_rate", cfg),
        )
    trainer.test(model=model, dataloaders=dataset.test_dataloader())

    if best_model_path:
        torch.save(
            model.model.state_dict(),
            best_model_path,
        )
    else:
        torch.save(
            model.model.state_dict(),
            (
                train_config_values("trained_model_dir", cfg)
                + cfg.train.model
                + pathlib.Path(train_config_values("dataset_dir", cfg)).name
                + "_finetuned"
                + ".pth"
            ),
        )


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
