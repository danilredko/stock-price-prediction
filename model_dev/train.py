import os

import wandb
from wandb.keras import WandbCallback
import tensorflow as tf

from model_dev.config import Config
from model_dev.model import ModelBuilder
from model_dev.data_loader import DataLoader


class Trainer:
    def __init__(self):
        wandb.init(project="stock-price-prediction")

        self.config = Config().config
        self.data_loader = DataLoader(**self.config["data_loader_args"])

        self.model_builder = ModelBuilder(
            (self.data_loader.x_train.shape[1],), **self.config["model_kwargs"]
        )
        self.model = self.model_builder.model

    def train(self):
        self.model.compile(
            optimizer="adam", loss="binary_crossentropy", metrics=["binary_accuracy"]
        )
        self.model.fit(
            self.data_loader.x_train,
            self.data_loader.y_train,
            epochs=15000,
            batch_size=200,
            validation_data=(self.data_loader.x_val, self.data_loader.y_val),
            callbacks=[
                tf.keras.callbacks.TensorBoard(
                    log_dir="logs",
                    histogram_freq=1,
                    write_graph=True,
                    write_images=False,
                    update_freq="epoch",
                    profile_batch=2,
                    embeddings_freq=0,
                    embeddings_metadata=None,
                ),
                WandbCallback(),
            ],
        )
        self.model.save(os.path.join(wandb.run.dir, "model.h5"))


if __name__ == "__main__":
    trainer = Trainer()
    trainer.train()
