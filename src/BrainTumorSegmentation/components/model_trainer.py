import os
import numpy as np
import tensorflow as tf
import segmentation_models_3D as sm
from BrainTumorSegmentation.utils.common import load_img, imageLoader
from BrainTumorSegmentation.models.unet_3D_model import unet_model
from BrainTumorSegmentation.entity.config_entity import TrainingConfig


class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config

    def train(self):
        train_img_list = os.listdir(self.config.train_img_dir)
        train_mask_list = os.listdir(self.config.train_mask_dir)
        val_img_list = os.listdir(self.config.val_img_dir)
        val_mask_list = os.listdir(self.config.val_mask_dir)

        train_gen = imageLoader(
            self.config.train_img_dir,
            train_img_list,
            self.config.train_mask_dir,
            train_mask_list,
            self.config.batch_size,
        )
        val_gen = imageLoader(
            self.config.val_img_dir,
            val_img_list,
            self.config.val_mask_dir,
            val_mask_list,
            self.config.batch_size,
        )

        wt0, wt1, wt2, wt3 = 0.25, 0.25, 0.25, 0.25

        dice_loss = sm.losses.DiceLoss(class_weights=np.array([wt0, wt1, wt2, wt3]))
        focal_loss = sm.losses.CategoricalFocalLoss()
        total_loss = dice_loss + (1 * focal_loss)

        metrics = ["accuracy", sm.metrics.IOUScore(threshold=0.5)]

        LR = self.config.lr
        optim = tf.keras.optimizers.Adam(LR)
        #######################################################################
        # Fit the model

        steps_per_epoch = len(train_img_list) // self.config.batch_size
        val_steps_per_epoch = len(val_img_list) // self.config.batch_size

        model = unet_model(
            IMG_HEIGHT=self.config.img_size,
            IMG_WIDTH=self.config.img_size,
            IMG_DEPTH=self.config.img_size,
            IMG_CHANNELS=self.config.channels,
            num_classes=self.config.num_classes,
        )

        model.compile(optimizer=optim, loss=total_loss, metrics=metrics)
        history = model.fit(
            train_gen,
            steps_per_epoch=steps_per_epoch,
            epochs=self.config.epochs,
            verbose=1,
            validation_data=val_gen,
            validation_steps=val_steps_per_epoch,
        )

        model.save(self.config.model_path)
