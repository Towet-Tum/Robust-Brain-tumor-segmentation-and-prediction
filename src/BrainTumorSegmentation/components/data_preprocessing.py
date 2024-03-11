import glob
import numpy as np
import splitfolders
import nibabel as nib
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.utils import to_categorical
from BrainTumorSegmentation.entity.config_entity import DataPreprocessConfig


class DataPreprocess:
    def __init__(self, config: DataPreprocessConfig):
        self.config = config

    def resize_and_normalize(self):
        t2_list = sorted(glob.glob(f"{self.config.dataset}/*/*t2.nii"))
        t1ce_list = sorted(glob.glob(f"{self.config.dataset}/*/*t1ce.nii"))
        flair_list = sorted(glob.glob(f"{self.config.dataset}/*/*flair.nii"))
        mask_list = sorted(glob.glob(f"{self.config.dataset}/*/*seg.nii"))

        scaler = MinMaxScaler()
        for img in range(len(t2_list)):  # Using t1_list as all lists are of same size
            print("Now preparing image and masks number: ", img)

            temp_image_t2 = nib.load(t2_list[img]).get_fdata()
            temp_image_t2 = scaler.fit_transform(
                temp_image_t2.reshape(-1, temp_image_t2.shape[-1])
            ).reshape(temp_image_t2.shape)

            temp_image_t1ce = nib.load(t1ce_list[img]).get_fdata()
            temp_image_t1ce = scaler.fit_transform(
                temp_image_t1ce.reshape(-1, temp_image_t1ce.shape[-1])
            ).reshape(temp_image_t1ce.shape)

            temp_image_flair = nib.load(flair_list[img]).get_fdata()
            temp_image_flair = scaler.fit_transform(
                temp_image_flair.reshape(-1, temp_image_flair.shape[-1])
            ).reshape(temp_image_flair.shape)

            temp_mask = nib.load(mask_list[img]).get_fdata()
            temp_mask = temp_mask.astype(np.uint8)
            temp_mask[temp_mask == 4] = 3  # Reassign mask values 4 to 3
            # print(np.unique(temp_mask))

            temp_combined_images = np.stack(
                [temp_image_flair, temp_image_t1ce, temp_image_t2], axis=3
            )

            # Crop to a size to be divisible by 64 so we can later extract 64x64x64 patches.
            # cropping x, y, and z
            temp_combined_images = temp_combined_images[56:184, 56:184, 13:141]
            temp_mask = temp_mask[56:184, 56:184, 13:141]

            val, counts = np.unique(temp_mask, return_counts=True)

            if (
                1 - (counts[0] / counts.sum())
            ) > 0.01:  # At least 1% useful volume with labels that are not 0
                print("Save Me")
                temp_mask = to_categorical(temp_mask, num_classes=4)
                np.save(
                    f"{self.config.images}/image_" + str(img) + ".npy",
                    temp_combined_images,
                )
                np.save(f"{self.config.masks}/mask_" + str(img) + ".npy", temp_mask)

    def train_val_split(self):
        splitfolders.ratio(
            self.config.dataset_path,
            output=self.config.splitted_dataset,
            seed=42,
            ratio=(0.75, 0.25),
            group_prefix=None,
        )
