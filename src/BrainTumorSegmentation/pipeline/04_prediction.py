import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import imageio
import matplotlib.pyplot as plt


class PredictionPipeline:
    def __init__(self, filename, model_path):
        self.filename = filename
        self.model_path = model_path
        self.model = None

    def load_model(self):
        self.model = load_model(self.model_path, compile=False)

    def load_image(self):
        return np.load(self.filename)

    def preprocess_image(self, image):
        # Add any preprocessing steps if needed
        return np.expand_dims(image, axis=0)

    def make_prediction(self, image):
        return self.model.predict(image)

    def get_argmax_prediction(self, prediction):
        return np.argmax(prediction, axis=4)[0, :, :, :]

    def save_as_image(self, array):
        images = []

        for i in range(array.shape[0]):
            # Assign unique colors to each class (adjust as needed)
            colors = {
                0: [0, 0, 0],  # Background
                1: [255, 0, 0],  # Class 1 (Red)
                2: [0, 255, 0],  # Class 2 (Green)
                3: [0, 0, 255],  # Class 3 (Blue)
                # Add more classes if needed
            }

            # Create an RGB image with unique colors for each class
            rgb_image = np.zeros((*array.shape[1:], 3), dtype=np.uint8)
            for class_id, color in colors.items():
                class_mask = (array[i] == class_id).astype(np.uint8)
                rgb_image[class_mask > 0] = color

            # Convert numpy array to PIL image
            image = Image.fromarray(rgb_image, mode="RGB")
            images.append(image)

        return images

    def create_gif(self, images, output_path):
        imageio.mimsave(output_path, images, format="GIF", duration=0.1)

    def make_prediction_pipeline(self):
        self.load_model()
        image = self.load_image()
        preprocessed_image = self.preprocess_image(image)
        prediction = self.make_prediction(preprocessed_image)
        argmax_prediction = self.get_argmax_prediction(prediction)

        return argmax_prediction


if __name__ == "__main__":
    filename = "artifacts/data_preprocess/dataset/images/image_50.npy"
    model_path = "artifacts/training/Brats_3D_2020.h5"
    output_path = "output_image.png"

    pipeline = PredictionPipeline(filename, model_path)
    result = pipeline.make_prediction_pipeline()
    image_list = pipeline.save_as_image(result)

    # Create GIF from the list of images
    output_gif = "output_animation.gif"
    pipeline.create_gif(image_list, output_gif)
