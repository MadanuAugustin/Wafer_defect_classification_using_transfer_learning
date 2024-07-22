import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os



class PredictionPipeline:
    def __init__(self, filename):
        self.filename = filename

    def predict(self):
        # Load the model
        model = load_model(os.path.join('model', 'base_model_updated.keras'))

        # Load and preprocess the image
        imagename = self.filename
        test_image = image.load_img(imagename, target_size=(224, 224))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)

        # Make a prediction
        predictions = model.predict(test_image)
        result = np.argmax(predictions, axis=1)

        # Define class labels (replace these with your actual class names)
        class_labels = ['Center', 'Donut', 'EdgeLoc', 'EdgeRing', 'Loc', 'NearFull', 'none', 'Random', 'Scratch']

        # Get the predicted class label
        predicted_class = class_labels[result[0]]

        return [{"image": predicted_class}]


