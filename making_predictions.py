from keras.models import model_from_json
from pathlib import Path
from keras.preprocessing import image
import numpy as np
from keras.applications import vgg16
import matplotlib.pyplot as plt

# Path to folder with dynacards
dynacards_path = Path("dynacards")

# These are the class labels from the training data (in order from 0 to 11)
class_labels = [
    "Normal operation",
    "Fluid pound",
    "Gas interference",
    "Gas locked pump",
    "Delayed closing of TV",
    "Pump barrel split",
    "Delayed closing of TV + Fluid pound"
]

# Load the json file that contains the model's structure
f = Path("model_structure.json")
model_structure = f.read_text()

# Recreate the Keras model object from the json data
model = model_from_json(model_structure)

# Re-load the model's trained weights
model.load_weights("model_weights.h5")

for img in dynacards_path.glob("*.png"):
    # Load each image to test, resizing it to 224x244 pixels (as required by this model)
    img = image.load_img(img, target_size=(224, 224))

    # Convert the image to a numpy array
    image_array = image.img_to_array(img)

    # Add a forth dimension to the image (since Keras expects a bunch of images, not a single image)
    images = np.expand_dims(image_array, axis=0)

    # Normalize the data
    images = vgg16.preprocess_input(images)

    # Use the pre-trained neural network to extract features from our test image (the same way we did to train the model)
    feature_extraction_model = vgg16.VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    features = feature_extraction_model.predict(images)

    # Given the extracted features, make a final prediction using our own model
    results = model.predict(features)

    # Since we are only testing one image with possible class, we only need to check the first result's first element
    single_result = results[0]

    # We will get a likelihood score for all 12 possible classes. Find out which class had the highest score.
    most_likely_class_index = int(np.argmax(single_result))
    class_likelihood = single_result[most_likely_class_index]

    # Get the name of the most likely class
    class_label = class_labels[most_likely_class_index]

    # Print the result
    print("Problem detected: {}".format(class_label))
    print("Likelihood: {:2f}".format(class_likelihood))

    # Draw the image as a plot
    plt.imshow(img)
    # Label the image
    plt.title(class_label)
    # Show the plot on the screen
    plt.show()