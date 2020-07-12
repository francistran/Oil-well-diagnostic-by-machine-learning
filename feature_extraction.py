from pathlib import Path
import numpy as np
import joblib
from keras.preprocessing import image
from keras.applications import vgg16
import keras

# Path to folders with training data
c01_path = Path("training_data") / "01_normal_operation"
c02_path = Path("training_data") / "02_fluid_pound"
c03_path = Path("training_data") / "03_gas_interference"
c04_path = Path("training_data") / "04_gas_locked_pump"
c05_path = Path("training_data") / "05_delayed_closing_of_TV"
c06_path = Path("training_data") / "06_pump_barrel_split"
c07_path = Path("training_data") / "07_delayed_closing_of_TV+fluid pound"

images = []
labels = []

# Load all the c01 images
for img in c01_path.glob("*.png"):
    # Load the image from disk
    img = image.load_img(img)

    # Convert the image to a numpy array
    image_array = image.img_to_array(img)

    # Add the image to the list of images
    images.append(image_array)

    # For each 'c01' image, the expected value should be 0
    labels.append(0)

# Load all the c02 images
for img in c02_path.glob("*.png"):
    # Load the image from disk
    img = image.load_img(img)

    # Convert the image to a numpy array
    image_array = image.img_to_array(img)

    # Add the image to the list of images
    images.append(image_array)

    # For each 'c02' image, the expected value should be 1
    labels.append(1)

# Load all the c03 images
for img in c03_path.glob("*.png"):
    # Load the image from disk
    img = image.load_img(img)

    # Convert the image to a numpy array
    image_array = image.img_to_array(img)

    # Add the image to the list of images
    images.append(image_array)

    # For each 'c03' image, the expected value should be 2
    labels.append(2)

# Load all the c04 images
for img in c04_path.glob("*.png"):
    # Load the image from disk
    img = image.load_img(img)

    # Convert the image to a numpy array
    image_array = image.img_to_array(img)

    # Add the image to the list of images
    images.append(image_array)

    # For each 'c04' image, the expected value should be 3
    labels.append(3)

# Load all the c05 images
for img in c05_path.glob("*.png"):
    # Load the image from disk
    img = image.load_img(img)

    # Convert the image to a numpy array
    image_array = image.img_to_array(img)

    # Add the image to the list of images
    images.append(image_array)

    # For each 'c05' image, the expected value should be 4
    labels.append(4)

# Load all the c06 images
for img in c06_path.glob("*.png"):
    # Load the image from disk
    img = image.load_img(img)

    # Convert the image to a numpy array
    image_array = image.img_to_array(img)

    # Add the image to the list of images
    images.append(image_array)

    # For each 'c06' image, the expected value should be 5
    labels.append(5)

# Load all the c07 images
for img in c07_path.glob("*.png"):
    # Load the image from disk
    img = image.load_img(img)

    # Convert the image to a numpy array
    image_array = image.img_to_array(img)

    # Add the image to the list of images
    images.append(image_array)

    # For each 'c07' image, the expected value should be 6
    labels.append(6)

# Create a single numpy array with all the images we loaded
x_train = np.array(images)

# Also convert the labels to a numpy array
y_train = np.array(labels)

# Normalize image data to 0-to-1 range
x_train = vgg16.preprocess_input(x_train)

# Load a pre-trained neural network to use as a feature extractor
pretrained_nn = vgg16.VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Extract features for each image (all in one pass)
features_x = pretrained_nn.predict(x_train)

# Convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, 7)

# Save the array of extracted features to a file
joblib.dump(features_x, "x_train.dat")

# Save the matching array of expected values to a file
joblib.dump(y_train, "y_train.dat")
