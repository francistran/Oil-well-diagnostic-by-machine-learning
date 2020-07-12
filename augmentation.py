import numpy as np
from numpy import expand_dims
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import array_to_img
from matplotlib import pyplot
from keras.applications.inception_v3 import preprocess_input

train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=1,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=False
)

test_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=1,
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = False
)

train_generator = train_datagen.flow_from_directory('augmentation/sample-train/',target_size=(224,224), save_to_dir='augmentation/sample-confirm/training/')

i=0
for batch in train_datagen.flow_from_directory('augmentation/sample-train/', target_size=(224,224), save_to_dir='augmentation/sample-confirm/training/'):
    i+=1
    if (i > 40):
        break

train_generator = test_datagen.flow_from_directory('augmentation/sample-train/',target_size=(224,224), save_to_dir='augmentation/sample-confirm/test/')

j=0
for batch in test_datagen.flow_from_directory('augmentation/sample-train/', target_size=(224,224), save_to_dir='augmentation/sample-confirm/test/'):
    j+=1
    if (j > 25):
        break


