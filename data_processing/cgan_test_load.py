import cv2
import tensorflow as tf
from tensorflow import keras
from keras import layers
from IPython import display
import matplotlib.pyplot as plt
import numpy as np
import tensorflow_datasets as tfds
from pandas import *
 
ds = tf.data.Dataset.load("saved_dataset")
print(ds)
for element in ds:
    print(element)
images, labels = tuple(zip(*ds))