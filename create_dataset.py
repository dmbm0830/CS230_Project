import cv2
import tensorflow as tf
from tensorflow import keras
from keras import layers
from IPython import display
import matplotlib.pyplot as plt
import numpy as np
import tensorflow_datasets as tfds
import pandas as pd
from datasets import Dataset

data = pd.read_csv("skin_csv_info.csv")
filenames = data["Filename"].tolist()
labels = np.asarray(data["Label (0 is Close and 1 is Full)"].tolist()).astype(np.float32)
image_list = []
for name in filenames:
    img = cv2.imread(name)
    r_img = cv2.resize(img, (512, 512))
    image_list.append(r_img)
# image_data = np.asarray(image_list, dtype=object)
df = pd.DataFrame(data={'images': image_list, 'labels': labels})
# print(image_data.shape)
# print(image_data[0].shape)
# print(image_data[0])
print("Created image data list")
# data = {"inputs": image_data, "labels": labels}
# ds = Dataset.from_dict(data)
# print("Created pre-tf Dataset")
# tf_ds = ds.to_tf_dataset(
#             columns=["inputs"],
#             label_cols=["labels"],
#             batch_size=128,
#             shuffle=True
#             )
# labels_p = tf.convert_to_tensor(labels, dtype=tf.float32)
# images_p = tf.convert_to_tensor(image_data, dtype=tf.float32)
# print("placeholders made")
tf_ds = tf.data.Dataset.from_tensor_slices((list(df['images'].values), df['labels'].values))
tf_ds.save("saved_dataset")
print("Dataset saved")