import cv2
import tensorflow as tf
from tensorflow import keras
from keras import layers
from IPython import display
import matplotlib.pyplot as plt
import numpy as np
import tensorflow_datasets as tfds
from pandas import *
import time
from matplotlib import gridspec

latent_dim = 100

con_label = layers.Input(shape=(1,))

# latent vector input
latent_vector = layers.Input(shape=(latent_dim,))

def label_conditioned_generator(n_classes=2, embedding_dim=100):
    # embedding for categorical input
    label_embedding = layers.Embedding(n_classes, embedding_dim)(con_label)
    # linear multiplication
    nodes = 4 * 4 
    label_dense = layers.Dense(nodes)(label_embedding)
    # reshape to additional channel
    label_reshape_layer = layers.Reshape((4, 4, 1))(label_dense)
    return label_reshape_layer

def latent_input(latent_dim=100):
    # image generator input
    nodes = 512 * 4 * 4
    latent_dense = layers.Dense(nodes)(latent_vector)
    latent_dense = layers.ReLU()(latent_dense)
    latent_reshape = layers.Reshape((4, 4, 512))(latent_dense)
    return latent_reshape

def define_generator():
    label_output = label_conditioned_generator()
    latent_vector_output= latent_input()
    # merge label_conditioned_generator and latent_input output
    merge = layers.Concatenate()([latent_vector_output, label_output])
    
    x = layers.Conv2DTranspose(64 * 8, kernel_size=4, strides= 2, padding='same', kernel_initializer=tf.keras.initializers.RandomNormal(
    mean=0.0, stddev=0.02), use_bias=False, name='conv_transpose_1')(merge)
    x = layers.BatchNormalization(momentum=0.1,  epsilon=0.8, center=1.0, scale=0.02, name='bn_1')(x)
    x = layers.ReLU(name='relu_1')(x)
    
    x = layers.Conv2DTranspose(64 * 4, kernel_size=4, strides= 2, padding='same', kernel_initializer=tf.keras.initializers.RandomNormal(
    mean=0.0, stddev=0.02), use_bias=False, name='conv_transpose_2')(x)
    x = layers.BatchNormalization(momentum=0.1,  epsilon=0.8, center=1.0, scale=0.02, name='bn_2')(x)
    x = layers.ReLU(name='relu_2')(x)
    
    x = layers.Conv2DTranspose(64 * 2, 4, 2, padding='same', kernel_initializer=tf.keras.initializers.RandomNormal(
    mean=0.0, stddev=0.02), use_bias=False, name='conv_transpose_3')(x)
    x = layers.BatchNormalization(momentum=0.1,  epsilon=0.8,  center=1.0, scale=0.02, name='bn_3')(x)
    x = layers.ReLU(name='relu_3')(x)
  

    x = layers.Conv2DTranspose(64 * 1, 4, 2, padding='same', kernel_initializer=tf.keras.initializers.RandomNormal(
    mean=0.0, stddev=0.02), use_bias=False, name='conv_transpose_4')(x)
    x = layers.BatchNormalization(momentum=0.1,  epsilon=0.8,  center=1.0, scale=0.02, name='bn_4')(x)
    x = layers.ReLU(name='relu_4')(x) 
    
    
    out_layer = layers.Conv2DTranspose(3, 4, 2,padding='same', kernel_initializer=tf.keras.initializers.RandomNormal(
    mean=0.0, stddev=0.02), use_bias=False, activation='tanh', name='conv_transpose_6')(x)
    
    
   # define model
    model = tf.keras.Model([latent_vector,  con_label], out_layer)
    return model

conditional_gen = define_generator()

conditional_gen.load_weights('genned_skins/gen_99.h5')

def generate_images(model, test_input, run):
  # Notice `training` is set to False.
  # This is so all layers run in inference mode (batchnorm).
    output = None
    for label in range(2):
        labels = tf.ones(10) * label
#         predictions = model([labels, test_input], training=False)
        predictions = model([test_input, labels], training=False)
        if output is None:
            output = predictions
        else:
            output = np.concatenate((output,predictions))
     
    nrow = 2
    ncol = 10
    fig = plt.figure(figsize=(25,25))
    gs = gridspec.GridSpec(nrow, ncol) 

    #output = output.reshape(-1, 128, 128, 2)
    #print("Generated Images are Conditioned on Label:", label_dict[np.array(labels)[0]])
    k = 0
    for i in range(nrow):
        for j in range(ncol):
            pred = (output[k, :, :, :] + 1 ) * 127.5
            pred = np.array(pred)  
            ax= plt.subplot(gs[i,j])
            ax.imshow(pred.astype(np.uint8))
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.axis('off')
            k += 1   

    plt.savefig('created_skins/result_grid_{:d}.png'.format(run),  dpi=300)
    # plt.show()


num_examples_to_generate = 10
latent_dim = 100
for i in range(50):
    noise = tf.random.normal([num_examples_to_generate, latent_dim])

    generate_images(conditional_gen, noise, i)