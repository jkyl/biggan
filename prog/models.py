import tensorflow as tf
import numpy as np
import glob
import tqdm
import time
import os

LAYER_DEPTHS = [512, 512, 512, 256, 128, 64, 32, 16] # TODO: dynamic final size
INITIALIZER = tf.keras.initializers.he_normal()#RandomNormal(mean=0, stddev=1, seed=None)

def InputFunc(z):
    ''''''
    x = tf.keras.layers.Dense(4*4*512, kernel_initializer=INITIALIZER)(z)
    x = tf.keras.layers.LeakyReLU(0.2)(x)
    x = tf.keras.layers.Reshape((4, 4, 512))(x)
    x = tf.keras.layers.Conv2D(512, 3, padding='same', kernel_initializer=INITIALIZER)(x)
    x = tf.keras.layers.LeakyReLU(0.2)(x)
    x_rgb = tf.keras.layers.Conv2D(3, 1, padding='same', kernel_initializer=INITIALIZER)(x)
    return x, x_rgb

def UpConvFunc(x, dim, n=2):
    ''''''
    x = tf.keras.layers.UpSampling2D((2, 2))(x)
    for i in range(n):
        x = tf.keras.layers.Conv2D(dim, 3, padding='same', kernel_initializer=INITIALIZER)(x)
        x = tf.keras.layers.LeakyReLU(0.2)(x)
    x_rgb = tf.keras.layers.Conv2D(3, 1, padding='same', kernel_initializer=INITIALIZER)(x)
    return x, x_rgb
    
def NVIDIA_generator():
    ''''''
    n_blocks = int(np.log2(1024//4))
    z = tf.keras.layers.Input([512])
    x, x_rgb = InputFunc(z)
    outputs = [x_rgb]
    for i in range(n_blocks):
        dim = LAYER_DEPTHS[i]
        x, x_rgb = UpConvFunc(x, dim, n=2)
        outputs.append(x_rgb)
    return tf.keras.models.Model(z, outputs)

def DownConvLayers(dim, n=2):
    '''
    This function returns a list of keras layers, not yet called on any input.
    This is because we want to reuse the low-res layers on feature maps extracted
    by the hi-res layers. 
    '''
    layers = []
    for i in range(n):
        layers.append(tf.keras.layers.Conv2D(
            min(512, dim*(2 if i==(n-1) else 1)), 
            3, padding='same', strides=(2 if i==(n-1) else 1), 
            kernel_initializer=INITIALIZER))
        layers.append(tf.keras.layers.LeakyReLU(0.2))
    return layers

def OutputLayers():
    return [
        tf.keras.layers.Conv2D(512, 3, padding='same', kernel_initializer=INITIALIZER),
        tf.keras.layers.LeakyReLU(0.2),
        tf.keras.layers.Conv2D(512, 4, padding='valid', kernel_initializer=INITIALIZER),
        tf.keras.layers.LeakyReLU(0.2),
        tf.keras.layers.Dense(1, kernel_initializer=INITIALIZER),
        tf.keras.layers.Flatten()]

def NVIDIA_discriminator():
    ''''''
    n_blocks = int(np.log2(1024//4))
    sizes = [1024*2**-i for i in range(n_blocks+1)]
    
    # construct input placeholders and project to hidden dim
    inputs = [tf.keras.layers.Input(
        (sizes[i], sizes[i], 3)) for i in range(n_blocks+1)]
    in_rgb = [tf.keras.layers.Conv2D(
        ([512]+LAYER_DEPTHS)[-(i+1)], 1, padding='same', 
        kernel_initializer=INITIALIZER)(
        inputs[i]) for i in range(n_blocks+1)]
        
    # build model blocks, params shared between all paths
    blocks = [DownConvLayers(
        LAYER_DEPTHS[-(i+1)], n=2) for i in range(n_blocks)] + [
    OutputLayers()]
        
    # define all paths to the output
    outputs = []
    for i in range(n_blocks+1):
        x = in_rgb[i]
        for block in blocks[i:]:
            for layer in block:
                x = layer(x)
        outputs = [x] + outputs
    output = tf.keras.layers.concatenate(outputs, axis=-1)
    return tf.keras.models.Model(inputs[::-1], output)
