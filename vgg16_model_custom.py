from tensorflow.keras import Sequential, Input, layers, optimizers, callbacks, regularizers
from tensorflow.keras.callbacks import EarlyStopping
from keras.applications.vgg16 import VGG16
from keras.models import Model
import tensorflow as tf


def custom_vgg16(activation = 'relu', layer_num = 10):

    input_shape = (30, 30, 1)
    input_layer = layers.Input(shape=input_shape)
    
    # Convert image to rgb and resize to 32x32
    rgb_layer = layers.Lambda(lambda x: tf.image.grayscale_to_rgb(x), output_shape=(30, 30, 3))(input_layer)
    resized_layer = layers.Lambda(lambda x: tf.image.resize(x, (32, 32)),output_shape=(32, 32, 3))(rgb_layer)
    
    # load the VGG16 model without the top FC layer
    vgg_model = VGG16(include_top=False, input_shape=(32, 32, 3))

    # truncate to keep the first few layers
    truncated_vgg16 = Model(inputs=vgg_model.input, outputs=vgg_model.layers[layer_num].output)
        
    truncated_vgg16.summary()
    
    conv3 = layers.Conv2D(32, (3, 3), activation=activation, padding='same')(truncated_vgg16(resized_layer))
    conv4 = layers.Conv2D(16, (3, 3), activation= activation, padding='same')(conv3)
    pool2 = layers.MaxPooling2D(2, 2)(conv4)
    
    flat = layers.Flatten()(pool2)
    dense1 = layers.Dense(128, activation=activation)(flat)
    output = layers.Dense(4, activation = "softmax")(dense1)
    
    # define model input and output
    model = Model(inputs=input_layer, outputs=output)
    
    # summarize
    model.summary()
    

    model.compile(optimizer = optimizers.Adam(learning_rate= 0.001), 
                loss = "categorical_crossentropy",
                metrics = ["accuracy",
                            "precision",
                            "recall",
                            "F1Score"])

    return model