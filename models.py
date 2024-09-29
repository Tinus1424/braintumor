from main import * 
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications import ResNet50, DenseNet121

metrics = ["accuracy",
            keras.metrics.Precision(class_id = 0),
            keras.metrics.Recall(class_id = 0),
            keras.metrics.Precision(class_id = 1),
            keras.metrics.Recall(class_id = 1),
            keras.metrics.Precision(class_id = 2),
            keras.metrics.Recall(class_id = 2),
            keras.metrics.Precision(class_id = 3),
            keras.metrics.Recall(class_id = 3),
           "F1Score"]

def baseline():
    baseline = Sequential()
    baseline.add(Input(shape = (30, 30, 1)))
    baseline.add(layers.Conv2D(32, (3, 3), activation = "relu"))
    baseline.add(layers.MaxPooling2D((2, 2)))
    baseline.add(layers.Conv2D(32, (3, 3), activation = "relu"))
    baseline.add(layers.MaxPooling2D((2, 2)))
    baseline.add(layers.Flatten())
    baseline.add(layers.Dense(32, activation = "relu"))
    baseline.add(layers.Dense(4, activation = "softmax"))
    baseline.compile(optimizer = "adam",
                    loss = "categorical_crossentropy",
                    metrics = metrics)
    return baseline
    
def cnn300k(activation = 'relu'):

    model = Sequential()
    model.add(Input(shape = (30, 30, 1)))

    model.add(layers.Conv2D(128, (3, 3), activation=activation, padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(64, (3, 3), activation=activation, padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2), strides=2))

    model.add(layers.Conv2D(32, (3, 3), activation=activation, padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(16, (3, 3), activation= activation, padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(2, 2))

    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation=activation))
    model.add(layers.Dropout(rate = 0.5))
    model.add(layers.Dense(32, activation = activation))
    model.add(layers.Dropout(rate = 0.25))
    model.add(layers.Dense(16, activation = activation))
    model.add(layers.Dense(4, activation = "softmax"))

    model.compile(optimizer = optimizers.Adam(learning_rate= 0.001), 
                loss = "categorical_crossentropy",
                metrics = metrics)
    return model

def vgg16():
    resolution = 224
    # initialize VGG16 model and make it non trainable. Don't get the last FC layer by setting include_top to false
    base_model = VGG16(weights="imagenet", include_top=False, input_shape=(resolution,resolution,3))
    base_model.trainable = False 
    for layer in base_model.layers:
        base_model.trainable = False
    
    # initialise model 
    model = Sequential()
    # add vgg16,resnet,densenet compliant input
    model.add(Input(shape = (resolution, resolution, 1)))
    
    # change our image to resolutionxresolution
    model.add(layers.Lambda(lambda x: tf.image.resize(x, (resolution, resolution)))) 
    
    # change image to rgb
    model.add(layers.Lambda(lambda x: tf.image.grayscale_to_rgb(x)))
    
    # add base model to be used for transfer learning
    model.add(base_model)
    model.add(layers.Flatten())
        
    model.add(layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    
    model.add(layers.Dense(4, activation='softmax'))
    
    # compile model 
    model.compile(
        optimizer="adam",
        loss='categorical_crossentropy',
        metrics= metrics,
    )
    return model

def resnet():
    resolution = 224 
    base_model = ResNet50(weights="imagenet", include_top=False, input_shape=(resolution, resolution,3))
    base_model.trainable = False ## 
    
    model = Sequential()
    model.add(Input(shape = (resolution, resolution, 1)))
    model.add(layers.Lambda(lambda x: tf.image.resize(x, (resolution, resolution)))) 
    model.add(layers.Lambda(lambda x: tf.image.grayscale_to_rgb(x)))
    model.add(base_model)
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(4, activation='softmax'))
    
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics= metrics,
    )
    return model

def densenet():
    resolution = 224
    base_model = DenseNet121(weights="imagenet", include_top=False, input_shape=(resolution, resolution, 3))
    base_model.trainable = False ## 
    
    model = Sequential()
    model.add(Input(shape = (resolution, resolution, 1)))
    model.add(layers.Lambda(lambda x: tf.image.resize(x, (resolution, resolution)))) 
    model.add(layers.Lambda(lambda x: tf.image.grayscale_to_rgb(x)))
    model.add(base_model)
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(4, activation='softmax'))
    
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=metrics,
    )
    return model
