# Group 15: Models file

from main import * 
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications import ResNet50, DenseNet121
from tensorflow.keras.models import Model

metrics = ["accuracy",
            keras.metrics.Precision(class_id = 0, name = "gioma_precision"),
            keras.metrics.Recall(class_id = 0, name = "gioma_recall"),
            keras.metrics.Precision(class_id = 1, name = "meningioma_precision"),
            keras.metrics.Recall(class_id = 1, name = "meningioma_recall"),
            keras.metrics.Precision(class_id = 2, name = "notumor_precision"),
            keras.metrics.Recall(class_id = 2, name = "notumor_recall"),
            keras.metrics.Precision(class_id = 3, name = "pituitary_precision"),
            keras.metrics.Recall(class_id = 3, name = "pituitary_recall"),
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
                metrics = metrics)

    return model

def transfer_learning(model_class, resolution,X_train, y_train, X_val, y_val, epochs=10, batch_size=32, optimizer: str = "adam", add_extra_layer: bool = False, finetune=False):
    
    # initialize VGG16 model and make it non trainable. Don't get the last FC layer by setting include_top to false
    base_model = model_class(weights="imagenet", include_top=False, input_shape=(resolution,resolution,3))
    if not finetune:
        base_model.trainable = False 
        for layer in base_model.layers:
            layer.trainable = False
    
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
    model.add(layers.Dense(128, activation='relu'))
    if add_extra_layer:
        model.add(layers.Dropout(0.3))
        model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(4, activation='softmax'))
        
    
    
    # compile model 
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics = metrics,
    )
    
    # add early stopping if accuracy does not change for 3 epochs
    early_stopping = EarlyStopping(monitor='val_loss', mode='max', patience=4,  restore_best_weights=True)
    
    # fit the model 
    model.fit(X_train, y_train, epochs=epochs, validation_data = (X_val, y_val), batch_size=batch_size, callbacks=[early_stopping])
    return model


