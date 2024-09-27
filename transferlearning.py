from main import * 
import tensorflow as tf

from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications import ResNet50, DenseNet121

def transfer_learning(model_class, resolution,X_train, y_train, X_val, y_val, epochs=10, batch_size=32, optimizer: str = "adam"):
    
    # initialize VGG16 model and make it non trainable. Don't get the last FC layer by setting include_top to false
    base_model = model_class(weights="imagenet", include_top=False, input_shape=(resolution,resolution,3))
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
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )
    
    # add early stopping if accuracy does not change for 3 epochs
    early_stopping = EarlyStopping(monitor='val_accuracy', mode='max', patience=3,  restore_best_weights=True)
    
    # fit the model 
    model.fit(X_train, y_train, epochs=epochs, validation_data = (X_val, y_val), batch_size=batch_size, callbacks=[early_stopping])
    return model




