from main import *

from keras.callbacks import ReduceLROnPlateau, Callback
from keras_tuner import HyperModel, RandomSearch, Hyperband

# keras tuner selects a combination of all hyperparameters at each run from the available options.
def model_hyperparameter_tuning(hp):

    """Selects a combination of all hyperparameters at each run from the available options """

    # Initialize sequential model
    model = Sequential()
    # input layer
    model.add(Input(shape=(30, 30, 1)))

    # add two conv blocks as per assignment baseline model requirements
    for i in range(1, 3):
        # conv_filters_i is hyperparameter, values are checked between 16 and 128 with 16 step
        # one value will be selected here and then a new one in next run
        conv_filters_1 = hp.Int(f"conv_filters_{i}", min_value=16, max_value=128, step=16)

        # check 3 and 5 kernel values
        kernel_choice_1 = hp.Choice(f'kernel_choice_{i}', values=[3, 5])

        # check 2x2 and 3x3 max pooling
        max_pooling_options_1 = hp.Choice(f'max_pooling_choice_{i}', values=["(2,2)", "(3,3)"])

        # Choice accepts int and str and not tuple, so convert it back to tuple
        max_pooling_choice_1 = eval(max_pooling_options_1)

        # activation function choices
        activation_choice_1 = hp.Choice(f'activation_choice_{i}', values=['relu', 'tanh', 'sigmoid'])

        # add conv block to model with chosen filter, kernel and activation choice
        model.add(layers.Conv2D(filters=conv_filters_1, kernel_size=kernel_choice_1, activation=activation_choice_1))

        # add max pooling layer
        model.add(layers.MaxPooling2D(pool_size=max_pooling_choice_1))

    # Values to test for dense layer
    dense_units_choice = hp.Int('dense_units', min_value=16, max_value=128, step=16)

    # Activation values for dense layer
    activation_choice_3 = hp.Choice('activation_choice_3', values=['relu', 'tanh', 'sigmoid'])

    # Choice of optimizers
    optimizer_choice = hp.Choice('optimizer', values=['adam', 'sgd', 'adadelta', 'adagrad'])

    # Choice of initial learning rate
    lr = hp.Float('lr', min_value=1e-6, max_value=1e-1, sampling='log')

    # convert to 1D layer
    model.add(layers.Flatten())

    # Add FC layer
    model.add(layers.Dense(units=dense_units_choice, activation=activation_choice_3))

    # Add output layer with softmax
    model.add(layers.Dense(4, activation="softmax"))

    # Intitialise the chosen activation function with initial learning rate
    if optimizer_choice == 'adam':
        optimizer = Adam(learning_rate=lr)
    elif optimizer_choice == 'adadelta':
        optimizer = Adadelta(learning_rate=lr)
    elif optimizer_choice == 'sgd':
        optimizer = SGD(learning_rate=lr)
    elif optimizer_choice == 'adagrad':
        optimizer = Adagrad(learning_rate=lr)

    # Compile the model with the chosen optimizer, loss function, and evaluation metric. 
    # categorical_crossentropy is used as it is multi-class classification and metric is accuracy since we have balanced classes.
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
    
    return model

def tune_hyperparameters(model_function, project_name):
    
    """Function to perform hyperparameter tuning using the Hyperband search algorithm.
    
    Arguments:
    model_function: The function defining the model 
    project_name: A unique name for the project to save results, change this for every unique run
    
    Returns:
    tuner: keras tuner object """
    
    # Initialize hyperband tuner.
    tuner = Hyperband(
        model_hyperparameter_tuning, 
        objective='val_accuracy', 
        max_epochs=10,  
        factor=3, 
        directory='hyperparameter_tuning',  
        project_name=project_name  
    )
    
    # Early stopping callback if val loss does not change for 5 epochs
    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
    
    # Reduce learning rate if loss does not change for 3 epochs by 0.2 factor
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.001)
    
    # Start the hyperparameter search using the defined tuner
    tuner.search(
        X_train, y_train, 
        epochs=10, 
        validation_data=(X_val, y_val),  
        verbose=1,  
        callbacks=[stop_early, reduce_lr] 
    )
    return tuner 

def print_tuning_summary(tuner, project_name):
    """
    Print results of hyperparameter tuning amd save to csv
    
    Args:
    tuner: tuner object
    project_name: projec name to use for saving csv
    """
    trials = tuner.oracle.trials

    results = []
    
    for id, trial in trials.items():
        t = trial.hyperparameters.values.copy()
        
        t["val_accuracy"] = trial.metrics.get_best_value("val_accuracy")
        results.append(t)
    
    # Convert results to dataframe
    df_results = pd.DataFrame(results)
    
    df_results.to_csv(f'{project_name}.csv', index=False)

    # Plot score vs each hyperparameter.This will plot some extra hyperparameters that tuner uses internally. 
    for column in df_results.columns:
        if column != 'val_accuracy': 
            plt.figure(figsize=(5, 5))
            sns.scatterplot(data=df_results, x=column, y='val_accuracy')  
            plt.title(f'{column} vs. Val accuracy') 
            plt.xlabel(column)  
            plt.ylabel('Val Accuracy')  
            plt.show()
    
    # Print sorted trial results, first one is best
    tuner.results_summary()
