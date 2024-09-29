def get_metrics(models, X, y, index):
    
    """
    Returns a dataframe with metrics and another with f1 per class.

    Parameters:
    - models: A list of models
    - X: Feature data (validation or test set)
    - y: True labels for the dataset
    - index: An iterable of class labels
    """
    list_metr = []
    list_f1 = []
    for model in models:
        dict = model.evaluate(X, y, batch_size = 32, return_dict = True)
        dict.pop("loss")
        list_f1.append(pd.Series(dict.pop("F1Score"), index = index))
        list_metr.append(dict)
    f1 = pd.DataFrame(list_f1).T
    metr = pd.DataFrame(list_metr).T
    return metr, f1



metrics = ["accuracy", 
           keras.metrics.Precision(class_id = 0), 
           keras.metrics.Recall(class_id = 0)
           keras.metrics.Precision(class_id = 1), 
           keras.metrics.Recall(class_id = 1)
           keras.metrics.Precision(class_id = 2), 
           keras.metrics.Recall(class_id = 2)
           keras.metrics.Precision(class_id = 3), 
           keras.metrics.Recall(class_id = 3),
           "F1Score"]