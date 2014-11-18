import graphlab as gl
import pandas as pd
import numpy as np


def get_datasets():
    """
    Reads the data into an SFrame and returns a training, validation and test
    set split 60%, 20% and 20% respectively.
    """
    
    full = gl.SFrame.read_csv('data/features.csv')

    train, other = full.random_split(0.6, seed=1969)
    valid, test = other.random_split(0.5, seed=1991)
    
    return train, valid, test


def boosted_trees(train, valid):
    """
    Creates a boosted decision tree model.

    Parameters
    ----------
    train : SFrame
        A GraphLab SFrame containing the training data.
    valid : SFrame
        A GraphLab SFraming containing the validation data.

    Returns
    -------
    out : BoostedTreesClassifier
        A boosted decision tree that targets the column 'bot' and uses the
        columns 'event_counts' and 'std' as features.

    See Also
    --------
    graphlab.com/products/create/docs

    """

    return gl.boosted_trees_classifier.create(train, target='bot', 
            features=['event_counts','std'], validation_set=valid)


def logistic_regression(train, valid):
    """
    Creates a logistic regression model. In this function we take the best 
    classifier from three models using ridge regression, LASSO and a
    combination of both.

    Parameters
    ----------
    train : SFrame
        A GraphLab SFrame containing the training data.
    valid : SFrame
        A GraphLab SFraming containing the validation data.

    Returns
    -------
    best_mod : LogisticClassifier
        A logistic classifier that targets the column 'bot' and uses the
        columns 'event_counts' and 'std' as features.

    See Also
    --------
    graphlab.com/products/create/docs

    """

    model_ridge = gl.logistic_classifier.create(train, target='bot',
            features=['event_counts', 'std'], l2_penalty=0.1,
            l1_penalty=0.0, feature_rescaling=True)

    model_lasso = gl.logistic_classifier.create(train, target='bot',
            features=['event_counts', 'std'], l2_penalty=0.0,
            l1_penalty=1.0, feature_rescaling=True)

    model_enet = gl.logistic_classifier.create(train, target='bot',
            features=['event_counts', 'std'], l2_penalty=0.5,
            l1_penalty=0.5, feature_rescaling=True)

    best_mod = None
    best_acc = 0.0
    for model in [model_ridge, model_lasso, model_enet]:
        acc = model.evaluate(valid, metric='accuracy')['accuracy']
        if acc > best_acc:
            best_mod = model
            best_acc = acc

    return best_mod 


def svm(train, valid):
    """
    Creates a support vector machine (SVM) model. We use the validation set to 
    pick the best model over several different penalties using accuracy as our
    metric for determing the best model.

    Parameters
    ----------
    train : SFrame
        A GraphLab SFrame containing the training data.
    valid : SFrame
        A GraphLab SFraming containing the validation data.

    Returns
    -------
    best_mod : SVMClassifier
        An SVM classifier that targets the column 'bot' and uses the columns
        'event_counts' and 'std' as features.

    See Also
    --------
    graphlab.com/products/create/docs

    """

    best_mod = None
    best_acc = 0.0
    for penalty in [10.0, 1.0, 0.1]:
        model = gl.svm_classifier.create(train, target='bot', 
                features=['event_counts', 'std'], penalty=penalty,
                feature_rescaling=True)
        acc = model.evaluate(valid, metric='accuracy')['accuracy']
        if acc > best_acc:
            best_mod = model
            best_acc = acc

    return best_mod


def neural_network(train, valid):
    """
    Creates a deep neural network using 3 hidden layers of perceptrons. I
    should note here that neural networks require a lot of parameter
    tuning which is how the parameter num_hidden_units was found.

    Parameters
    ----------
    train : SFrame
        A GraphLab SFrame containing the training data.
    valid : SFrame
        A GraphLab SFraming containing the validation data.

    Returns
    -------
    model : NeuralNetClassifier
        A neural network that targets the column 'bot' and uses the columns
        'event_counts' and 'std' as features.

    See Also
    --------
    graphlab.com/products/create/docs

    """

    percpt_net = gl.deeplearning.MultiLayerPerceptrons(num_hidden_layers=3,
            num_hidden_units=[300, 300, 2])

    model = gl.neuralnet_classifier.create(train, target='bot',
            features=['event_counts', 'std'], network=percpt_net,
            max_iterations=30, validation_set=valid, metric='accuracy')

    return model


def balance_data(train):
    """
    Since the number of user examples in our data is greater than the number of
    bot examples it may help to rebalance the data so the number of examples of
    users and bots are equal. This function will add more bot examples so that
    we have a roughly equal number of user and bot examples in our training data.

    Params
    ------
    train : SFrame
        A GraphLab SFrame containing training data.

    Returns
    -------
    train : SFrame
        A GraphLab SFrame containing a more balanced set of training data.

    See Also
    --------
    graphlab.com/products/create/docs

    """

    tmp = train[train['bot'] == 1]
    for i in xrange(len(train[train['bot'] == 0]) // len(tmp)):
        train = train.append(tmp)

    # Turn the data into a numpy array so we can shuffle it.
    nm = train.to_dataframe().values
    np.random.shuffle(nm)

    train = gl.SFrame(pd.DataFrame(nm))
    train.rename({'0': 'client_ip', '1': 'event_counts', '2': 'std', '3': 'bot'})

    return train


if __name__ == '__main__':
    # Get our training, validation and testing sets.
    train, valid, test = get_datasets()

    # Train the models.
    model_btc = boosted_trees(train, valid)
    model_log = logistic_regression(train, valid)
    model_svm = svm(train, valid)

    # For neural networks we also balance our training data.
    train = balance_data(train)
    model_nn = neural_network(train, valid)
    print model_nn.evaluate(test)

    # Print out each model evaluated over the test set.
    for model in [model_btc, model_log, model_svm, model_nn]:
        print model.evaluate(test)

    # We'll save the logistic regression model to deploy using GraphLab's
    # predictive services.
    model_log.save('model.log')

    # Here we're letting graphlab choose a classifier for us.
    train = train.append(valid)
    model = gl.classifier.create(train, target='bot', 
            features=['event_counts', 'std'])
    print model

