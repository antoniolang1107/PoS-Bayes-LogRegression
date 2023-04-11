import numpy as np

def train_logistic_regression(train_data):
    epoch_limit = 5

    num_samples, num_features = train_data.shape

    weights = np.zeros((num_features,1))
    bias = 0

    normalized_data = normalize_data(train_data[:,len(train_data)-1:])
    labels = train_data[:,-1]

    for _ in range(epoch_limit):
        for index, sample in enumerate(normalized_data):
            prediction = sigmoid_activation(sample, weights, bias)
            weight_update, bias_update = update_values(sample, labels[index],prediction)

            weights -= weight_update
            bias -= bias_update


    return {"weights": weights,
            "bias": bias}

def test_logistic_regression(test_data, model):
    """
    Input: test_data, trained weights and biases
    Return: Model's accuracy
    """
    num_correct = 0
    slice_length = len(test_data[0])-1
    weights = model['weights']
    bias = model['bias']
    for sample in test_data:
        label = sample[-1]
        num_correct += sigmoid_activation(sample[:slice_length],
                                                weights,
                                                bias) == label
    return num_correct / len(test_data)

def sigmoid_activation(sample, weight, bias):
    return 1/(1+np.exp(np.dot(sample, weight) + bias)) >= 0.5

def update_values(sample, label, prediction):
    learning_rate = 0.05
    return learning_rate*(prediction-label)*sample,\
            learning_rate*(prediction-label)

def normalize_data(samples):
    num_samples, num_features = samples.shape

    for _ in range(num_features):
        samples = (samples-samples.mean(axis=0)) / samples.std(axis=0)
    return samples