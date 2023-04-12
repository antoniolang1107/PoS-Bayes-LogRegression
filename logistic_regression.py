import numpy as np
import pandas as pd

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
    weights = model['weights']
    bias = model['bias']
    for sample in test_data.iterrows():
        sample_as_list = list(sample[1])
        label = sample_as_list.pop()
        num_correct += sigmoid_activation(sample_as_list,
                                                weights,
                                                bias) == label
    return num_correct / len(test_data)

def sigmoid_activation(sample, weight, bias):
    return 1/(1+np.exp(np.dot(weight, sample) + bias)) >= 0.5

def update_values(sample, label, prediction):
    learning_rate = 0.05
    return learning_rate*(prediction-label)*sample,\
            learning_rate*(prediction-label)

def normalize_data(samples):
    num_samples, num_features = samples.shape

    for _ in range(num_features):
        samples = (samples-samples.mean(axis=0)) / samples.std(axis=0)
    return samples

def parse_dict(dicts_list):
    samples = []
    labels = []
    for document in dicts_list:
        samples.append(document[0])
        labels.append(document[1])
    df = pd.DataFrame(samples)
    df = df.fillna(0)
    df['label'] = labels
    return df

if __name__ == "__main__":
    data = [({'hello': 1, 'this':1, 'is':1, 'my':1, 'test':1}, 1),
            ({'hello': 1, 'welcome':1, 'to':1, 'my':1, 'spam':1}, 0)]
    test_data = [({'good': 1, 'this':1, 'is':1, 'my':1, 'test':1}, 1),
            ({'pepperoni': 1, 'welcome':1, 'to':1, 'my':1, 'spam':1}, 0),
            ({'hello': 1, 'apples':1, 'is':1, 'my':1, 'test':1}, 1),
            ({'welcome': 1, 'this':1, 'welcome':1, 'my':1, 'test':1}, 0)]
    test_df = parse_dict(test_data)
    
    test_model = {"weights":[1,2,-3,0.1, 0.5, 0, -0.6, 2, -0.3, -0.8, -1], "bias":2.5}
    # train_logistic_regression()
    print(f"log acc: {test_logistic_regression(test_df, test_model)}")