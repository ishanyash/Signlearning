import numpy as np
import json
import math
import random


def sigmoid(x):
    s = 1.0 / (1.0 + np.exp(-1.0 * x))
    return s


def predict(features, weights):
    '''
    Returns 1D array of probabilities
    that the class label == 1
    '''
    z = np.dot(features, weights)
    return sigmoid(z)


def cost_function(features, labels, weights):
    '''
    Using Mean Absolute Error

    Features:(100,3)
    Labels: (100,1)
    Weights:(3,1)
    Returns 1D matrix of predictions
    Cost = (labels*log(predictions) + (1-labels)
            *log(1-predictions) ) / len(labels)
    '''
    observations = len(labels)
    # print("observations:", observations)

    predictions = predict(features, weights)
    # print("predictions:", predictions)

    # Take the error when label=1
    class1_cost = np.zeros((len(labels), 1))
    class2_cost = np.zeros((len(labels), 1))
    # print("cost", class2_cost)
    for i in range(0, len(labels)):
        if labels[i] == 1:
            class1_cost[i] = -1 * np.log(predictions[i])
        # print("c1", class1_cost)

    # Take the error when label=0
        if labels[i] == 0:
            class2_cost[i] = -1 * np.log(1 - predictions[i])

        # print("c2", np.log(1 - predictions))

    # Take the sum of both costs
    cost = class1_cost + class2_cost

    # Take the average cost
    cost = cost.sum() / observations

    return cost


def update_weights(features, labels, weights, lr):
    '''
    Vectorized Gradient Descent

    Features:(200, 3)
    Labels: (200, 1)
    Weights:(3, 1)
    '''
    N = len(features)

    # 1 - Get Predictions
    predictions = predict(features, weights)

    # 2 Transpose features from (200, 3) to (3, 200)
    # So we can multiply w the (200,1)  cost matrix.
    # Returns a (3,1) matrix holding 3 partial derivatives --
    # one for each feature -- representing the aggregate
    # slope of the cost function across all observations
    gradient = np.dot(np.transpose(features),  predictions - labels)

    # 3 Take the average cost derivative for each feature
    gradient /= N

    # 4 - Multiply the gradient by our learning rate
    gradient *= lr

    # 5 - Subtract from our weights to minimize cost
    weights -= gradient

    return weights


def decision_boundary(prob):
    return 1 if prob >= .5 else 0


def classify(predictions):
    '''
    input  - N element array of predictions between 0 and 1
    output - N element array of 0s (False) and 1s (True)
    '''
    decision_boundary = np.vectorize(decision_boundary)
    return decision_boundary(predictions).flatten()


def train(features, labels, weights, lr, iters):
    cost_history = []

    for i in range(iters):
        weights = update_weights(features, labels, weights, lr)

        # Calculate error for auditing purposes
        cost = cost_function(features, labels, weights)
        cost_history.append(cost)

        # Log Progress
        # if i % 100 == 0:
        print "iter: " + str(i) + " cost: " + str(cost)

    return weights, cost_history

with open("E_data_pos.json") as f:
    data_pos = json.load(f)
    with open("data_neg.json") as f2:
        data_neg = json.load(f2)
        dot_table_pos = []
        dot_table_neg = []
        dot_table = []
        for frame in data_pos:
            dot_table_pos.append(frame['finger_dot_table'])
        for frame in data_neg:
            dot_table_neg.append(frame['finger_dot_table'])

        # print(dot_table_pos)
        print("dot_table_pos : " , len(dot_table_pos))
        print("dot_table_neg : " , len(dot_table_neg))
        dot_table = np.vstack((dot_table_pos, dot_table_neg))
        # print("height:", len(dot_table))
        # print("width:", len(dot_table[0]))
        labels_pos = np.ones((len(dot_table_pos), 1))
        labels_neg = np.zeros((len(dot_table_neg), 1))
        labels = np.vstack((labels_pos, labels_neg))
        weights = np.zeros((len(dot_table_pos[0]), 1))
        # weight_neg = np.zeros((len(dot_table_neg[0]), 1))
        # weights = np.vstack((weight_pos, weight_neg))
        for i in range(0, len(weights)):
            weights[i] = random.uniform(0, 1)

        # print(dot_table)

        # print(len(dot_table))
        # print(len(dot_table[0]))
        # print(len(weight))
        # print(len(weight[0]))
        train(dot_table, labels, weights, 0.01, 1000)

        print(weights)
        with open("E_dot_weights.json", "w") as f:
            json.dump(weights.tolist(), f)
