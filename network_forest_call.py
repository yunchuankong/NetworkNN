## partially referred to https://machinelearningmastery.com/implement-random-forest-scratch-python/
from __future__ import print_function
from random import seed
from random import randrange
import random
from csv import reader
from math import sqrt
from sklearn.utils import shuffle
import numpy as np
from sklearn import metrics
import time

# start_time = time.time()

use_network = True
# use_network = False
n_folds = 5
percent = 0.2 ## n_test / n_total
CV = False
sample_size_ratio = 0.6 ## subsampling ratio
max_depth = 5
min_size = 40
max_features = 20
n_trees = 200

# Load a CSV file
def load_csv(filename):
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset

# Convert string column to float
def str_column_to_float(dataset, column):
    for row in dataset:
        row[column] = float(row[column].strip())

# Convert string column to integer
def str_column_to_int(dataset, column):
    class_values = [row[column] for row in dataset]
    unique = set(class_values)
    lookup = dict()
    for i, value in enumerate(unique):
        lookup[value] = i
    for row in dataset:
        row[column] = lookup[row[column]]
    return lookup


# Split a dataset into k folds
def cross_validation_split(dataset, n_folds):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / n_folds)
    for i in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split

def training_testing_split(dataset, percent):
    cut = int(len(dataset) * percent)
    return dataset[:cut][:], dataset[cut:][:]

# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
    pred_labels = list(zip(*predicted))[0]
    pred_probs = list(zip(*predicted))[1]
    correct = 0
    for i in range(len(actual)):
        if actual[i] == pred_labels[i]:
            correct += 1
    accuracy = correct / float(len(actual)) * 100.0
    auc = metrics.roc_auc_score(actual, pred_probs) * 100.0
    return accuracy, auc


# Evaluate an algorithm using a cross validation split
def evaluate_algorithm(dataset, algorithm, n_folds, percent, CV, *args):
    scores = list()
    if CV:
        folds = cross_validation_split(dataset, n_folds)
        for i, fold in enumerate(folds):
            train_set = list(folds)
            train_set.remove(fold)
            train_set = sum(train_set, [])
            # test_set = fold
            predicted = algorithm(train_set, fold, *args)
            actual = [row[-1] for row in fold]
            accuracy = accuracy_metric(actual, predicted)
            scores.append(accuracy)
            # print("Fold "+str(i+1)+" of "+str(n_folds)+" finished.")
    else:
        test_set, train_set = training_testing_split(dataset, percent)
        predicted = algorithm(train_set, test_set, *args)
        actual = [row[-1] for row in test_set]
        accuracy = accuracy_metric(actual, predicted)
        scores.append(accuracy)
    return scores

# Calculate the Gini index for a split dataset
def gini_index(groups, classes):
    # count all samples at split point
    n_instances = float(sum([len(group) for group in groups]))
    # sum weighted Gini index for each group
    gini = 0.0
    for group in groups:
        size = float(len(group))
        # avoid divide by zero
        if size == 0:
            continue
        score = 0.0
        # score the group based on the score for each class
        for class_val in classes:
            p = [row[-1] for row in group].count(class_val) / size
            score += p * p
            # weight the group score by its relative size
        gini += (1.0 - score) * (size / n_instances)
    return gini


# Search the neighbors for a feature
def get_neighbors(index, network):
    neighbors = []
    for edge in network:
        if index in edge:
            neighbors.append(edge[1-edge.index(index)])
    return neighbors

# Select feature subset according to neighbors using breadth first graph search
def get_features(max_features, network, initial):
    # initial = shuffle(range(n_features))[0]
    features = [initial]
    neighbors = None
    candidates = [initial] ## store the vertices that can be used as center
    used = [] ## store the vertices that have been used as center
    while len(features) < max_features and candidates:
        index = candidates.pop(0)
        used.append(index)
        neighbors = shuffle(get_neighbors(index, network))
        candidates += [item for item in neighbors if item not in used]
        features += [item for item in neighbors if item not in features]
        if len(features) > max_features:
            features = features[:max_features]
    del neighbors
    del candidates
    del used
    # print ("%d features selected." % len(features))
    return features

# Select the best split point for a dataset
def get_split(dataset, max_features, network, initial):
    class_values = list(set(row[-1] for row in dataset))
    b_index, b_value, b_groups = None, None, None ## guess: b_ for best_
    b_score = gini_index([dataset, []], class_values)
    # print b_score
    if use_network:
        features = get_features(max_features, network, initial)
    else:
        features = list()
        while len(features) < max_features:
            index = randrange(len(dataset[0])-1)
            if index not in features:
                features.append(index)
    for index in features:
        for row in dataset:
            groups = [r for r in dataset if r[index] < row[index]], \
                     [r for r in dataset if r[index] >= row[index]]
            gini = gini_index(groups, class_values)
            # print gini
            if gini < b_score:
                var_importance[index] += b_score - gini
                b_index, b_value, b_score, b_groups = index, row[index], gini, groups
    return {'index': b_index, 'value': b_value, 'groups': b_groups}


# Create a terminal node value
def to_terminal(group):
    outcomes = [row[-1] for row in group]
    return max(set(outcomes), key=outcomes.count)


# Create child splits for a node or make terminal
def split(node, max_depth, min_size, max_features, depth, network):
    initial = node['index']
    left, right = node['groups']
    del (node['groups'])
    # check for a no split
    if not left or not right:
        node['left'] = node['right'] = to_terminal(left + right)
        return
    # check for max depth
    if depth >= max_depth:
        node['left'], node['right'] = to_terminal(left), to_terminal(right)
        return
    # process left child
    if len(left) <= min_size:
        node['left'] = to_terminal(left)
    else:
        try:
            node['left'] = get_split(left, max_features, network, initial)
            split(node['left'], max_depth, min_size, max_features, depth + 1, network)
        except:
            node['left'] = to_terminal(left)
    # process right child
    if len(right) <= min_size:
        node['right'] = to_terminal(right)
    else:
        try:
            node['right'] = get_split(right, max_features, network, initial)
            split(node['right'], max_depth, min_size, max_features, depth + 1, network)
        except:
            node["right"] = to_terminal(right)


# Build a decision tree
def build_tree(train, max_depth, min_size, max_features, network):
    seed(1)
    initial = shuffle(range(len(train[0])-1))[0]
    root = get_split(train, max_features, network, initial)
    split(root, max_depth, min_size, max_features, 1, network)
    return root


# Make a prediction with a decision tree
def predict(node, row):
    if row[node['index']] < node['value']:
        if isinstance(node['left'], dict):
            return predict(node['left'], row)
        else:
            return node['left']
    else:
        if isinstance(node['right'], dict):
            return predict(node['right'], row)
        else:
            return node['right']


# Make a prediction with a list of bagged trees
def bagging_predict(trees, row):
    predictions = [predict(tree, row) for tree in trees]
    prob = sum(predictions) / float(len(predictions))
    return max(set(predictions), key=predictions.count), prob


# Create a random subsample from the dataset with replacement
def subsample(dataset, ratio):
    n_samples = int(round(len(dataset) * ratio))
    indices = shuffle(range(len(dataset)), n_samples=n_samples)
    return [row for i,row in enumerate(dataset) if i in indices]


# Random Forest Algorithm
def random_forest(train, test, max_depth, min_size, sample_size_ratio, n_trees, max_features, network):
    trees = list()
    for i in range(n_trees):
        sample = subsample(train, sample_size_ratio)
        tree = build_tree(sample, max_depth, min_size, max_features, network)
        # print ("Tree %d built." %  (i+1))
        trees.append(tree)
    predictions = [bagging_predict(trees, row) for row in test]
    return (predictions)

# Test the random forest algorithm
# load and prepare data
filename = 'C:/Users/yunchuan/Dropbox/Research_Yu/jungle/simulation_ngf/data_expression_sim.csv'
dataset = load_csv(filename)[1:] ## ignore the first row which are the column names generated by R
# convert string attributes to integers
for i in range(0, len(dataset[0]) - 1):
    str_column_to_float(dataset, i)
# convert class column to integers
str_column_to_int(dataset, len(dataset[0]) - 1)

# load and prepare network
filename = 'C:/Users/yunchuan/Dropbox/Research_Yu/jungle/simulation_ngf/data_network_sim.csv'
network = np.array(load_csv(filename)[1:], dtype=int) ## ignore the first row for the same reason
network = network.tolist()

var_importance = [0] * (len(dataset[0]) - 1)

# evaluate the algorithm
scores = evaluate_algorithm(dataset, random_forest,
                            n_folds, percent, CV,
                            max_depth, min_size, sample_size_ratio, n_trees, max_features,
                            network)
# accuracy = list(zip(*scores))[0]
# auc = list(zip(*scores))[1]
# print('Accuracy: %s' % accuracy)
# print('Mean Accuracy: %.1f%%' % (sum(accuracy) / float(len(accuracy))))
# print(auc)
# print(sum(auc) / float(len(auc)*100))
print (scores[0][1]*0.01) ## auc, for train-test split


# np.savetxt("var_importance.csv", var_importance, delimiter=",")


# print("Total time used: %s minutes " % ((time.time() - start_time)/60) )