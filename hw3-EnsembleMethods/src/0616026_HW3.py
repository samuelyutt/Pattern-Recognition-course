import csv
import math
import random
import numpy as np
import matplotlib.pyplot as plt

# Datasets path
train_data_paths = {
    'x': '../data/x_train.csv',
    'y': '../data/y_train.csv'
}
test_data_paths = {
    'x': '../data/x_test.csv',
    'y': '../data/y_test.csv'
}


def data_process(data_paths):
    data = []
    x_data_path = data_paths['x']
    y_data_path = data_paths['y']

    with open(x_data_path, newline='') as csvfile:
        x_data = list(csv.reader(csvfile))
    with open(y_data_path, newline='') as csvfile:
        y_data = list(csv.reader(csvfile))

    for i in range(1, len(x_data)):
        data.append({
            'x': [float(x) for x in x_data[i]],
            'y': int(y_data[i][0])
        })

    return data


def targets_classifier(data, sequence):
    target_cnts = {}

    for idx in sequence:
        target = data[idx]['y']
        if target in target_cnts:
            target_cnts[target] += 1
        else:
            target_cnts[target] = 1

    return target_cnts


def gini(data, sequence):
    gini_val = 0
    total_cnt = len(sequence)
    target_cnts = targets_classifier(data, sequence)

    for target in target_cnts:
        p = target_cnts[target] / total_cnt
        gini_val += p ** 2

    return gini_val


def entropy(data, sequence):
    entropy_val = 0
    total_cnt = len(sequence)
    target_cnts = targets_classifier(data, sequence)

    for target in target_cnts:
        p = target_cnts[target] / total_cnt
        entropy_val -= p * math.log(p)

    return entropy_val


def main():
    # Read the datasets
    train_data = data_process(train_data_paths)
    entropy_val = entropy(train_data, [1, 2])
    gini_val = gini(train_data, [1, 1, 2])
    print(entropy_val)



if __name__ == '__main__':
    main()
