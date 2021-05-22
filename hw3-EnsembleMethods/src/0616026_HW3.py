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


def sequence_classifier(data, sequence, attribute, threshold):
    sequence_lt = []
    sequence_ge = []

    for idx in sequence:
        value = data[idx]['x'][attribute]
        if value < threshold:
            sequence_lt.append(idx)
        else:
            sequence_ge.append(idx)

    return sequence_lt, sequence_ge


def gini(data, sequence):
    gini_val = 0.0
    total_cnt = len(sequence)
    target_cnts = targets_classifier(data, sequence)

    for target in target_cnts:
        p = target_cnts[target] / total_cnt
        gini_val += p ** 2

    return gini_val


def entropy(data, sequence):
    entropy_val = 0.0
    total_cnt = len(sequence)
    target_cnts = targets_classifier(data, sequence)

    for target in target_cnts:
        p = target_cnts[target] / total_cnt
        entropy_val -= p * math.log(p)

    return entropy_val


class Node():
    def __init__(self, data, sequence, attributes, cur_depth, criterion='gini', max_depth=None):
        self.data = data
        self.sequence = sequence
        self.attributes = attributes
        self.cur_depth = cur_depth
        self.criterion = criterion
        self.max_depth = max_depth
        self.entropy_val = entropy(data, sequence)

        if self.entropy_val == 0 or (max_depth != None and cur_depth >= max_depth):
            self.decide()
        else:
            self.split()


    def decide(self):
        self.type = 'decide'
        target_cnts = targets_classifier(self.data, self.sequence)
        max_target_cnt = -1

        for target in target_cnts:
            if target_cnts[target] > max_target_cnt:
                max_target_cnt = target_cnts[target]
                self.decision = target


    def split(self):
        self.type = 'split'
        self.attribute = 0
        self.threshold = 0.0
        max_info_gain_or_gini = None
        split_sequence_lt = []
        split_sequence_ge = []
        
        for tmp_attribute in self.attributes:
            for idx in self.sequence:
                info_gain_or_gini = None
                tmp_threshold = self.data[idx]['x'][tmp_attribute]
                sequence_lt, sequence_ge = sequence_classifier(self.data, self.sequence, tmp_attribute, tmp_threshold)

                if sequence_lt == [] or sequence_ge == []:
                    continue

                if self.criterion == 'gini':
                    pass
                elif self.criterion == 'entropy':
                    entropy_lt = entropy(self.data, sequence_lt)
                    entropy_ge = entropy(self.data, sequence_ge)
                    split_entropy_val = (entropy_lt * len(sequence_lt) + entropy_ge * len(sequence_ge))
                    info_gain_or_gini = self.entropy_val - split_entropy_val

                if max_info_gain_or_gini == None or info_gain_or_gini > max_info_gain_or_gini:
                    max_info_gain_or_gini = info_gain_or_gini
                    self.attribute = tmp_attribute
                    self.threshold = tmp_threshold
                    split_sequence_lt = sequence_lt
                    split_sequence_ge = sequence_ge

        self.node_lt = Node(
            self.data,
            split_sequence_lt,
            self.attributes,
            self.cur_depth + 1,
            self.criterion,
            self.max_depth
        )
        self.node_ge = Node(
            self.data,
            split_sequence_ge,
            self.attributes,
            self.cur_depth + 1,
            self.criterion,
            self.max_depth
        )


    def traverse(self, x_data):
        if self.type == 'decide':
            return self.decision
        elif self.type == 'split':
            if x_data[self.attribute] < self.threshold:
                return self.node_lt.traverse(x_data)
            else:
                return self.node_ge.traverse(x_data)


class DecisionTree():
    def __init__(self, data, sequence, attributes, criterion='gini', max_depth=None):
        self.data = data
        self.sequence = sequence
        self.attributes = attributes
        self.criterion = criterion
        self.max_depth = max_depth
        self.root = Node(
            self.data,
            self.sequence,
            self.attributes,
            0,
            self.criterion,
            self.max_depth
        )


    def predict(self, x_data):
        return self.root.traverse(x_data)


    def predict_all(self, test_data):
        results = []
        for data in test_data:
            prediction = self.predict(data['x'])
            results.append(prediction == data['y'])
        return results



def main():
    # Read the datasets
    train_data = data_process(train_data_paths)
    all_sequence = [i for i in range(len(train_data))]
    all_attributes = [i for i in range(len(train_data[0]['x']))]
    
    tree = DecisionTree(train_data, all_sequence, all_attributes, criterion='entropy')

    test_data = data_process(test_data_paths)
    results = tree.predict_all(test_data)
    accuracy = sum(results) / len(results)
    print(results, accuracy)




if __name__ == '__main__':
    main()
