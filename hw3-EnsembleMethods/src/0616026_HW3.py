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


def testing(tree_or_forest, test_data):
    results = tree_or_forest.predict_all(test_data)
    correctness = [results[i] == test_data[i]['y'] for i in range(len(test_data))]
    accuracy = sum(correctness) / len(correctness)
    return accuracy, correctness


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
                    gini_lt = gini(self.data, sequence_lt)
                    gini_ge = gini(self.data, sequence_ge)
                    avg_gini_val = (gini_lt * len(sequence_lt) + gini_ge * len(sequence_ge)) / len(self.sequence)
                    info_gain_or_gini = avg_gini_val
                elif self.criterion == 'entropy':
                    entropy_lt = entropy(self.data, sequence_lt)
                    entropy_ge = entropy(self.data, sequence_ge)
                    avg_entropy_val = (entropy_lt * len(sequence_lt) + entropy_ge * len(sequence_ge)) / len(self.sequence)
                    info_gain_or_gini = self.entropy_val - avg_entropy_val

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
            results.append(prediction)
        return results


class RandomForest():
    def __init__(self, train_data, n_estimators, max_features, bootstrap, criterion='gini', max_depth=None):
        self.trees = []

        all_sequence = [i for i in range(len(train_data))]
        all_attributes = [i for i in range(len(train_data[0]['x']))]

        for i in range(n_estimators):
            selected_sequence = []
            if bootstrap:
                for _ in range(len(all_sequence)):
                    selected_sequence.append(random.choice(all_sequence))
            else:
                selected_sequence = all_sequence
            selected_attributes = random.sample(all_attributes, max_features)

            self.trees.append(
                DecisionTree(
                    train_data,
                    selected_sequence,
                    selected_attributes,
                    criterion=criterion,
                    max_depth=max_depth
                )
            )


    def predict(self, x_data):
        predictions = []
        for tree in self.trees:
            predictions.append(tree.predict(x_data))
        return max(predictions, key=predictions.count)


    def predict_all(self, test_data):
        results = []
        for data in test_data:
            prediction = self.predict(data['x'])
            results.append(prediction)
        return results


def main():
    # Read the datasets
    train_data = data_process(train_data_paths)
    all_sequence = [i for i in range(len(train_data))]
    all_attributes = [i for i in range(len(train_data[0]['x']))]
    
    tree_entropy = DecisionTree(train_data, all_sequence, all_attributes, criterion='entropy')
    tree_gini = DecisionTree(train_data, all_sequence, all_attributes, criterion='gini')
    forest_entropy = RandomForest(train_data, n_estimators=5, max_features=15, bootstrap=False, criterion='entropy', max_depth=None)
    forest_gini = RandomForest(train_data, n_estimators=5, max_features=15, bootstrap=False, criterion='gini', max_depth=None)
    forest_entropy_b = RandomForest(train_data, n_estimators=5, max_features=15, bootstrap=True, criterion='entropy', max_depth=None)
    forest_gini_b = RandomForest(train_data, n_estimators=5, max_features=15, bootstrap=True, criterion='gini', max_depth=None)
    
    
    test_data = data_process(test_data_paths)

    accuracy, correctness = testing(tree_entropy, test_data)
    print(accuracy)
    accuracy, correctness = testing(tree_gini, test_data)
    print(accuracy)
    accuracy, correctness = testing(forest_entropy, test_data)
    print(accuracy)
    accuracy, correctness = testing(forest_gini, test_data)
    print(accuracy)
    accuracy, correctness = testing(forest_entropy_b, test_data)
    print(accuracy)
    accuracy, correctness = testing(forest_gini_b, test_data)
    print(accuracy)





if __name__ == '__main__':
    main()
