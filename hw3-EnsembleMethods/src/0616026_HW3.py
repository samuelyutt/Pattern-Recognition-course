import csv
import math
import random
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
    # Remove the header row of input data
    # Transform data type from string to float
    # Return processed date and feature names
    data = []
    x_data_path = data_paths['x']
    y_data_path = data_paths['y']

    with open(x_data_path, newline='') as csvfile:
        x_data = list(csv.reader(csvfile))
    with open(y_data_path, newline='') as csvfile:
        y_data = list(csv.reader(csvfile))

    feature_names = x_data[0]

    for i in range(1, len(x_data)):
        data.append({
            'x': [float(x) for x in x_data[i]],
            'y': int(y_data[i][0])
        })

    return data, feature_names


def targets_classifier(data, sequence):
    # Return each target's count of the sequence
    target_cnts = {}

    for idx in sequence:
        target = data[idx]['y']
        if target in target_cnts:
            target_cnts[target] += 1
        else:
            target_cnts[target] = 1

    return target_cnts


def sequence_classifier(data, sequence, attribute, threshold):
    # Return two sequences seperated by the given threshold of the
    # selected attribute
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
    # Calculate the gini index of the given sequence
    gini_val = 0.0
    total_cnt = len(sequence)
    target_cnts = targets_classifier(data, sequence)

    for target in target_cnts:
        p = target_cnts[target] / total_cnt
        gini_val += p ** 2

    return gini_val


def entropy(data, sequence):
    # Calculate the entropy of the given sequence
    entropy_val = 0.0
    total_cnt = len(sequence)
    target_cnts = targets_classifier(data, sequence)

    for target in target_cnts:
        p = target_cnts[target] / total_cnt
        entropy_val -= p * math.log(p)

    return entropy_val


def testing(tree_or_forest, test_data):
    # Return the accuracy of the predictions of test data by
    # the given decision tree or random forest
    results = tree_or_forest.predict_all(test_data)
    correctness = [results[i] == test_data[i]['y'] for i in range(len(test_data))]
    accuracy = sum(correctness) / len(correctness)
    return accuracy


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
        # Make this node type of decide
        # Assign the decision for this node when traversed
        self.type = 'decide'
        target_cnts = targets_classifier(self.data, self.sequence)
        max_target_cnt = -1

        for target in target_cnts:
            if target_cnts[target] > max_target_cnt:
                max_target_cnt = target_cnts[target]
                self.decision = target


    def split(self):
        # Make this node type of split
        # lt stands for less than
        # ge stands for greater equal
        self.type = 'split'
        self.attribute = 0
        self.threshold = 0.0
        max_info_gain_or_gini = None
        split_sequence_lt = []
        split_sequence_ge = []

        # Find the best attribute and threshold to split
        for tmp_attribute in self.attributes:
            for idx in self.sequence:
                info_gain_or_gini = None
                tmp_threshold = self.data[idx]['x'][tmp_attribute]
                sequence_lt, sequence_ge = sequence_classifier(self.data, self.sequence, tmp_attribute, tmp_threshold)

                if sequence_lt == [] or sequence_ge == []:
                    # If one of the seperated sequence is empty, then the other sequence
                    # is the same as the current sequence, so this division is unnecessary
                    continue

                if self.criterion == 'gini':
                    # Calculate new gini index
                    gini_lt = gini(self.data, sequence_lt)
                    gini_ge = gini(self.data, sequence_ge)
                    avg_gini_val = (gini_lt * len(sequence_lt) + gini_ge * len(sequence_ge)) / len(self.sequence)
                    info_gain_or_gini = avg_gini_val
                elif self.criterion == 'entropy':
                    # Calculate information gain
                    entropy_lt = entropy(self.data, sequence_lt)
                    entropy_ge = entropy(self.data, sequence_ge)
                    avg_entropy_val = (entropy_lt * len(sequence_lt) + entropy_ge * len(sequence_ge)) / len(self.sequence)
                    info_gain_or_gini = self.entropy_val - avg_entropy_val

                if max_info_gain_or_gini == None or info_gain_or_gini > max_info_gain_or_gini:
                    # Update max_info_gain_or_gini, self.attribute, self.threshold,
                    # split_sequence_lt, split_sequence_ge when a better attribute
                    # or threshold is found
                    max_info_gain_or_gini = info_gain_or_gini
                    self.attribute = tmp_attribute
                    self.threshold = tmp_threshold
                    split_sequence_lt = sequence_lt
                    split_sequence_ge = sequence_ge

        # Construct two chiid nodes according to the best attribute and threshold found
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
        # Return the decision of this node when traversed
        if self.type == 'decide':
            return self.decision
        elif self.type == 'split':
            if x_data[self.attribute] < self.threshold:
                return self.node_lt.traverse(x_data)
            else:
                return self.node_ge.traverse(x_data)


    def count_features(self):
        # Return the used features count bellow this node when traversed
        if self.type == 'decide':
            return {}
        elif self.type == 'split':
            features_cnt_total = {self.attribute: 1}
            features_cnt_lt = self.node_lt.count_features()
            features_cnt_ge = self.node_ge.count_features()
            for features_cnt in [features_cnt_lt, features_cnt_ge]:
                for attribute in features_cnt:
                    if attribute in features_cnt_total:
                        features_cnt_total[attribute] += features_cnt[attribute]
                    else:
                        features_cnt_total[attribute] = features_cnt[attribute]
            return features_cnt_total


class DecisionTree():
    def __init__(self, data, sequence, attributes, criterion='gini', max_depth=None):
        self.data = data
        self.sequence = sequence
        self.attributes = attributes
        self.criterion = criterion
        self.max_depth = max_depth
        self.root = Node(
            self.data, self.sequence, self.attributes,
            0, self.criterion, self.max_depth
        )


    def predict(self, x_data):
        # Return the prediciton of the given data predicted by this tree
        return self.root.traverse(x_data)


    def predict_all(self, test_data):
        # Return a list of the predicitons of the given data
        # predicted by this tree
        results = []
        for data in test_data:
            prediction = self.predict(data['x'])
            results.append(prediction)
        return results

    def count_features(self):
        # Return the used features count of this tree
        return self.root.count_features()


class RandomForest():
    def __init__(self, train_data, n_estimators, max_features, bootstrap, criterion='gini', max_depth=None):
        self.trees = []

        all_sequence = [i for i in range(len(train_data))]
        all_attributes = [i for i in range(len(train_data[0]['x']))]

        for i in range(n_estimators):
            selected_sequence = []
            if bootstrap:
                # Bootstrap sampling
                # Create training sequence by drawing at random
                # Each chosen sequence is NOT removed from the orginal set
                for _ in range(len(all_sequence)):
                    selected_sequence.append(random.choice(all_sequence))
            else:
                selected_sequence = all_sequence
            
            # Random choose max_features of all attributes for the tree
            selected_attributes = random.sample(all_attributes, max_features)

            # Construct and append the tree to self.trees
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
        # Return the prediciton of the given data predicted by this forest
        predictions = []
        for tree in self.trees:
            predictions.append(tree.predict(x_data))
        return max(predictions, key=predictions.count)


    def predict_all(self, test_data):
        # Return a list of the predicitons of the given data
        # predicted by this forest
        results = []
        for data in test_data:
            prediction = self.predict(data['x'])
            results.append(prediction)
        return results


def main():
    # Read the datasets and initial params
    train_data, feature_names = data_process(train_data_paths)
    test_data, feature_names = data_process(test_data_paths)
    all_sequence = [i for i in range(len(train_data))]
    all_attributes = [i for i in range(len(train_data[0]['x']))]


    """
    Question 2.1

    Using Criterion=‘gini’ to train the model and show the accuracy score of
    test data by Max_depth=3 and Max_depth=10, respectively.
    """

    clf_depth3 = DecisionTree(
        train_data, all_sequence, all_attributes,
        criterion='gini',
        max_depth=3
    )
    clf_depth10 = DecisionTree(
        train_data, all_sequence, all_attributes,
        criterion='gini',
        max_depth=10
    )
    clf_depth3_accuracy = testing(clf_depth3, test_data)
    clf_depth10_accuracy = testing(clf_depth10, test_data)
    print(f'Accuracy of clf_depth3:', clf_depth3_accuracy)
    print(f'Accuracy of clf_depth10:', clf_depth10_accuracy)


    """
    Question 2.2

    Using Max_depth=3, showing the accuracy score of test data by
    Criterion=‘gini’ and Criterion=’entropy’, respectively.
    """
    
    clf_gini = DecisionTree(
        train_data, all_sequence, all_attributes,
        criterion='gini',
        max_depth=3
    )
    clf_entropy = DecisionTree(
        train_data, all_sequence, all_attributes,
        criterion='entropy',
        max_depth=3
    )
    clf_gini_accuracy = testing(clf_gini, test_data)
    clf_entropy_accuracy = testing(clf_entropy, test_data)
    print(f'Accuracy of clf_gini:', clf_gini_accuracy)
    print(f'Accuracy of clf_entropy:', clf_entropy_accuracy)


    """
    Question 3

    Plot the feature importance of your Decision Tree model.
    You can use the model for Question 2.1, max_depth=10.
    You can simply count the number of a feature used in the tree,
    instead of the formula in the reference.
    Find more details on the sample code.

    To continue running, please close the plot window after it is shown.
    """

    clf_depth10_features_cnt = clf_depth10.count_features()
    sorted_features_cnt = sorted(
        clf_depth10_features_cnt.items(),
        key=lambda x:x[1]
    )
    plt.title('Feature Importance')
    plt.barh(
        [feature_names[pair[0]] for pair in sorted_features_cnt],
        [pair[1] for pair in sorted_features_cnt],
    )
    plt.subplots_adjust(left=0.3)
    plt.show()


    """
    Question 4.1

    Using Criterion=‘gini’, Max_depth=None, Max_features=sqrt(n_features), Bootstrap=True
    to train the model and show the accuracy score of test data by
    n_estimators=10 and n_estimators=100, respectively.
    """

    clf_10tree = RandomForest(
        train_data,
        n_estimators=10,
        max_features=int(math.sqrt(len(all_attributes))),
        bootstrap=True,
        criterion='gini',
        max_depth=None
    )
    clf_100tree = RandomForest(
        train_data,
        n_estimators=100,
        max_features=int(math.sqrt(len(all_attributes))),
        bootstrap=True,
        criterion='gini',
        max_depth=None
    )
    clf_10tree_accuracy = testing(clf_10tree, test_data)
    clf_100tree_accuracy = testing(clf_100tree, test_data)
    print(f'Accuracy of clf_10tree:', clf_10tree_accuracy)
    print(f'Accuracy of clf_100tree:', clf_100tree_accuracy)


    """
    Question 4.2

    Using Criterion=‘gini’, Max_depth=None, N_estimators=10,
    showing the accuracy score of test data by
    Max_features=sqrt(n_features) and Max_features=n_features, respectively.
    """

    clf_random_features = RandomForest(
        train_data,
        n_estimators=10,
        max_features=int(math.sqrt(len(all_attributes))),
        bootstrap=False,
        criterion='gini',
        max_depth=None
    )
    clf_all_features = RandomForest(
        train_data,
        n_estimators=10,
        max_features=len(all_attributes),
        bootstrap=False,
        criterion='gini',
        max_depth=None
    )
    clf_random_features_accuracy = testing(clf_random_features, test_data)
    clf_all_features_accuracy = testing(clf_all_features, test_data)
    print(f'Accuracy of clf_random_features:', clf_random_features_accuracy)
    print(f'Accuracy of clf_all_features:', clf_all_features_accuracy)


if __name__ == '__main__':
    main()
