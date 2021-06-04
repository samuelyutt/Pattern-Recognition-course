import csv
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC, SVR

# Datasets path
x_train_data_path = '../data/x_train.npy'
y_train_data_path = '../data/y_train.npy'
x_test_data_path = '../data/x_test.npy'
y_test_data_path = '../data/y_test.npy'

# HW1 datasets path
hw1_train_data_path = '../../hw1-Regression/data/train_data.csv'
hw1_test_data_path = '../../hw1-Regression/data/test_data.csv'

# Params
min_C, C_cnt = (0.1, 7)
min_gamma, gamma_cnt = (0.00001, 7)


def cross_validation(x_train, y_train, k=5):
    cross_validated_data = []
    
    total_data_cnt = len(x_train)
    tmp_fold_data_cnt = total_data_cnt // k
    
    # Initial all indexes
    all_indexes = [i for i in range(total_data_cnt)]
    random.shuffle(all_indexes)

    # Split the indexes into k folds
    folds = []
    fold_start_i = 0
    for fold_i in range(k):
        fold_data_cnt = tmp_fold_data_cnt
        if fold_i < total_data_cnt % k:
            fold_data_cnt += 1
        fold_end_i = fold_start_i + fold_data_cnt
        fold = all_indexes[fold_start_i:fold_end_i]
        folds.append(fold)
        fold_start_i = fold_end_i

    # Contruct splits by folds
    for split_i in range(k):
        # Assign one fold as validition fold
        validation_fold = folds[split_i]
        
        # Assign the rest folds as training folds
        training_folds = []
        for i in range(k):
            if i != split_i:
                training_folds += folds[i]
        
        # Sort the folds
        validation_fold.sort()
        training_folds.sort()
        
        cross_validated_data.append([training_folds, validation_fold])

    return cross_validated_data


def plot_grid_search_results(grid_search_results, C_range, gamma_range):
    data = np.array(grid_search_results)
    fig, ax = plt.subplots()
    plt.imshow(
        data,
        cmap='RdBu'
    )
    plt.title('Hyperparameter Gridsearch')
    plt.xlabel('Gamma Parameter')
    ax.set_xticks(range(len(gamma_range)))
    ax.set_xticklabels(gamma_range)
    plt.ylabel('C Parameter')
    ax.set_yticks(range(len(C_range)))
    ax.set_yticklabels(C_range)
    plt.colorbar()
    for (i, j), z in np.ndenumerate(data):
        ax.text(j, i, '{:0.2f}'.format(z),
            ha='center', va='center', color='white')
    plt.show()


def main():
    C_range = [min_C * 10 ** i for i in range(C_cnt)]
    gamma_range = [min_gamma * 10 ** i for i in range(gamma_cnt)]

    """
    Question 1

    K-fold data partition
    """

    # Read the datasets
    x_train = np.load(x_train_data_path)
    y_train = np.load(y_train_data_path)
    x_test = np.load(x_test_data_path)
    y_test = np.load(y_test_data_path)
    
    # Cross-validation
    cross_validated_data = cross_validation(x_train, y_train, k=5)
    for split in cross_validated_data:
        print(split)

    """
    Question 2
    
    Grid Search & Cross-validation:
    using sklearn.svm.SVC to train a classifier on the provided train set
    and conduct the grid search of C and gamma, “kernel’=’rbf’
    to find the best hyperparameters by cross-validation.
    Print the best hyperparameters you found.
    Note: We suggest use K=5
    """

    # Grid search
    grid_search_results = []
    max_accuracy, best_C, best_gamma = None, None, None
    for C in C_range:
        total_accuracies = []

        for gamma in gamma_range:
            # Apply each pair of C, gamma at a time
            total_accuracy = 0.0
            
            # Calculate average accuracy
            for split in cross_validated_data:
                training_folds = split[0]
                validation_fold = split[1]
                
                # Construct SVM and fit the data
                clf = SVC(C=C, gamma=gamma, kernel='rbf')
                clf.fit(
                    [x_train[idx] for idx in training_folds],
                    [y_train[idx] for idx in training_folds]
                )

                # Predict on validation data
                y_pred = clf.predict(
                    [x_train[idx] for idx in validation_fold]
                )

                # Calculate accuracy
                correct_cnt = np.sum(
                    np.equal(
                        y_pred,
                        np.array([y_train[idx] for idx in validation_fold])
                    )
                )
                total_accuracy += correct_cnt / len(validation_fold)
            
            total_accuracy /= len(cross_validated_data)
            total_accuracies.append(total_accuracy)

            # Update max_accuracy, best_C, best_gamma if a better pair of
            # parameters is found
            if max_accuracy is None or total_accuracy > max_accuracy:
                max_accuracy = total_accuracy
                best_C = C
                best_gamma = gamma
        grid_search_results.append(total_accuracies)

    print(f'Best hyperparameters found: C = {best_C}, gamma = {best_gamma}')

    """
    Question 3
    
    Plot the grid search results of your SVM.
    The x, y represent the hyperparameters of gamma and C, respectively.
    And the color represents the average score of validation folds.
    
    To continue running, please close the plot window after it is shown.
    """
    
    plot_grid_search_results(grid_search_results, C_range, gamma_range)

    """
    Question 4

    Train your SVM model by the best hyperparameters you found from question 2
    on the whole training set and evaluate the performance on the test set.
    """

    clf = SVC(C=best_C, gamma=best_gamma, kernel='rbf')
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    correct_cnt = np.sum(np.equal(y_pred, y_test))
    accuracy = correct_cnt / len(y_pred)
    print('Accuracy score:', accuracy)

    """
    Question 5

    Consider the dataset used in HW1 for regression.
    Please redo the above questions 2 to 4 with the dataset replaced by that
    used in HW1, while the task is changed from classification to regression.
    You should use the SVM regression model RBF kernel with grid search for
    hyperparameters and K-fold cross-validation (you can use any K for
    cross-validation). Then compare the linear regression model you have
    implemented in HW1 with SVM by showing the Mean Square Errors of both
    models on the test set.
    """

    # Precedure 5.1
    # Read the datasets
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    with open(hw1_train_data_path, newline='') as csvfile:
        for row in list(csv.reader(csvfile))[1:]:
            x_train.append([float(row[0])])
            y_train.append(float(row[1]))
    with open(hw1_test_data_path, newline='') as csvfile:
        for row in list(csv.reader(csvfile))[1:]:
            x_test.append([float(row[0])])
            y_test.append(float(row[1]))
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test = np.array(x_test)
    y_test = np.array(y_test)

    # Cross-validation
    cross_validated_data = cross_validation(x_train, y_train, k=5)
    for split in cross_validated_data:
        print(split)

    # Precedure 5.2
    # Grid search
    grid_search_results = []
    min_error, best_C, best_gamma = None, None, None
    for C in C_range:
        total_errors = []

        for gamma in gamma_range:
            # Apply each pair of C, gamma at a time
            total_error = 0.0
            
            # Calculate average mean square error
            for split in cross_validated_data:
                training_folds = split[0]
                validation_fold = split[1]
                
                # Construct SVM and fit the data
                clf = SVR(C=C, gamma=gamma, kernel='rbf')
                clf.fit(
                    [x_train[idx] for idx in training_folds],
                    [y_train[idx] for idx in training_folds]
                )

                # Predict on validation data
                y_pred = clf.predict(
                    [x_train[idx] for idx in validation_fold]
                )

                # Calculate mean square error
                total_error += np.mean(np.square((
                    y_pred -
                    np.array([y_train[idx] for idx in validation_fold])
                )))

            total_error /= len(cross_validated_data)
            total_errors.append(total_error)

            # Update min_error, best_C, best_gamma if a better pair of
            # parameters is found
            if min_error is None or total_error < min_error:
                min_error = total_error
                best_C = C
                best_gamma = gamma
        grid_search_results.append(total_errors)

    print(f'Best hyperparameters found: C = {best_C}, gamma = {best_gamma}')

    # Precedure 5.3
    plot_grid_search_results(grid_search_results, C_range, gamma_range)

    # Precedure 5.4
    # Train the model by the best hyperparameters found on the whole training
    # set and evaluate the performance on the test set
    clf = SVR(C=best_C, gamma=best_gamma, kernel='rbf')
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    mean_error = np.mean(np.square(y_pred - y_test))
    print('Mean error:', mean_error)


if __name__ == '__main__':
    main()
