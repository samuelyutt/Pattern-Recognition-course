import random
import numpy as np
from sklearn.svm import SVC, SVR

# Datasets path
x_train_data_path = '../data/x_train.npy'
y_train_data_path = '../data/y_train.npy'
x_test_data_path = '../data/x_test.npy'
y_test_data_path = '../data/y_test.npy'


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
        validition_fold = folds[split_i]
        
        # Assign the rest folds as training folds
        training_folds = []
        for i in range(k):
            if i != split_i:
                training_folds += folds[i]
        
        # Sort the folds
        validition_fold.sort()
        training_folds.sort()
        
        cross_validated_data.append([training_folds, validition_fold])

    return cross_validated_data


def main():
    # Read the datasets
    x_train = np.load(x_train_data_path)
    y_train = np.load(y_train_data_path)
    x_test = np.load(x_test_data_path)
    y_test = np.load(y_test_data_path)

    cross_validated_data = cross_validation(x_train, y_train)
    print(cross_validated_data)


if __name__ == '__main__':
    main()