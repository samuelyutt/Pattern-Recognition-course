import random
import numpy as np
from sklearn.svm import SVC, SVR

# Datasets path
x_train_data_path = '../data/x_train.npy'
y_train_data_path = '../data/y_train.npy'
x_test_data_path = '../data/x_test.npy'
y_test_data_path = '../data/y_test.npy'

# Params
C_interval = (0.01, 1000.0)
gamma_interval = (0.0001, 1000.0)


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


def main():
    # Read the datasets
    x_train = np.load(x_train_data_path)
    y_train = np.load(y_train_data_path)
    x_test = np.load(x_test_data_path)
    y_test = np.load(y_test_data_path)

    cross_validated_data = cross_validation(x_train, y_train)
    # print(cross_validated_data)


    # split = cross_validated_data[0]
    # training_folds = split[0]
    # validation_fold = split[1]
    # # clf = SVC(gamma='auto')
    # clf = SVC(C=10000, gamma=0.0001)

    # print([x_train[i] for i in training_folds])
    # print(np.array([x_train[i] for i in training_folds]))
    
    # clf.fit(
    #     [x_train[i] for i in training_folds],
    #     [y_train[i] for i in training_folds]
    # )

    # predictions = clf.predict(
    #     [x_train[i] for i in validation_fold]
    # )
    # print(np.array([1, 0, 1]))
    # print(np.array([1, 0, 0]))

    # print(np.equal(np.array([1, 0, 1]), np.array([1, 0, 0])))

    
    # correct_cnt = np.sum((np.equal(np.array([1, 0, 1]), np.array([1, 0, 0]))))

    # print(correct_cnt)

    
    min_C, max_C = C_interval
    min_gamma, max_gamma = gamma_interval

    C = min_C
    while C <= max_C:
        gamma = min_gamma
        
        while gamma <= max_gamma:
            clf = SVC(C=C, gamma=gamma)
            print()
            print(C, gamma)
            
            for split in cross_validated_data:
                training_folds = split[0]
                validation_fold = split[1]
                
                clf.fit(
                    [x_train[idx] for idx in training_folds],
                    [y_train[idx] for idx in training_folds]
                )

                predictions = clf.predict(
                    [x_train[idx] for idx in validation_fold]
                )

                correct_cnt = np.sum(
                    np.equal(
                        predictions,
                        np.array([y_train[idx] for idx in validation_fold])
                    )
                )

                print(correct_cnt / len(validation_fold))
            gamma *= 10
        C *= 10


if __name__ == '__main__':
    main()