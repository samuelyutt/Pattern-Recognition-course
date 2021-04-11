import numpy as np
import matplotlib.pyplot as plt

# Datasets path
x_train_data_path = '../data/x_train.npy'
y_train_data_path = '../data/y_train.npy'
x_test_data_path = '../data/x_test.npy'
y_test_data_path = '../data/y_test.npy'


def cal_mean(datasets, labels):
    classified_vector = {}

    for i in range(len(datasets)):
        new_data = [datasets[i][0], datasets[i][1]]
        label = labels[i]

        if label not in classified_vector:
            classified_vector[label] = np.array([new_data])
        else:
            classified_vector[label] = np.append(
                classified_vector[label],
                [new_data],
                axis=0
            )
    
    m0 = np.mean(classified_vector[0], axis=0)
    m1 = np.mean(classified_vector[1], axis=0)
    print(
        f"mean vector of class 0: {m0}",
        f"mean vector of class 1: {m1}"
    )
    return m0, m1


def main():
    # Read the datasets
    x_train = np.load(x_train_data_path)
    y_train = np.load(y_train_data_path)
    x_test = np.load(x_test_data_path)
    y_test = np.load(y_test_data_path)

    m0, m1 = cal_mean(x_train, y_train)
    
    print(x_train)
    print(y_train)
    # print(x_test)
    # print(y_test)

    


if __name__ == '__main__':
    main()