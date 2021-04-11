import numpy as np
import matplotlib.pyplot as plt

# Datasets path
x_train_data_path = '../data/x_train.npy'
y_train_data_path = '../data/y_train.npy'
x_test_data_path = '../data/x_test.npy'
y_test_data_path = '../data/y_test.npy'


def classify_data(datasets, labels):
    classified_vector = {}

    for i in range(len(datasets)):
        data = [datasets[i][0], datasets[i][1]]
        label = labels[i]

        if label not in classified_vector:
            classified_vector[label] = np.array([data])
        else:
            classified_vector[label] = np.append(
                classified_vector[label],
                [data],
                axis=0
            )

    return classified_vector


def cal_mean(classified_vector):
    m0 = np.mean(classified_vector[0], axis=0)
    m1 = np.mean(classified_vector[1], axis=0)
    print(
        f"mean vector of class 0: {m0}",
        f"mean vector of class 1: {m1}"
    )
    return m0, m1


def cal_within_class_scatter_matrix(classified_vector, m0, m1):
    classified_sum = {}

    for label in classified_vector:
        m = m0 if label == 0 else m1

        for data in classified_vector[label]:
            diff = np.array([data - m])
            dot = np.matmul(diff.T, diff)
            
            if label not in classified_sum:
                classified_sum[label] = dot
            else:
                classified_sum[label] += dot

    sw = classified_sum[0] + classified_sum[1]
    print(f"Within-class scatter matrix SW: {sw}")
    return sw

def cal_between_class_scatter_matrix(m0, m1):
    diff = np.array([m1 - m0])
    sb = np.matmul(diff.T, diff)
    print(f"Between-class scatter matrix SB: {sb}")
    return sb


def main():
    # Read the datasets
    x_train = np.load(x_train_data_path)
    y_train = np.load(y_train_data_path)
    x_test = np.load(x_test_data_path)
    y_test = np.load(y_test_data_path)

    classified_vector = classify_data(x_train, y_train)
    m0, m1 = cal_mean(classified_vector)
    sw = cal_within_class_scatter_matrix(classified_vector, m0, m1)
    sb = cal_between_class_scatter_matrix(m0, m1)
    
    # print(x_train)
    # print(y_train)
    # print(x_test)
    # print(y_test)

    


if __name__ == '__main__':
    main()