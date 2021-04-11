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
        f'mean vector of class 0: {m0}',
        f'mean vector of class 1: {m1}'
    )
    return m0, m1


def cal_within_class_scatter_matrix(classified_vector, m0, m1):
    classified_sum = {}

    for label in classified_vector:
        m = m0 if label == 0 else m1

        for data in classified_vector[label]:
            diff = np.array([data - m])
            dot = np.matmul(diff.T, diff)
            print('data-m', diff)
            print('dot', dot)
            print()

            if label not in classified_sum:
                classified_sum[label] = dot
            else:
                classified_sum[label] += dot

    print('classified_sum', classified_sum)

    sw = classified_sum[0] + classified_sum[1]
    print(f'Within-class scatter matrix SW:\n{sw}')
    return sw


def cal_between_class_scatter_matrix(m0, m1):
    diff = np.array([m1 - m0])
    sb = np.matmul(diff.T, diff)
    print(f'Between-class scatter matrix SB:\n{sb}')
    return sb


def cal_weight(sw, m0, m1):
    sw_inv = np.linalg.inv(sw)
    diff = np.array([m1 - m0])
    w = np.matmul(sw_inv, diff.T)
    print(f'Fisher’s linear discriminant:\n{w}')
    return w


def project_data(datasets, w):
    projection_array = np.array([[]])
    a = w[0] / w[1]
    b = w[1] / w[0]
    c = 1 / (a + b)
    for data in datasets:
        x_proj = (a * data[0] + data[1]) * c
        y_proj = b * x_proj
        projtion_vector = [x_proj[0], y_proj[0]]
        if not projection_array.any():
            projection_array = np.array([projtion_vector])
        else:
            projection_array = np.append(
                projection_array,
                [projtion_vector],
                axis=0
            )
    return projection_array


def FLD(data, sw, sb, m0, m1, w):
    a = np.matmul(
        np.matmul(w.T, sb),
        w
    )
    b = np.matmul(
        np.matmul(w.T, sw),
        w
    )
    print ('a', a, b, data)
    print(a/b)
    return a/b


def main():
    # Read the datasets
    x_train = np.load(x_train_data_path)
    y_train = np.load(y_train_data_path)
    x_test = np.load(x_test_data_path)
    y_test = np.load(y_test_data_path)

    data_x = x_train
    data_y = y_train
    # data_x = np.array([[1.0, 1.0], [2.0, 2.3], [-6.5, 1.6], [1.0, -1.0], [2.0, -2.5]])
    # data_y = np.array([1, 1, 1, 0, 0])
    train_classified_vector = classify_data(x_train, y_train)
    m0, m1 = cal_mean(train_classified_vector)
    sw = cal_within_class_scatter_matrix(train_classified_vector, m0, m1)
    sb = cal_between_class_scatter_matrix(m0, m1)
    w = cal_weight(sw, m0, m1)
    
    # print(x_train)
    # print(y_train)
    # print(x_test)
    # print(y_test)


    test_classified_vector = classify_data(x_test, y_test)
    # test_classified_vector = classify_data(x_train, y_train)
    
    for label in test_classified_vector:
        datasets = test_classified_vector[label]
        datasets_project = project_data(datasets, w)
        dot_style = 'ro' if label == 0 else 'bo'

        for i in range(len(datasets)):
            plt.plot(
                [datasets[i][0], datasets_project[i][0]],
                [datasets[i][1], datasets_project[i][1]],
                color='steelblue',
                linewidth=0.5
            )

        plt.plot(
            [data[0] for data in datasets],
            [data[1] for data in datasets],
            dot_style
        )
        plt.plot(
            [data[0] for data in datasets_project],
            [data[1] for data in datasets_project],
            dot_style
        )

    slope = (w[1] / w[0])[0]
    x = np.linspace(-4, 4, 1000)
    plt.plot(x, slope * x )
    plt.title(f'Projection Line: w={slope}, b={0.0}')
    plt.show()

    


if __name__ == '__main__':
    main()