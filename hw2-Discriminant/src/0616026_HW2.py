import numpy as np
import matplotlib.pyplot as plt

# Datasets path
x_train_data_path = '../data/x_train.npy'
y_train_data_path = '../data/y_train.npy'
x_test_data_path = '../data/x_test.npy'
y_test_data_path = '../data/y_test.npy'


def classify_data(datasets, labels):
    # Classify the given datasets according to their labels
    # Return the classified vectors
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
    # Calculate the mean vector of each given datasets
    m0 = np.mean(classified_vector[0], axis=0)
    m1 = np.mean(classified_vector[1], axis=0)
    print(
        f'mean vector of class 0: {m0}',
        f'mean vector of class 1: {m1}'
    )
    return m0, m1


def cal_within_class_scatter_matrix(classified_vector, m0, m1):
    # Calcutate the within class scatter matrix
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
    print(f'Within-class scatter matrix SW:\n{sw}')
    return sw


def cal_between_class_scatter_matrix(m0, m1):
    # Calcutate the between class scatter matrix
    diff = np.array([m1 - m0])
    sb = np.matmul(diff.T, diff)
    print(f'Between-class scatter matrix SB:\n{sb}')
    return sb


def cal_weight(sw, m0, m1):
    # Calculate the weights of FLD
    sw_inv = np.linalg.inv(sw)
    diff = np.array([m1 - m0])
    w = np.matmul(sw_inv, diff.T)
    print(f'Fisher’s linear discriminant:\n{w}')
    return w


def project_data(datasets, w):
    # Project the datasets to the projection line
    # according to the given weights
    projection_array = np.array([[]])
    a = w[0] / w[1]
    b = w[1] / w[0]
    c = 1 / (a + b)
    for data in datasets:
        x_proj = (a * data[0] + data[1]) * c
        y_proj = b * x_proj
        projection_vector = [x_proj[0], y_proj[0]]
        if not projection_array.any():
            projection_array = np.array([projection_vector])
        else:
            projection_array = np.append(
                projection_array,
                [projection_vector],
                axis=0
            )
    return projection_array


def cal_nearest_neighbor_idx(data_project, datasets_project):
    # Calculate the index of the nearest neighbor
    # among the projected datasets
    # Note that this function assumes the projection line are NOT vertical
    x_values = np.array([data[0] for data in datasets_project])
    abs_val_array = np.abs(x_values - data_project[0])
    min_diff_idx = abs_val_array.argmin()
    return min_diff_idx


def FLD(train_datasets, train_labels, test_datasets, w):
    # Predict the test datasets according to the given training datasets,
    # training labels, and trained weights
    predictions = np.array([], dtype=np.int64)

    # Project the datasets to the projection line
    x_train_datasets_project = project_data(train_datasets, w)
    x_test_datasets_project = project_data(test_datasets, w)

    for x_test_data in x_test_datasets_project:
        # Get the nearest neighbor's index and
        # predict the testing ata
        nearest_neighbor_idx = cal_nearest_neighbor_idx(
            x_test_data,
            x_train_datasets_project
        )
        prediction = train_labels[nearest_neighbor_idx]
        predictions = np.append(predictions, prediction)

    return predictions


def plot_results(classified_vector, w):
    # Plot the results
    for label in classified_vector:
        datasets = classified_vector[label]
        datasets_project = project_data(datasets, w)

        # Colorize the data with each class
        dot_style = 'ro' if label == 0 else 'bo'

        # Plot the lines between the datasets and projected points
        for i in range(len(datasets)):
            plt.plot(
                [datasets[i][0], datasets_project[i][0]],
                [datasets[i][1], datasets_project[i][1]],
                color='steelblue',
                linewidth=0.5
            )

        # Plot the datasets
        plt.plot(
            [data[0] for data in datasets],
            [data[1] for data in datasets],
            dot_style
        )

        # Plot the projected points on the projection line
        plt.plot(
            [data[0] for data in datasets_project],
            [data[1] for data in datasets_project],
            dot_style
        )

    # Plot the best projection line on the training data
    slope = (w[1] / w[0])[0]
    x = np.linspace(-4, 4, 1000)
    plt.plot(x, slope * x)

    # Show the slope and intercept on the title
    plt.title(f'Projection Line: w={slope}, b={0.0}')
    plt.show()


def main():
    # Read the datasets
    x_train = np.load(x_train_data_path)
    y_train = np.load(y_train_data_path)
    x_test = np.load(x_test_data_path)
    y_test = np.load(y_test_data_path)

    # Training
    train_classified_vector = classify_data(x_train, y_train)
    m0, m1 = cal_mean(train_classified_vector)
    sw = cal_within_class_scatter_matrix(train_classified_vector, m0, m1)
    sb = cal_between_class_scatter_matrix(m0, m1)
    w = cal_weight(sw, m0, m1)

    # Testing
    predictions = FLD(x_train, y_train, x_test, w)
    correct_cnt = np.sum((np.equal(predictions, y_test)))
    acc = correct_cnt / len(x_test)
    print(f'Accuracy of test-set {acc}')

    # Plot the results
    test_classified_vector = classify_data(x_test, y_test)
    plot_results(test_classified_vector, w)


if __name__ == '__main__':
    main()
