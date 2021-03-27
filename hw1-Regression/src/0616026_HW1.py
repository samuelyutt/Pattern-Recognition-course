import csv
import numpy as np
import matplotlib.pyplot as plt

# Data path
train_data_path = '../data/train_data.csv'
test_data_path = '../data/test_data.csv'

# Params
learning_rate = 0.002
iterations_limit = 1000
weights = {
    'mean_absolute_error': [0.0, 0.0],
    'mean_square_error': [0.0, 0.0]
}

def data_process(data):
    processed_data = []
    for row in data[1:-1]:
        processed_row = [float(row[0]), float(row[1])]
        processed_data.append(processed_row)
    return processed_data

def partial_differential(method, data_point, term):
    if method == 'mean_absolute_error':
        if weights[method][0] + weights[method][1] * data_point[0] > data_point[1]:
            if term == 0:
                return 1
            if term == 1:
                return data_point[0]
        else:
            if term == 0:
                return -1
            if term == 1:
                return -data_point[0]
    elif method == 'mean_square_error':
        if term == 0:
            # W.R.T b0
            # b0 + b1 * x - y
            return 2 * weights[method][0] + weights[method][1] * data_point[0] - data_point[1]
        elif term == 1:
            # W.R.T b1
            # (b0 + b1 * x - y) * x
            return 2 * (weights[method][0] + weights[method][1] * data_point[0] - data_point[1]) * data_point[0]

def mean_of_partial_differential(method, data, term):
    sum_of_partial_differential = 0.0
    for row in data:
        sum_of_partial_differential += partial_differential(method, row, term)
    return sum_of_partial_differential / len(data)

def gradient_descent(method, data):
    new_b0 = weights[method][0] - learning_rate * mean_of_partial_differential(method, data, 0)
    new_b1 = weights[method][1] - learning_rate * mean_of_partial_differential(method, data, 1)
    weights[method][0] = new_b0
    weights[method][1] = new_b1

def error_value(method, data_point):
    prediction = weights[method][0] + data_point[0] * weights[method][1]
    if method == 'mean_absolute_error':
        return abs(prediction - data_point[1])
    elif method == 'mean_square_error':
        return (prediction - data_point[1]) ** 2

def mean_of_error_value(method, data):
    sum_of_error_value = 0.0
    for row in data:
        sum_of_error_value += error_value(method, [row[0], row[1]])
    return sum_of_error_value / len(data)

def main():
    with open(train_data_path, newline='') as csvfile:
        train_data = data_process(list(csv.reader(csvfile)))
    with open(test_data_path, newline='') as csvfile:
        test_data = data_process(list(csv.reader(csvfile)))

    print(len(train_data))

    plot_values = [[], []]
    for row in train_data:
        plot_values[0].append(row[0])
        plot_values[1].append(row[1])
    # plt.plot(plot_values[0], plot_values[1], 'ro')

    plot_values = [[], []]
    for row in test_data:
        plot_values[0].append(row[0])
        plot_values[1].append(row[1])
    # plt.plot(plot_values[0], plot_values[1], 'o')

    for method in weights:
        losses = []
        for iteration_cnt in range(iterations_limit):
            gradient_descent(method, train_data)
            loss = mean_of_error_value(method, train_data)
            # print(iteration_cnt, loss)
            losses.append(loss)

        plt.plot([i for i in range(len(losses))], losses, label=method)
        # x = np.linspace(-3, 3, 100)
        # plt.plot(x, weights[method][0] + weights[method][1] * x, label=method)

    print(weights)
    
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()