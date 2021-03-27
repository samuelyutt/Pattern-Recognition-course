import csv
import numpy as np
import matplotlib.pyplot as plt

# Datasets path
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
    # Remove the header row of input data
    # Transform data type from string to float
    processed_data = []
    for row in data[1:-1]:
        processed_row = [float(row[0]), float(row[1])]
        processed_data.append(processed_row)
    return processed_data


def partial_differential(method, data_point, term):
    # Calculate the partial differential value of the given data point W.R.T
    # the given term
    if method == 'mean_absolute_error':
        # For MAE
        if (weights[method][0] + weights[method][1] * data_point[0] >
                data_point[1]):
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
        # For MSE
        if term == 0:
            # W.R.T b0
            # b0 + b1 * x - y
            return (2 * weights[method][0] +
                    weights[method][1] * data_point[0] -
                    data_point[1])
        elif term == 1:
            # W.R.T b1
            # (b0 + b1 * x - y) * x
            return (2 * (weights[method][0] +
                    weights[method][1] * data_point[0] -
                    data_point[1]) * data_point[0])


def mean_of_partial_differential(method, data, term):
    # Sum up the partial differential value of each data
    # Return the mean of the total
    sum_of_partial_differential = 0.0
    for row in data:
        sum_of_partial_differential += partial_differential(method, row, term)
    return sum_of_partial_differential / len(data)


def gradient_descent(method, data):
    # Update the weights
    new_b0 = (weights[method][0] -
              learning_rate * mean_of_partial_differential(method, data, 0))
    new_b1 = (weights[method][1] -
              learning_rate * mean_of_partial_differential(method, data, 1))
    weights[method][0] = new_b0
    weights[method][1] = new_b1


def model(method, data_point):
    # Perdict a new data by the trained weights
    return weights[method][0] + data_point[0] * weights[method][1]


def error_value(method, data_point):
    # Calculate the error value of the given data
    prediction = model(method, data_point)
    if method == 'mean_absolute_error':
        return abs(prediction - data_point[1])
    elif method == 'mean_square_error':
        return (prediction - data_point[1]) ** 2


def mean_of_error_value(method, data):
    # Sum up the error value value of each data
    # Return the mean of the total
    sum_of_error_value = 0.0
    for row in data:
        sum_of_error_value += error_value(method, [row[0], row[1]])
    return sum_of_error_value / len(data)


def main():
    # Read the datasets
    with open(train_data_path, newline='') as csvfile:
        train_data = data_process(list(csv.reader(csvfile)))
    with open(test_data_path, newline='') as csvfile:
        test_data = data_process(list(csv.reader(csvfile)))

    for method in weights:
        # Training
        losses = []
        for iteration_cnt in range(iterations_limit):
            gradient_descent(method, train_data)
            loss = mean_of_error_value(method, train_data)
            losses.append(loss)
        plt.plot([i for i in range(len(losses))], losses, label=method)

        # Testing
        mean_error = mean_of_error_value(method, test_data)
        print(method,
              'between predictions and the ground truths on the testing data:',
              mean_error)

    # Results and plots
    print('learning_rate\t', learning_rate)
    print('iterations\t', iterations_limit)
    print('weights\t\t', weights,
          '(intercepts (β0) and weights (β1) respectively for each method)')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
