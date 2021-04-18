import os
import numpy
import scipy.spatial as sp


def classify_by_mean(mean_matrices, test_matrix):
    classify_3 = 0
    classify_4 = 0
    for (row, column), time in numpy.ndenumerate(test_matrix):
        dist_to_3 = abs(mean_matrices['Gen10-3'][row, column] - time)
        dist_to_4 = abs(mean_matrices['Gen10-4'][row, column] - time)
        if dist_to_3 <= dist_to_4:
            classify_3 += 1
        else:
            classify_4 += 1
    return 3 if classify_3 >= classify_4 else 4


def classify_by_strong_cell_mean(mean_matrices, test_matrix, strong_cells):
    print(f'starting classification')
    classify_3 = 0
    classify_4 = 0

    for (row, col) in strong_cells['Gen10-3']:
        dist_to_3 = abs(mean_matrices['Gen10-3'][row, col] - test_matrix[row, col])
        dist_to_4 = abs(mean_matrices['Gen10-4'][row, col] - test_matrix[row, col])
        if dist_to_3 <= dist_to_4:
            classify_3 += 1

    classify_3 = classify_3 / len(strong_cells['Gen10-3'])

    for (row, col) in strong_cells['Gen10-4']:
        dist_to_3 = abs(mean_matrices['Gen10-3'][row, col] - test_matrix[row, col])
        dist_to_4 = abs(mean_matrices['Gen10-4'][row, col] - test_matrix[row, col])
        if dist_to_4 <= dist_to_3:
            classify_4 += 1

    classify_4 = classify_4 / len(strong_cells['Gen10-4'])
    print(f'classification done.')
    return 3 if classify_3 >= classify_4 else 4


def predict_by_cells(test_path, mean_matrices):
    correct = 0
    correct_3 = 0
    correct_4 = 0
    all = 0
    for folder in os.listdir(test_path):
        for filename in os.listdir(test_path + '/' + str(folder)):
            all += 1
            test_matrix = numpy.load(f'{test_path}/{folder}/{filename}')
            # can be 3 or 4 depends on the prediction
            predict_matrix_num = classify_by_mean(mean_matrices, test_matrix)
            if '3' in folder:
                expected_prediction = 3
            else:
                expected_prediction = 4
            if predict_matrix_num == expected_prediction:
                correct += 1
                if predict_matrix_num == 4:
                    correct_4 += 1
                else:
                    correct_3 += 1
    return correct / all


def create_mean_matrices(train_path):
    mean_matrices = dict()
    for folder in os.listdir(train_path):
        all_matrix = list()
        for filename in os.listdir(train_path + '/' + str(folder)):
            data = numpy.load(f'{train_path}/{folder}/{filename}')
            all_matrix.append(numpy.asmatrix(data))
        mean_matrix = numpy.mean(all_matrix, axis=0)
        mean_matrices[folder] = mean_matrix
        numpy.save(f'{train_path}/{folder}/mean_map_{folder}.npy', mean_matrix)
    return mean_matrices


# returns a matrix where a cell with 1 represent that it classified correctly
def get_cells_accuracy(mean_matrices, test_matrix, correct_prediction):
    count_correct_cells = numpy.zeros((371, 371), dtype=float)
    for (row, column), time in numpy.ndenumerate(test_matrix):
        dist_to_3 = abs(mean_matrices['Gen10-3'][row, column] - time)
        dist_to_4 = abs(mean_matrices['Gen10-4'][row, column] - time)
        if (dist_to_3 < dist_to_4 and correct_prediction == 3) or (dist_to_4 < dist_to_3 and correct_prediction == 4):
            count_correct_cells[row, column] += 1
    return count_correct_cells


def get_strong_cells(train_path, mean_matrices):
    strong_cells = dict()
    strong_cells['Gen10-3'] = list()
    strong_cells['Gen10-4'] = list()
    train_size = 0.
    for folder in os.listdir(train_path):
        accuracy_matrix = numpy.zeros((371, 371), dtype=float)
        correct_prediction = 3 if '3' in folder else 4
        for filename in os.listdir(train_path + '/' + str(folder)):
            train_size += 1
            test_matrix = numpy.load(f'{train_path}/{folder}/{filename}')
            correct_cells = get_cells_accuracy(mean_matrices, test_matrix, correct_prediction)
            accuracy_matrix = [[accuracy_matrix[i][j] + correct_cells[i][j] for j in range(len(accuracy_matrix[0]))]
                               for i in range(len(accuracy_matrix))]

        # normalize the matrix by the size of the train set
        accuracy_matrix = numpy.true_divide(accuracy_matrix, train_size)

        # add to list the indexes of all cells that classified corectly over 50%
        for (row, col), val in numpy.ndenumerate(accuracy_matrix):
            if (correct_prediction == 4 and val >= 0.7) or (correct_prediction == 3 and val >= 0.45):
                strong_cells[folder].append((row, col))
    strong_cells['Gen10-4'] = list(set(strong_cells['Gen10-4']) - set(strong_cells['Gen10-3']))
    return strong_cells


def predict_by_strong_cells(test_path, mean_matrices, strong_cells):
    print('starting prediction')
    correct = 0
    correct_3 = 0
    correct_4 = 0
    all = 0
    for folder in os.listdir(test_path):
        print(f'starting folder {test_path}')
        for filename in os.listdir(test_path + '/' + str(folder)):
            all += 1
            test_matrix = numpy.load(f'{test_path}/{folder}/{filename}')
            # can be 3 or 4 depends on the prediction
            predict_matrix_num = classify_by_strong_cell_mean(mean_matrices, test_matrix, strong_cells)
            if '3' in folder:
                expected_prediction = 3
            else:
                expected_prediction = 4
            if predict_matrix_num == expected_prediction:
                correct += 1
                if predict_matrix_num == 4:
                    correct_4 += 1
                else:
                    correct_3 += 1
        print('folder completed')
    return correct / all


def save_strong_cells(strong_cells):
    strong_cells_list = list()
    strong_cells_list += strong_cells['Gen10-3']
    strong_cells_list += strong_cells['Gen10-4']
    import csv
    with open("strong_cells.csv", "a") as my_csv:
        csv_writer = csv.writer(my_csv, delimiter=',')
        csv_writer.writerow(map(lambda x: x, strong_cells_list))
        my_csv.close()


def main():
    cwd = os.getcwd()
    train_path = f'{cwd}/train'
    test_path = f'{cwd}/test'
    test_path2 = f'{cwd}/test2'
    mean_matrices = create_mean_matrices(train_path)
    substruct_matrix = numpy.subtract(mean_matrices['Gen10-3'],mean_matrices['Gen10-4'])
    import matplotlib.pyplot as mt
    mt.matshow(substruct_matrix)
    mt.show()
    strong_cells = get_strong_cells(test_path, mean_matrices)
    save_strong_cells(strong_cells)
    accuracy = predict_by_cells(test_path2, mean_matrices)
    accuracy_strong = predict_by_strong_cells(test_path2, mean_matrices, strong_cells)
    print(f'prediction accuracy by all cells: {accuracy}')
    print(f'prediction accuracy by strong cells: {accuracy_strong}')


if __name__ == '__main__':
    main()
