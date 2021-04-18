import os
import string
import numpy as np
import pandas as pd
import csv
import PredictByCell


def write_to_csv(data_path):
    cwd = os.getcwd()
    # strong_cells_list = pd.read_csv(f'{cwd}/strong_cells.csv')
    # with open('strong_cells.csv', newline='') as f:
    #     reader = csv.reader(f)
    #     strong_cells_list = list(reader)
    # strong_cells_list = strong_cells_list.pop()
    for folder in os.listdir(data_path):
        for filename in os.listdir(data_path + '/' + str(folder)):
            data_list = list()
            data_matrix = np.load(f'{data_path}/{folder}/{filename}')
            # for cell in strong_cells_list:
            #     import re
            #     split = re.split('[,()]', cell)
            #     row = int(split[1])
            #     col = int(split[2])
            #     data_list.append(data_matrix[row, col])
            data_array = data_matrix.flatten()
            data_list = list(np.asarray(data_array))
            data_list.insert(0, f'{folder}_{filename}')
            data_list.append(3 if '3' in folder else 4)
            with open("full_data.csv", "a") as my_csv:
                csv_writer = csv.writer(my_csv, delimiter=',')
                csv_writer.writerow(map(lambda x: x, data_list))
                my_csv.close()


def main():
    cwd = os.getcwd()
    data_path = f'{cwd}/test2'
    write_to_csv(data_path)


if __name__ == '__main__':
    main()
