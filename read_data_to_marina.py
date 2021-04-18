from os import listdir
from os.path import join, isfile

import pandas as pd
from datetime import datetime

def read_data(dir):
    def parse_file_nm_to_datetime(file_nm):
        split = file_nm.split('_')
        date_str = '_'.join(split[-5:])
        # TODO: YOU WOULD PROBABLY LIKE TO CHANGE THIS LINE
        return datetime.strptime(date_str, '%d_%m_%H_%M_%S.csv')

    def paths_to_files(input_dir):
        paths = [(parse_file_nm_to_datetime(file_nm), join(input_dir, file_nm)) for
                    file_nm in
                    listdir(input_dir) if
                    # TODO: YOU WOULD PROBABLY LIKE TO CHANGE THIS LINE (.csv to .npy or something?)
                    isfile(join(input_dir, file_nm)) and file_nm.endswith('.csv')]
        paths.sort(key=lambda tup: tup[0].timestamp())
        paths = [path for _, path in paths]
        return paths

    paths = paths_to_files(dir)
    print('Start to read files from hdd')
    dfs = [pd.read_csv(full_path) for full_path in paths]
    df = pd.concat(dfs)
    return df