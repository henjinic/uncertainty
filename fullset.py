"""fullset.py

1. to create fullset from maps

Hyeonjin Kim
2019.08.27
"""
import common
import numpy as np

MAP_NAMES = common.DISASTER_NAMES + common.FEATURE_NAMES
INPUT_PATHS = [f'{common.BASE_DIR}/maps/{name}.txt' for name in MAP_NAMES]
OUTPUT_PATH = f'{common.BASE_DIR}/fullset.csv'

def correct_nodata(matrix):
    matrix[matrix == -9999] = -99
    return matrix

def main():
    matrices = [correct_nodata(common.load_data(path, skip=6)) for path in INPUT_PATHS]

    matrix = np.dstack(matrices)
    rows = matrix.reshape(-1, len(MAP_NAMES))

    common.save_data(rows, OUTPUT_PATH)

if __name__ == '__main__':
    main()
