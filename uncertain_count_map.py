"""uncertain_count_map.py

2-1.

Hyeonjin Kim
2019.10.18
"""
import common
import numpy as np

FULLSET_PATH = f'{common.BASE_DIR}/fullset.csv'
OUTPUT_PATH = f'{common.BASE_DIR}/uncertain_count_map.csv'

def get_trainset(fullset, upward=False, with_id=False):
    result = []

    if upward:
        fullset = np.flipud(fullset.reshape(common.N_ROWS, common.N_COLS, -1)).reshape(-1, fullset.shape[-1])

    for row in fullset:
        disaster_prob_vec = row[:common.N_DISASTER]
        feature_vec = row[common.N_DISASTER:]

        categorized_vec = common.categorize(disaster_prob_vec, 0.4, 0.6)

        if common.is_certain(categorized_vec) and -99 not in feature_vec:
            label = common.classify(categorized_vec)
            result.append(feature_vec.tolist() + [label])

    if with_id:
        return np.concatenate((np.arange(len(result))[:, np.newaxis], result), axis=1)
    else:
        return np.array(result)

def get_uncertain_count_map(fullset):
    result = []
    for row in fullset:
        disaster_prob_vec = row[:common.N_DISASTER]
        feature_vec = row[common.N_DISASTER:]

        categorized_vec = np.array(common.categorize(disaster_prob_vec, 0.4, 0.6))

        if np.count_nonzero(categorized_vec == 9) == 5:
            result.append(-99)
        else:
            result.append(np.count_nonzero(np.logical_or(categorized_vec == 1, categorized_vec == 0)))
    return np.array(result)


def main():
    print('Loading...')
    fullset = common.load_data(FULLSET_PATH, sep=',')

    print('Processing...')
    countmap = get_uncertain_count_map(fullset)

    print('Saving...')
    common.save_map(countmap.reshape(common.N_ROWS, -1), OUTPUT_PATH)

    print('Done!')

if __name__ == '__main__':
    main()
