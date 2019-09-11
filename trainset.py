"""trainset.py

2.
to extract certain rows from fullset,
label by predefined classes,
and create trainset.

Hyeonjin Kim
2019.08.27
"""
import common
import numpy as np

FULLSET_PATH = f'{common.BASE_DIR}/fullset.csv'
OUTPUT_PATH = f'{common.BASE_DIR}/trainset.csv'

def get_trainset(fullset):
    result = []
    for row in fullset:
        disaster_prob_vec = row[:common.N_DISASTER]
        feature_vec = row[common.N_DISASTER:]

        categorized_vec = common.categorize(disaster_prob_vec, 0.4, 0.6)

        if common.is_certain(categorized_vec) and -99 not in feature_vec:
            label = common.classify(categorized_vec)
            result.append(feature_vec.tolist() + [label])

    return np.array(result)

def main():
    fullset = common.load_data(FULLSET_PATH, sep=',')

    trainset = get_trainset(fullset)

    common.save_data(trainset, OUTPUT_PATH)

if __name__ == '__main__':
    main()
