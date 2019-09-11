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

        if common.is_certain(categorized_vec):
            label = common.classify(categorized_vec)
            result.append(feature_vec.tolist() + [label])

    return np.array(result)

# def get_uncertain(dataset):
#     result = []
#     for row in dataset:
#         if -99 in row:
#             continue

#         category_vec = categorize(row[:5], 0.4, 0.6)

#         if -1 not in category_vec:
#             continue

#         result.append(row[5:])
#     return np.array(result)

def main():
    fullset = common.load_data(FULLSET_PATH, sep=',')

    trainset = get_trainset(fullset)

    common.save_data(trainset, OUTPUT_PATH)

if __name__ == '__main__':
    main()
