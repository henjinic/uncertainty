"""clust.py

1-2. k-modes

Hyeonjin Kim
2019.09.11
"""
from kmodes.kmodes import KModes
import common
import json
import numpy as np

FULLSET_PATH = f'{common.BASE_DIR}/fullset.csv'
RESULT_PATH = f'{common.BASE_DIR}/clust_summary.csv'
JSON_PATH = f'{common.BASE_DIR}/label_to_code.json'

N_CLUST = common.N_CLASS - 1
N_INIT = 100

def get_clust_samples(fullset):
    result = []
    for row in fullset:
        disaster_prob_vec = row[:common.N_DISASTER]
        categorized_vec = common.categorize(disaster_prob_vec, 0.4, 0.6)
        if common.is_certain(categorized_vec) and sum(categorized_vec) != 0:
            result.append(categorized_vec)
    return np.array(result)

def get_label_to_codes(clust_samples, clust_labels):
    result = {}
    for sample, label in zip(clust_samples, clust_labels.tolist()):
        code = common.encode_binary_vec(sample)
        if label in result:
            if code not in result[label]:
                result[label].append(code)
        else:
            result[label] = [code]
    label_to_codes[N_CLUST] = [0]

    for key in result:
        result[key].sort()

    return result

def main():
    fullset = common.load_data(FULLSET_PATH, sep=',')

    clust_samples = get_clust_samples(fullset)

    km = KModes(n_clusters=N_CLUST, n_init=N_INIT, init='Huang', verbose=True)
    clust_labels = km.fit_predict(clust_samples)

    label_to_codes = get_label_to_codes(clust_samples, clust_labels)

    with open(JSON_PATH, 'w') as f:
        json.dump(label_to_codes, f, sort_keys=True)

    common.save_data([[km.cost_]] + km.cluster_centroids_.tolist(), RESULT_PATH)

if __name__ == '__main__':
    main()
