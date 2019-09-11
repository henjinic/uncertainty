"""result.py

4. to show various forms of the result

Hyeonjin Kim
2019.08.29
"""
import collections
import common
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import MinMaxScaler

FULLSET_PATH = f'{common.BASE_DIR}/fullset.csv'
TRAINSET_PATH = f'{common.BASE_DIR}/trainset.csv'
MODEL_PATH = f'{common.BASE_DIR}/model/model.h5'

UNPRED_OUTPUT_PATH = f'{common.BASE_DIR}/unpredicted_map.csv'
PRED_OUTPUT_PATH = f'{common.BASE_DIR}/predicted_map.csv'
PROBS_OUTPUT_PATH = f'{common.BASE_DIR}/full_probs.csv'

N_MODEL = 3

def get_codes(fullset):
    results = []
    for i, row in enumerate(fullset):
        disaster_prob_vec = row[:common.N_DISASTER]
        feature_vec = row[common.N_DISASTER:]

        categorized_vec = common.categorize(disaster_prob_vec, 0.4, 0.6)
        if common.is_certain(categorized_vec):
            label = common.classify(categorized_vec)
            results.append(label)
        elif -99 in feature_vec:
            results.append(-99)
        else:
            results.append(common.N_CLASS)

    return np.array(results)

def remove_leftmost_nodata(vec):
    idx = vec.tolist().index(common.Category.UNCERTAIN)
    zerovec = vec.copy()
    onevec = vec.copy()
    zerovec[idx] = 0
    onevec[idx] = 1
    return zerovec, onevec

def span(categorized_vec):
    having_nodata_vecs = collections.deque([categorized_vec])
    possible_vecs = []
    while having_nodata_vecs:
        top_vec = having_nodata_vecs.pop()
        if not common.is_certain(top_vec):
            zerovec, onevec = remove_leftmost_nodata(top_vec)
            having_nodata_vecs.appendleft(zerovec)
            having_nodata_vecs.appendleft(onevec)
        else:
            possible_vecs.append(top_vec)
    return possible_vecs

def check_and_decide(disaster_prob_vec, probs):
    categorized_vec = common.categorize(disaster_prob_vec, 0.4, 0.6, nodata_to_uncertain=True)
    possible_vecs = span(np.array(categorized_vec))
    possible_labels = list(map(common.classify, possible_vecs))
    decrease_idx = np.argsort(-np.array(probs))
    for i, idx in enumerate(decrease_idx):
        if idx in possible_labels:
            return idx, i
    print('error')

def get_encoded_codes(fullset):
    result = []
    for row in fullset:
        disaster_prob_vec = row[:common.N_DISASTER]
        categorized_vec = common.categorize(disaster_prob_vec, 0.4, 0.6)
        result.append('"' + ''.join(map(str, categorized_vec)) + '"')
    return result

def get_hunnit_prob_vecs(labels):
    result = []
    for label in labels:
        row = [0.0] * common.N_CLASS
        row[label] = 1.0
        result.append(row)
    return np.array(result)

def main():
    fullset = common.load_data(FULLSET_PATH, sep=',')
    codes = get_codes(fullset)
    uncertain_mask = (codes == common.N_CLASS)

    uncertain_set = fullset[uncertain_mask]
    uncertain_features = common.onehot_encode(uncertain_set[:, common.N_DISASTER:], 0)

    trainset = common.load_data(TRAINSET_PATH, sep=',')
    trainset = common.onehot_encode(trainset, 0)

    prob_sum = np.zeros((uncertain_features.shape[0], common.N_CLASS))
    for i in range(N_MODEL):
        x_train, _, _, _ = common.split(trainset, i)
        _, normalized_features = common.normalize(x_train, uncertain_features)
        prob_sum += tf.keras.models.load_model(common.numbering(MODEL_PATH, i)).predict(normalized_features)
        print(i, ' is done.')
    probs = prob_sum / N_MODEL
    linenum_to_prob = { idx: prob for idx, prob in zip(np.nonzero(uncertain_mask)[0], probs) }

    # unpredicted map
    common.save_map(codes.reshape(common.N_ROWS, -1), UNPRED_OUTPUT_PATH)

    # predicted map
    counter = [0] * common.N_CLASS
    predicted_map = codes.copy()
    for i, (row, code) in enumerate(zip(fullset, codes)):
        if code == common.N_CLASS:
            predicted_map[i], order = check_and_decide(row[:common.N_DISASTER], linenum_to_prob[i])
            counter[order] += 1
    common.save_map(predicted_map.reshape(common.N_ROWS, -1), PRED_OUTPUT_PATH)
    print(counter)

    # full_probs
    encoded_codes = get_encoded_codes(fullset)
    certain_mask = (codes < common.N_CLASS) & (codes >= 0)

    certain_set = codes[certain_mask]
    cerntain_probs = get_hunnit_prob_vecs(certain_set)
    linenum_to_certain_prob = { idx: prob for idx, prob in zip(np.nonzero(certain_mask)[0], cerntain_probs) }

    full_probs = []
    for i, code in enumerate(encoded_codes):
        if i in linenum_to_prob:
            _, order = check_and_decide(fullset[i][:common.N_DISASTER], linenum_to_prob[i])
            full_probs.append([code] + linenum_to_prob[i].tolist() + [order + 1])
        elif i in linenum_to_certain_prob:
            full_probs.append([code] + linenum_to_certain_prob[i].tolist() + [0])
        else:
            full_probs.append([0])

    cur_id = 0
    reversed_full_probs = []
    for row in np.flipud(np.array(full_probs).reshape(common.N_ROWS, -1)).reshape(-1):
        if len(row) == 1:
            continue
        reversed_full_probs.append([cur_id] + row)
        cur_id += 1
    common.save_data(reversed_full_probs, PROBS_OUTPUT_PATH)

if __name__ == '__main__':
    main()
