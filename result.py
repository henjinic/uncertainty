"""result.py

4. to create result table

Hyeonjin Kim
2019.09.20
"""
import collections
import common
import tensorflow as tf
import numpy as np

FULLSET_PATH = f'{common.BASE_DIR}/fullset.csv'
TRAINSET_PATH = f'{common.BASE_DIR}/trainset.csv'
MODEL_PATH = f'{common.BASE_DIR}/model/model.h5'

OUTPUT_PATH = f'{common.BASE_DIR}/result.csv'

N_MODEL = 10

UNCERTAIN_LABEL = common.N_CLASS

def get_types(fullset):
    results = []
    for row in fullset:
        disaster_prob_vec = row[:common.N_DISASTER]
        feature_vec = row[common.N_DISASTER:]

        categorized_vec = common.categorize(disaster_prob_vec, 0.4, 0.6)
        if common.is_certain(categorized_vec):
            label = common.classify(categorized_vec)
            results.append(label)
        elif -99 in feature_vec:
            results.append(-99)
        else:
            results.append(UNCERTAIN_LABEL)

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
    return decrease_idx[0], 0

def get_probs_for_uncertain(uncertainset):
    trainset = common.load_data(TRAINSET_PATH, sep=',')

    encoded_uncertainset = common.onehot_encode(uncertainset[:, common.N_DISASTER:], 0)
    encoded_trainset = common.onehot_encode(trainset, 0)

    prob_sums = np.zeros((len(uncertainset), common.N_CLASS))
    for i in range(N_MODEL):
        x_train, _, _, _ = common.split(encoded_trainset, i)
        _, normalized_uncertainset = common.normalize(x_train, encoded_uncertainset)
        prob_sums += tf.keras.models.load_model(common.numbering(MODEL_PATH, i)).predict(normalized_uncertainset)
        print(f'{i} is done.')

    return prob_sums / N_MODEL

def main():
    print("Loading...")
    fullset = common.load_data(FULLSET_PATH, sep=',')

    types = get_types(fullset)

    print("Predicting...")
    uncertain_mask = (types == UNCERTAIN_LABEL)
    uncertainset = fullset[uncertain_mask]
    probs = get_probs_for_uncertain(uncertainset)
    linenum_to_probs = { idx: prob for idx, prob in zip(np.nonzero(uncertain_mask)[0], probs) }

    print("Deciding...")
    probs_and_predictions = []
    for i, (row, type_) in enumerate(zip(fullset, types)):
        if type_ == UNCERTAIN_LABEL:
            probs = linenum_to_probs[i].tolist()
            prediction, order = check_and_decide(row[:common.N_DISASTER], probs)
            probs_and_predictions.append(probs + [prediction] + [order + 1])
        elif type_ == -99:
            probs_and_predictions.append([-99] * (common.N_CLASS + 2))
        else:
            probs = [0.0] * common.N_CLASS
            probs[type_] = 1.0
            probs_and_predictions.append(probs + [type_] + [0])

    print("Saving...")
    common.save_data(np.concatenate((types[:, np.newaxis], probs_and_predictions), axis=1), OUTPUT_PATH)

if __name__ == '__main__':
    main()
