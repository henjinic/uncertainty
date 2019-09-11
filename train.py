"""train.py

3. to train feed forward network

Hyeonjin Kim
2019.08.27
"""
import common
import json
import numpy as np
import tensorflow as tf

TRAINSET_PATH = f'{common.BASE_DIR}/trainset.csv'
MODEL_PATH = f'{common.BASE_DIR}/model/model.h5'
HISTORY_PATH = f'{common.BASE_DIR}/history/history.json'

N_MODEL = 10
N_EPOCH = 100

def create_model(input_dim):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(64, kernel_regularizer=tf.keras.regularizers.l2(0.001), input_shape=(input_dim,), activation='relu'),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(128, kernel_regularizer=tf.keras.regularizers.l2(0.001), activation='relu'),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(64, kernel_regularizer=tf.keras.regularizers.l2(0.001), activation='relu'),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(4, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def save_history(hist, path):
    decoded_hist = { k: list(map(float, hist.history[k])) for k in hist.history }
    with open(path, 'w') as f:
        json.dump(decoded_hist, f)

def train(x, y, epoch):
    model = create_model(x.shape[1])
    history = model.fit(x, y, verbose=1, validation_split=0.2, epochs=epoch)
    return model, history

def main():
    trainset = common.load_data(TRAINSET_PATH, sep=',')
    trainset = common.onehot_encode(trainset, 0)

    for i in range(N_MODEL):
        x_train, x_test, y_train, y_test = common.split(trainset, i)
        x_train, x_test = common.normalize(x_train, x_test)

        model, history = train(x_train, y_train, N_EPOCH)
        model.evaluate(x_test, y_test)

        model.save(common.numbering(MODEL_PATH, i))
        save_history(history, common.numbering(HISTORY_PATH, i))

        print(i, ' is done.')

if __name__ == '__main__':
    main()
