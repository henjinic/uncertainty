# Disaster Type Classifications

2019.07.29 ~

### `0. common.py`
### `1. fullset.py`
   * *in*
     * `./maps/fire.txt`
     * `...`
     * `./maps/tavg.txt`
   * *out*
     * `./fullset.csv`
### `2. trainset.py`
   * *in*
     * `./fullset.csv`
   * *out*
     * `./trainset.csv`
### `3. train.py`
   * *in*
     * `./trainset.csv`
   * *out*
     * `./model/model_{0..i}.h5`
     * `./history/history_{0..i}.h5`
### `4. result.py`
   * *in*
     * `./fullset.csv`
     * `./trainset.csv`
     * `./model/model_{0..i}.h5`
   * *out*
     * `./unpredicted_map.csv`
     * `./predicted_map.csv`
     * `./full_probs.csv`
