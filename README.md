# Disaster Type Classifications

2019.07.29 ~

### [`0. common.py`](/common.py)
### [`1. fullset.py`](/fullset.py)
   * *in*
     * `./maps/fire.txt`
     * `...`
     * `./maps/tavg.txt`
   * *out*
     * `./fullset.csv`
### [`1-2. clust.py`](/clust.py)
   * *in*
     * `./fullset.csv`
   * *out*
     * `./clust_summary.csv`
     * `./label_to_codes.json`
### [`2. trainset.py`](/trainset.py)
   * *in*
     * `./fullset.csv`
   * *out*
     * `./trainset.csv`
### [`3. train.py`](/train.py)
   * *in*
     * `./trainset.csv`
   * *out*
     * `./model/model_{0..i}.h5`
     * `./history/history_{0..i}.h5`
### [`4. result.py`](/result.py)
   * *in*
     * `./fullset.csv`
     * `./trainset.csv`
     * `./model/model_{0..i}.h5`
   * *out*
     * `./unpredicted_map.csv`
     * `./predicted_map.csv`
     * `./full_probs.csv`
