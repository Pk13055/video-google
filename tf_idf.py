#!/usr/bin/env python3
"""
    Get the auxillary information requried to
    build up the tf-IDF table
"""
import os
from sys import argv as rd
import pickle
import numpy as np


def main():
    file_data = pickle.load(open('file_data.pkl', 'rb'))
    N = len(file_data) + 1 # missing file 28.png
    n_clusters = 1000
    tf_idf = np.zeros((N + 1, n_clusters)) # zero indexed
    doc_count = np.zeros((n_clusters, 1))
    for doc_no, clus_dist in file_data:
        current_idxs = np.unique(clus_dist[:, -1]).astype(int)
        doc_count[current_idxs] += 1
        cur_counts = [np.count_nonzero(clus_dist[:, -1] == _) for _ in
                      current_idxs]
        tf_idf[doc_no][current_idxs] = cur_counts
    tf_idf = tf_idf[1:, :]
    doc_count = np.log(N / doc_count)
    tf_idf = tf_idf * doc_count.T
    pickle.dump(tf_idf, open('tf_idf-table.pkl', 'wb'))


if __name__ == "__main__":
    main()
