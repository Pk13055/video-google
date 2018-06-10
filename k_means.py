#!/usr/bin/env python3
"""
    script to call k_means with the available
    resources. JUST to daemonize
"""
import os
from sys import argv as rd
import numpy as np
from sklearn.cluster import k_means
import pickle


def main():
    dataset = pickle.load(open('descriptors.pkl', 'rb'))
    print(dataset.shape)
    knn = k_means(X=dataset, n_clusters=1000, n_jobs=-2, n_init=100,
                  max_iter=1e4)
    pickle.dump(knn, open('clusters.pkl', 'wb'))


if __name__ == "__main__":
    main()

