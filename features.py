#!/usr/bin/env python3
"""
    Script to extract sift features from each frame
    and prepare a common store
"""
import pickle
import os
from sys import argv as rd
from multiprocessing import Pool
import numpy as np
import pandas as pd
import cv2

DATA_DIR = os.path.join(os.getcwd(), 'dataset')
sift = cv2.xfeatures2d.SIFT_create()

def getSIFT(img_name):
    '''
        @description return the SIFT features
        along with the img_name
        @param img_name -> str: img name
    '''
    fullpath = lambda x: os.path.join(DATA_DIR, 'frames', x)
    kps, des = sift.detectAndCompute(cv2.imread(
        fullpath(img_name), cv2.IMREAD_GRAYSCALE), None)
    return (img_name, des)


def main():
    files = os.listdir(os.path.join(DATA_DIR, 'frames'))
    features = Pool(8).map_async(getSIFT, files).get()
    pickle.dump(features, open('features.pkl', 'wb'))
    print([_[-1].shape if _[-1] is not None else _[0] for _ in features])


if __name__ == "__main__":
    main()
