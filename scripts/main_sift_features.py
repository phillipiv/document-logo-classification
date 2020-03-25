import os
import time

import numpy as np
import cv2
from matplotlib import pyplot as plt

import util


print_times = False

# doc_dir = util.DOCUMENTS_DIR
doc_dir = '../data/document_other'
for doc_path in os.listdir(doc_dir):

    company = doc_path.split('.')[0]

    TEST_DOCUMENT_PATH = os.path.join(doc_dir, doc_path)
    img2 = cv2.imread(TEST_DOCUMENT_PATH, 0)

    img2 = cv2.resize(img2, (0, 0), fx=0.5, fy=0.5)

    sift = cv2.xfeatures2d.SIFT_create()
    kp2, des2 = sift.detectAndCompute(img2, None)

    lengths = []
    imgs = []
    img_paths = []
    goods = []
    kps = []

    tic_ = time.time()
    # logo_dir = util.LOGOS_V2_DIR
    logo_dir = os.path.join('..', 'data', 'logos_v3')
    for img_path in os.listdir(logo_dir):
        if print_times:
            print('')
            print(img_path)

        tic = time.time()
        TEMPLATE_PATH = os.path.join(os.path.join(logo_dir, img_path))
        img1 = cv2.imread(TEMPLATE_PATH, 0)

        if print_times:
            print(time.time() - tic)

        img1 = cv2.resize(img1, (0, 0), fx=0.5, fy=0.5)

        imgs.append(img1)

        # MIN_MATCH_COUNT = 1000

        # img1 = cv2.imread('box.png', 0)          # queryImage
        # img2 = cv2.imread('box_in_scene.png', 0) # trainImage

        # Initiate SIFT detector
        # sift = cv2.SIFT()

        tic = time.time()
        # find the keypoints and descriptors with SIFT
        kp1, des1 = sift.detectAndCompute(img1, None)
        if print_times:
            print(time.time() - tic)

        kps.append([kp1, kp2])

        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)

        tic = time.time()
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        if print_times:
            print(time.time() - tic)

        tic = time.time()
        matches = flann.knnMatch(des1, des2, k=2)
        if print_times:
            print(time.time() - tic)

        tic = time.time()
        # store all the good matches as per Lowe's ratio test.
        good = []
        for m, n in matches:
            if m.distance < 0.4 * n.distance:
                good.append(m)
        if print_times:
            print(time.time() - tic)

        img_paths.append(img_path)
        lengths.append(len(good))
        goods.append(good)

    print(time.time() - tic_)
    for j in np.argsort(lengths)[-5:]:
        print(company, img_paths[j], lengths[j])

    j = np.argsort(lengths)[-1]

    print('')
