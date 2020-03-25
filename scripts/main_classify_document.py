import os
import argparse

import numpy as np
import cv2

import util


def get_company_logo(img, logo_dir):

    # img = cv2.resize(img, (1650, 2330))  # resize image to standard format
    img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)  # reduce size to half, to speed computing

    # sift descriptors of input document
    sift = cv2.xfeatures2d.SIFT_create()
    kp2, des2 = sift.detectAndCompute(img, None)

    # auxiliary arrays
    lengths = []
    img_paths = []

    for img_path in os.listdir(logo_dir):

        TEMPLATE_PATH = os.path.join(os.path.join(logo_dir, img_path))  # path to image logo
        img1 = cv2.imread(TEMPLATE_PATH, 0)  # image logo

        img1 = cv2.resize(img1, (0, 0), fx=0.5, fy=0.5)  # reduce logo size to half, to speed computing

        # find the keypoints and descriptors with SIFT
        kp1, des1 = sift.detectAndCompute(img1, None)

        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)

        flann = cv2.FlannBasedMatcher(index_params, search_params)

        matches = flann.knnMatch(des1, des2, k=2)

        # store all the good matches as per Lowe's ratio test.
        good = []
        for m, n in matches:
            if m.distance < 0.4 * n.distance:
                good.append(m)

        # update arrays
        img_paths.append(img_path)
        lengths.append(len(good))

    j = np.argsort(lengths)[-1]  # index to best match logo

    if lengths[j] > 5:
        return img_paths[j].split('_')[0]
    else:
        return 'unknown'


parser = argparse.ArgumentParser()
parser.add_argument('--doc_path', nargs='?')
parser.add_argument('--logo_dir', nargs='?')

args = parser.parse_args()

DOCUMENT_PATH = args.doc_path  # document path
img = cv2.imread(DOCUMENT_PATH, 0)  # document image

logo = get_company_logo(img)  # get matching logo
logo_dir = args.doc_path  # logos directory

print('Company logo:', logo)
