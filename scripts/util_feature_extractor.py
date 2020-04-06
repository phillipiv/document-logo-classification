import joblib

import cv2
import numpy as np


def preproc_image(img):

    # Convert the training image to RGB
    training_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Convert the training image to gray scale
    training_gray = cv2.cvtColor(training_image, cv2.COLOR_RGB2GRAY)

    return training_gray


def sift_features(img):

    sift = cv2.xfeatures2d.SIFT_create()

    train_keypoints, train_descriptor = sift.detectAndCompute(img, None)

    return train_keypoints, train_descriptor


def surf_features(img):

    surf = cv2.xfeatures2d.SURF_create(800)

    train_keypoints, train_descriptor = surf.detectAndCompute(img, None)

    return train_keypoints, train_descriptor


def brief_features(img):

    fast = cv2.FastFeatureDetector_create()
    brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()

    train_keypoints = fast.detect(img, None)

    train_keypoints, train_descriptor = brief.compute(img, train_keypoints)

    return train_keypoints, train_descriptor


def orb_features(img):
    orb = cv2.ORB_create()

    train_keypoints, train_descriptor = orb.detectAndCompute(img, None)

    return train_keypoints, train_descriptor


def load_descriptors(path):
    descriptors = joblib.load(path)

    for img_path in descriptors.keys():

        for f_name in descriptors[img_path].keys():
            keypoints_list = []
            for point in descriptors[img_path][f_name]['keypoints']:

                temp_feature = cv2.KeyPoint(x=point[0][0], y=point[0][1], _size=point[1], _angle=point[2],
                                            _response=point[3], _octave=point[4], _class_id=point[5])

                keypoints_list.append(temp_feature)

            descriptors[img_path][f_name]['keypoints'] = keypoints_list

    return descriptors


def get_params(f_name):

    if f_name in ['sift_features', 'surf_features']:
        crossCheck_ = False
        norm_ = cv2.NORM_L1
    else:
        crossCheck_ = True
        norm_ = cv2.NORM_HAMMING

    return crossCheck_, norm_


def calculate_matches(doc_features, logos_features):

    img_paths = []
    values = []

    for img_path in logos_features.keys():

        goods = []
        # for f_name in logos_features[img_path].keys():
        for f_name in ['sift_features', 'surf_features']:

            crossCheck_, norm_ = get_params(f_name)

            bf = cv2.BFMatcher(norm_, crossCheck=crossCheck_)

            # Perform the matching between the ORB descriptors of the training image and the test image
            matches = bf.knnMatch(logos_features[img_path][f_name]['descriptors'],
                               doc_features[f_name]['descriptors'], k=2)

            # The matches with shorter distance are the ones we want.
            # matches = sorted(matches, key=lambda x: x.distance)

            good = []
            for m, n in matches:
                if m.distance < 0.4 * n.distance:
                    good.append(m)

            goods.append(len(good))

        img_paths.append(img_path)
        values.append(sum(goods))
        # print(img_path, matches_, np.round(matches_mean, 2), np.round(matches_inv, 2), np.mean(matches_mean))
        # print(img_path, goods, sum(goods))

    return img_paths, values