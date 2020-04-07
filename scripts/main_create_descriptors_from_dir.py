import os
import joblib
import argparse

import cv2

from util_feature_extractor import sift_features, surf_features, orb_features, brief_features

parser = argparse.ArgumentParser()
parser.add_argument('--logos_dir', nargs='?', default='../data/logos_v2')

args = parser.parse_args()

DATA_DIR = os.path.join('..', 'data')
FEATURES_DIR = os.path.join('..', 'features')

logos_dir = args.logos_dir

logos_descriptors = {}
for logo_path in os.listdir(logos_dir):
    temp_descriptors = {}
    logo = cv2.imread(os.path.join(logos_dir, logo_path))
    for f in [sift_features, surf_features, orb_features, brief_features]:
        keypoints, descriptors = f(logo)
        keypoints_list = []
        for point in keypoints:
            temp = [point.pt, point.size, point.angle, point.response, point.octave, point.class_id]
            keypoints_list.append(temp)

        temp_descriptors[f.__name__] = {'keypoints': keypoints_list,
                                        'descriptors': descriptors}

    logos_descriptors[logo_path] = temp_descriptors

filename = os.path.join(FEATURES_DIR, 'logos_descriptors.joblib')
joblib.dump(logos_descriptors, filename)
