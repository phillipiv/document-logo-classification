import os
import argparse
import joblib

import cv2

from util_feature_extractor import load_descriptors, sift_features, surf_features, orb_features, brief_features


DATA_DIR = os.path.join('..', 'data')
FEATURES_DIR = os.path.join('..', 'features')

parser = argparse.ArgumentParser()
parser.add_argument('--logo_path', nargs='?', default='../data/logos_v2/abc_000.png')
parser.add_argument('--logos_dir', nargs='?', default=None)
parser.add_argument('--descriptors', nargs='?', default='../features/logos_descriptors.joblib')

args = parser.parse_args()

logo_path = args.logo_path
logos_dir = args.logos_dir
logos_features_path = args.descriptors

logos_features = joblib.load(logos_features_path)

if logos_dir is not None:
    for logo_path_ in os.listdir(logos_dir):
        logo_path = os.path.join(logos_dir, logo_path_)
        temp_descriptors = {}
        logo = cv2.imread(logo_path)
        for f in [sift_features, surf_features, orb_features, brief_features]:
            keypoints, descriptors = f(logo)
            keypoints_list = []
            for point in keypoints:
                temp = [point.pt, point.size, point.angle, point.response, point.octave, point.class_id]
                keypoints_list.append(temp)

            temp_descriptors[f.__name__] = {'keypoints': keypoints_list,
                                            'descriptors': descriptors}

        logos_features[logo_path] = temp_descriptors

joblib.dump(logos_features, logos_features_path)
