import os
import argparse

import cv2

from util_feature_extractor import sift_features, surf_features, orb_features, brief_features, load_descriptors, \
                                    calculate_matches
from util import preprocess_document


def get_company_logo(doc, logos_features):
    results = {}

    doc_features = {}
    for f in [sift_features, surf_features, orb_features, brief_features]:
        doc_keypoints, doc_descriptors = f(doc)
        doc_features[f.__name__] = {'keypoints': doc_keypoints,
                                    'descriptors': doc_descriptors}

    img_paths, values = calculate_matches(doc_features, logos_features)

    for j, img_path in enumerate(img_paths):
        company_name = img_path[:-8]
        if company_name in results.keys():
            results[company_name] = results[company_name] + values[j]
        else:
            results[company_name] = values[j]

    sorted_list = list(sorted(((v, k) for k, v in results.items())))

    if sorted_list[-1][0] > 40:
        return sorted_list[-1][1]
    else:
        return 'unknown'


DATA_DIR = os.path.join('..', 'data')
FEATURES_DIR = os.path.join('..', 'features')

parser = argparse.ArgumentParser()
parser.add_argument('--doc_path', nargs='?', default='../data/document_test/anaf.jpg')

args = parser.parse_args()

doc_path = args.doc_path  # document path

logos_features_path = os.path.join(FEATURES_DIR, 'logos_descriptors.joblib')
logos_features = load_descriptors(logos_features_path)

doc = cv2.imread(doc_path)

# doc = preprocess_document(doc)

logo = get_company_logo(doc, logos_features)

print('Company logo:', logo)
