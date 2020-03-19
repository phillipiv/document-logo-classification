import os

import numpy as np
import cv2
from matplotlib import pyplot as plt

import util

for doc_path in os.listdir(util.DOCUMENTS_DIR):

    company = doc_path.split('.')[0]

    TEST_DOCUMENT_PATH = os.path.join(util.DOCUMENTS_DIR, doc_path)
    img2 = cv2.imread(TEST_DOCUMENT_PATH, 0)

    sift = cv2.xfeatures2d.SIFT_create()
    kp2, des2 = sift.detectAndCompute(img2, None)

    lengths = []
    imgs = []
    img_paths = []
    goods = []
    kps = []
    for img_path in os.listdir(util.LOGOS_DIR):

        TEMPLATE_PATH = os.path.join(os.path.join(util.LOGOS_DIR, img_path))
        img1 = cv2.imread(TEMPLATE_PATH, 0)
        imgs.append(img1)

        # MIN_MATCH_COUNT = 1000

        # img1 = cv2.imread('box.png', 0)          # queryImage
        # img2 = cv2.imread('box_in_scene.png', 0) # trainImage

        # Initiate SIFT detector
        # sift = cv2.SIFT()

        # find the keypoints and descriptors with SIFT
        kp1, des1 = sift.detectAndCompute(img1, None)

        kps.append([kp1, kp2])

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

        img_paths.append(img_path)
        lengths.append(len(good))
        goods.append(good)

    for j in np.argsort(lengths)[-5:]:
        print(company, img_paths[j], lengths[j])

    j = np.argsort(lengths)[-1]
    if lengths[j] > 0:
        try:
            kp1, kp2 = kps[j]
            src_pts = np.float32([kp1[m.queryIdx].pt for m in goods[j]]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in goods[j]]).reshape(-1, 1, 2)

            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            matchesMask = mask.ravel().tolist()

            h, w = imgs[j].shape
            pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, M)

            img2 = cv2.polylines(img2, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

            draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                               singlePointColor=None,
                               matchesMask=matchesMask,  # draw only inliers
                               flags=2)

            img3 = cv2.drawMatches(imgs[j], kp1, img2, kp2, goods[j], None, **draw_params)

            plt.figure()
            plt.imshow(img3)
            plt.savefig(os.path.join('.', doc_path.replace('jpg', 'png')))
            # plt.show()

        except Exception:
            pass


    print('')
