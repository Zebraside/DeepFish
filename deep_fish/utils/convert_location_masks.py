import os

import numpy as np
import pandas as pd
import cv2
import tqdm

def __get_default_detector():
    params = cv2.SimpleBlobDetector_Params()

    params.minThreshold = 10
    params.maxThreshold = 256

    params.filterByColor = False
    params.blobColor = 255

    params.filterByArea = True
    params.minArea = 35
    params.maxArea = 1000

    params.filterByCircularity = False
    params.filterByConvexity = False
    params.filterByInertia = False

    return cv2.SimpleBlobDetector_create(params)


def count_objects(mask, is_black_background=True, detector=__get_default_detector()):
    return len(detector.detect(np.bitwise_not(mask) if is_black_background else mask))

def convert_location_masks(data_csv, src_dir, dst_dir, new_radius=5, is_black_background=True):
    mask_names = [name + ".png" for name in pd.read_csv(data_csv)["ID"].tolist()]

    if not os.path.exists(dst_dir):
        os.mkdir(dst_dir)

    os.makedirs(os.path.join(dst_dir, "empty"), exist_ok=True)
    os.makedirs(os.path.join(dst_dir, "valid"), exist_ok=True)

    detector = __get_default_detector()
    for name in tqdm.tqdm(mask_names):
        mask = cv2.imread(os.path.join(src_dir, name), cv2.IMREAD_GRAYSCALE)
        points = [[int(i) for i in keypoint.pt] for keypoint in detector.detect(np.bitwise_not(mask) if is_black_background else mask)]

        for point in points:
            mask = cv2.circle(mask, point, new_radius, color=255, thickness=-1)

        cv2.imwrite(os.path.join(dst_dir, name), mask)


if __name__ == "__main__":
    BASE_DIR = "/work/data/DeepFish/Localization/"
    convert_location_masks(BASE_DIR + "train.csv",
                           BASE_DIR + "masks",
                           BASE_DIR + "12_masks",
                           12)
    
    convert_location_masks(BASE_DIR + "test.csv",
                        BASE_DIR + "masks",
                        BASE_DIR + "12_masks",
                        12)

    convert_location_masks(BASE_DIR + "val.csv",
                    BASE_DIR + "masks",
                    BASE_DIR + "12_masks",
                    12)