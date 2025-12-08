
import glob
import os
from natsort import natsorted

import cv2
import copy
import numpy as np
import skimage
from skimage.feature import peak_local_max
from scipy.spatial import distance as dist
from PIL import Image
import pandas as pd


def get_series(path_images, leaf_uid=None, verbose=False):
    """
    Creates two lists of file paths: to key point coordinate files and to images
    for each of the samples monitored over time, stored in date-wise folders.
    :return:
    """
    if verbose:
        print("Getting Series...")
    id_series = []

    if verbose:
        print("Getting paths...")
    images = glob.glob(f'{path_images}/*.JPG')
    if verbose:
        print("Extracting IDs...")
    image_image_id = ["_".join(os.path.basename(l).split("_")[2:4]).replace(".JPG", "") for l in images]
    if verbose:
        print("Removing duplicate IDs...")
    uniques = natsorted(np.unique(image_image_id))

    if verbose:
        print("Filtering by leaf_uid...")
    # filter by leaf_uid
    if leaf_uid: 
        uniques = [u for u in uniques if str(u) == str(leaf_uid)]

    if verbose:
        print("Compiling list...")
    # compile the lists
    for unique_sample in uniques:
        image_idx = [index for index, image_id in enumerate(image_image_id) if unique_sample == image_id]
        sample_image_names = [images[i] for i in image_idx]
        # sort to ensure sequential processing of subsequent images
        sample_image_names = sorted(sample_image_names, key=lambda i: os.path.splitext(os.path.basename(i))[0])
        id_series.append(sample_image_names)

    return id_series


def remove_double_detections(x, y, tol):
    """
    Removes one of two coordinate pairs if their distance is below 50
    :param x: x-coordinates of points
    :param y: y-coordinates of points
    :param tol: minimum distance required for both points to be retained
    :return: the filtered list of points and their x and y coordinates
    """
    point_list = np.array([[a, b] for a, b in zip(x, y)], dtype=np.int32)
    dist_mat = dist.cdist(point_list, point_list, "euclidean")
    np.fill_diagonal(dist_mat, np.nan)
    dbl_idx = np.where(dist_mat < tol)[0].tolist()[::2]
    point_list = np.delete(point_list, dbl_idx, axis=0)
    x = np.delete(x, dbl_idx, axis=0)
    y = np.delete(y, dbl_idx, axis=0)
    return point_list, x, y


def reject_outliers(data, tol=None, m=2.):
    """
    Detects outliers in 1d and returns the list index of the outliers
    :param data: 1d array
    :param tol: a tolerance in absolute distance
    :param m: number of sd s to tolerate
    :return: list index of outliers
    """
    d = np.abs(data - np.mean(data))
    mdev = np.mean(d)
    s = d / mdev if mdev else np.zeros(len(d))
    idx = np.where(s > m)[0].tolist()  # no difference available for first point - no changes
    if tol is not None:
        abs_diff = np.abs(np.diff(data))
        abs_diff = np.append(abs_diff[0], abs_diff)
        idx = [i for i in idx if abs_diff[i] > tol]  # remove outliers within the absolute tolerance

    return idx


def get_keypoints(file_path, shape):

    coords = pd.read_table(file_path, sep=r"[,\s]+", engine="python")
    
    # if file is already in pixel coordinates
    if coords.shape[1] == 2 and list(coords.columns) == ["x", "y"]:
        # Already pixel coordinates → just convert to list of tuples
        return coords.to_numpy(dtype=float)

    coords = pd.read_table(file_path, header=None, sep=" ")
    
    # get key point coordinates from YOLO output
    x = coords.iloc[:, 5] * shape[1]
    y = coords.iloc[:, 6] * shape[0]

    # remove double detections
    # TODO can this be done during inference via non maximum suppression
    point_list, x, y = remove_double_detections(x=x, y=y, tol=50)

    # remove outliers in the key point detections from YOLO errors,
    outliers_x = reject_outliers(x, tol=None, m=3.)  # larger extension, larger variation
    outliers_y = reject_outliers(y, tol=None, m=2.5)  # smaller extension, smaller variation
    outliers = outliers_x + outliers_y
    point_list = np.delete(point_list, outliers, 0)
    keypoints = [tuple(pt) for pt in point_list]

    return keypoints

def remove_points_from_mask(mask, classes):
    """
    Removes predicted pycnidia and rust pustules from the mask. Replaces the relevant pixel values with the average
    of the surrounding pixels. Points need to be transformed separately and added again to the transformed mask.
    :param mask: the mask to remove the points from
    :param classes: ta list with indices of the classes that are represented as points
    :return: mask with key-points removed
    """

    mask = copy.copy(mask)
    for cl in classes:
        idx = np.where(mask == cl)
        y_points, x_points = idx
        for i in range(len(y_points)):
            row, col = y_points[i], x_points[i]
            surrounding_pixels = mask[max(0, row - 1):min(row + 2, mask.shape[0]),
                                 max(0, col - 1):min(col + 2, mask.shape[1])]
            average_value = np.mean(surrounding_pixels)
            mask[row, col] = average_value
    return mask


def rotate_translate_warp_points(mask, classes, rot, box, tf, target_shape, warped):
    """
    rotates, translates, and warps points to match the transformed segmentation mask.
    Filters detected point lying outside the roi.
    :param mask: The original full-sized segmentation mask that includes all classes
    :param classes: List of integers specifying the class of point labels
    :param rot: rotation matrix applied to the msak
    :param box: the corner coordinates of the bounding box used to crop he roi from the image
    :param tf: the transformation matrix
    :param target_shape: the dimension of the desired output image
    :param warped: the warped segmentation mask of the roi, without the points
    :return: The complemented warped roi
    """

    # get input shape
    w = box[1, 0] - box[0, 0]
    h = box[2, 1] - box[1, 1]
    input_shape = (h, w)

    # loop over classes that are represented as points
    for cl in classes:

        # get class pixel positions
        idx = np.where(mask == cl)

        # if there are any pixels to transform, do so, else leave unchanged
        if len(idx[0]) == 0:
            continue

        # extract points
        points = np.array([[a, b] for a, b in zip(idx[1], idx[0])], dtype=np.int32)

        # rotate points
        points_rot = np.intp(cv2.transform(np.array([points]), rot))[0]

        # translate points
        tx, ty = (-box[0][0], -box[0][1])
        translation_matrix = np.array([
            [1, 0, tx],
            [0, 1, ty]
        ], dtype=np.float32)
        points_trans = np.intp(cv2.transform(np.array([points_rot]), translation_matrix))[0]

        # remove any rotated and translated point outside the roi
        mask_pw = (points_trans[:, 1] < input_shape[0]) & (points_trans[:, 1] > 0) & \
                  (points_trans[:, 0] < input_shape[1]) & (points_trans[:, 0] > 0)
        points_filtered = points_trans[mask_pw]

        # create and warp the point mask
        point_mask = np.zeros(input_shape).astype("uint8")
        point_mask[points_filtered[:, 1], points_filtered[:, 0]] = 255
        lm = np.stack([point_mask, point_mask, point_mask], axis=2)
        warped_pycn_mask = skimage.transform.warp(lm, tf, output_shape=target_shape)
        coordinates = peak_local_max(warped_pycn_mask[:, :, 0], min_distance=1)
        warped[coordinates[:, 0], coordinates[:, 1]] = cl

    return warped


def triangle_aspect_ratio(triangle):
    # Side lengths
    a = np.linalg.norm(triangle[0] - triangle[1])
    b = np.linalg.norm(triangle[1] - triangle[2])
    c = np.linalg.norm(triangle[2] - triangle[0])
    s = 0.5 * (a + b + c)  # semi-perimeter

    # Heron's formula for area
    area = max(np.sqrt(max(s * (s - a) * (s - b) * (s - c), 0)), 1e-8)

    # Height from longest side (h = 2A / base)
    longest = max(a, b, c)
    height = 2 * area / longest

    aspect_ratio = longest / height
    return aspect_ratio
