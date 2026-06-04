import os
import cv2
import kornia as K
import kornia.feature as KF
import numpy as np
import torch
import skimage as ski
import math
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from torch_tps import ThinPlateSpline
from typing import List, Tuple

from leafalignment.utils import convert_image_to_tensor, group_by_argmax, affine, affine_warp_expand, check_orientation, RandomHomography
from leafalignment.plotting import plot_matches_conf, plot_match_coverage


CONSISTENCY_DEFAULT = None # {'consistency_tolerance': 10, 'transform': {'type': 'rotation', 'params': {'rotation': -10}}}
FILTERING_DEFAULT = {'filtering_strategy': 'confidence', 'n_landmarks': 500, 'n_landmarks_tol': 50, 'min_conf': 0.5}


def loftr_match(img_fix: torch.Tensor, img_mov: torch.Tensor, mask_fix: torch.Tensor=None, mask_mov: torch.Tensor=None, verbose: bool=True, return_n_matches: bool=False):
    """
    Detects Feature matches between fixed and moving images using LoFTR

    Args:
        img_fix: fixed image
        img_mov: moving image
        mask_fix: Optional mask of fixed image
        mask_mov: Optional mask of moving image
        verbose: Whether to produce detailed output (for diagnostic purposes)
        return_n_matches: Whether to return numbers of total matches, confident matches, inliers

    Returns:
        Keypoints in fixed image
        Keypoints in moving image
        Confidence of matches
        Classification of inliers, using RANSAC/Fundamental matrix
        (Dictionary of numbers of total matches, confident matches, inliers)
    """

    # match with LoFTR
    matcher = KF.LoFTR(pretrained="outdoor")

    # Add batch dim if needed
    if img_fix.dim() == 3:
        img_fix = img_fix.unsqueeze(0)
    if img_mov.dim() == 3:
        img_mov = img_mov.unsqueeze(0)
    if (mask_fix is not None) and (mask_mov is not None):
        if mask_fix.dim() == 2:
            mask_fix = mask_fix.unsqueeze(0)
        if mask_mov.dim() == 2:
            mask_mov = mask_mov.unsqueeze(0)
        if mask_fix.dim() == 4: # masks should have not batch dimension
            mask_fix = mask_fix.squeeze(0)
        if mask_mov.dim() == 4:
            mask_mov = mask_mov.squeeze(0)

        input_dict = {
            "image0": K.color.rgb_to_grayscale(img_fix),  # LofTR works on grayscale images only
            "image1": K.color.rgb_to_grayscale(img_mov),
            "mask0": mask_fix,
            "mask1": mask_mov,
        }
    else:
        input_dict = {
            "image0": K.color.rgb_to_grayscale(img_fix),  # LofTR works on grayscale images only
            "image1": K.color.rgb_to_grayscale(img_mov),
        }

    # detect matches
    with torch.inference_mode():
        correspondences = matcher(input_dict)    

    # select inliers
    mkpts0 = correspondences["keypoints0"]
    mkpts1 = correspondences["keypoints1"]
    confidence = correspondences["confidence"]

    if mkpts0.shape[0] < 8 or mkpts1.shape[0] < 8:
        print("Not enough points to perform inlier detection.")
        n_inliers = None
        inliers = None
    else:
        _, inliers = cv2.findFundamentalMat(mkpts0.detach().cpu().numpy().copy(), mkpts1.detach().cpu().numpy().copy(), cv2.USAC_MAGSAC, 1.0, 0.995, 10000)
        if inliers is None:
            n_inliers = None
        else:
            inliers = inliers > 0
            n_inliers = inliers.sum()

    if verbose:
        print(f"Total matches: {len(mkpts0)}")
        print(f"Matches with Confidence > 0.5: {torch.sum(confidence > 0.5)}")
        print(f"Inliers: {n_inliers} ({n_inliers/(len(mkpts0)+1e-13):.2%})")

    if return_n_matches:
        n_matches = {'total_matches': len(mkpts0), 'conf_matches': torch.sum(confidence > 0.5), 'inliers': n_inliers }
        return mkpts0, mkpts1, confidence, inliers, n_matches
    else:
        return mkpts0, mkpts1, confidence, inliers


# ----- Torch-based TPS Implementation -----

def fit_tps_torch(target_keypts: torch.Tensor, moving_keypts: torch.Tensor, alpha: float=0.0):
    """
    Fits the TPS transform between two sets of keypoints with format (x,y).

    Args:
        target_keypts: keypoints in target image
        moving_keypts: keypoints in moving image
        alpha: smoothing/regularization parameters

    Returs:
        ThinPlateSpline
    """

    target_keypts = target_keypts[..., [1, 0]] # meshgrid builder expects (y,x) format
    moving_keypts = moving_keypts[..., [1, 0]]

    # Fit the thin plate spline from output to input
    tps = ThinPlateSpline(alpha)
    tps.fit(target_keypts, moving_keypts)

    return tps

def warp_tps_points_torch(tps: ThinPlateSpline, points: torch.Tensor):
    """Applies TPS transform to points"""
    points = points[..., [1, 0]] # tps was trained in (y,x) format
    return tps.transform(points)[..., [1, 0]] # return output back in (x,y) format

def warp_tps_torch(tps_list: List[ThinPlateSpline], image: torch.Tensor, interpolation_mode: str='bilinear'):
    """
    Apply a series of TPS transformations to an image, using the specified interpolation mode.

    Args:
        tps_list: list of TPS transformations to be applied
        image: image to transform
        interpolation_mode: what interpolation mode to use when warping the image. For masks use 'nearest'

    Returns:
        torch.Tensor: warped image
    """
    image = convert_image_to_tensor(image)
    height = image.shape[2]
    width = image.shape[3]
    size = torch.tensor((height, width))

    # create pixel index vectors
    i = torch.arange(height, dtype=torch.float32)
    j = torch.arange(width, dtype=torch.float32)

    # create row/col coordinate matrices
    ii, jj = torch.meshgrid(i, j, indexing="ij")
    # combine into coordinate grid
    output_indices = torch.cat((ii[..., None], jj[..., None]), dim=-1)  # (H,W,2)
    # flatten grid -> tps expects list
    input_indices = output_indices.reshape(-1, 2) # (H*W, 2)


    if not isinstance(tps_list, list):
        tps_list = [tps_list]
    # apply tps transforms
    for tps in tps_list:
        if tps is not None:
            input_indices = tps.transform(input_indices)

    # reshape back into a grid
    input_indices = input_indices.reshape(height, width, 2)

    # normalize to [-1,1]
    grid = 2 * input_indices / size - 1
    grid = torch.flip(grid, (-1,)) # Grid sample works with x,y coordinates, not row, col

    return torch.nn.functional.grid_sample(image, grid[None], mode=interpolation_mode, align_corners=False)

def torch_tps(target_keypts: torch.Tensor, moving_keypts: torch.Tensor, moving_img: torch.Tensor, alpha: float=0.0, verbose: bool=False):
    """
    Fit a TPS transform between two sets of keypoints and use it to warp the moving image.

    Args:
        target_keypts: keypoints in target image
        moving_keypts: keypoints in moving image
        moving_img: moving image, to be transformed
        alpha: smoothing/regularization parameters
        verbose: Whether to produce detailed output (for diagnostic purposes)

    Returns:
        torch.Tensor: warped moving image
    """
    if verbose:
        print("Fitting TPS...")
    tps = fit_tps_torch(target_keypts, moving_keypts, alpha=alpha)
    if verbose:
        print("Warping Moving Image...")
    warped = warp_tps_torch(tps, moving_img)
    return warped


def register_loftr_tps(img_fixed: torch.Tensor, img_moving: torch.Tensor, threshold: float=0.5, smoothing: float=0.5, mask_moving: torch.Tensor=None, verbose: bool=False, plot_loftr_matches: bool=False, return_tps: bool=False):
    """
    Uses loftr to detect matches between the fixed and moving image, filters the matches by confidence, then uses TPS to transform the moving image
    If a mask of the moving image is provided, it is also warped.
    Optionally, the TPS transform can be returned.

    Note: This function is deprecated. Use `register_single_leaf` from `registration.py` instead.

    Args:
        img_fixed: fixed image
        img_moving: moving image
        threshold: minimum confidence threshold
        smoothing: smoothing hyperparameter. higher values lead to more "rigid" transforms
        mask_moving: Optional mask of moving image
        verbose: Whether to produce detailed output (for diagnostic purposes)
        plot_loftr_matches: whether to plot figures showing distribution of matches 
        return_tps: whether to return the TPS

    Returns:
        torch.Tensor: registered moving image
        (torch.Tensor: mask of registered moving image. only returned if mask of moving image is provided.)
        (ThinPlateSpline: TPS object used to register the image. only returned if return_tps==True.)
    """
    if img_fixed is None or img_moving is None:
        if return_tps:
            if mask_moving is not None:
                return None, None, None
            else:
                return None, None
        else:
            if mask_moving is not None:
                return None, None
            else:
                return None

    mkpts0, mkpts1, confidence, _, n_matches = loftr_match(img_fixed, img_moving, verbose=verbose, return_n_matches=True)

    if plot_loftr_matches:
        fig, ax = plot_matches_conf(img_fixed, mkpts0, img_moving, mkpts1, confidence, N_show=50, vertical=True)
        fig.show()
        fig, axs = plot_match_coverage(img_fixed, mkpts0, img_moving, mkpts1, confidence)
        fig.show()
    
    if n_matches['conf_matches'] > 3:
        kpts0, kpts1 = filter_matches_by_confidence(mkpts0, mkpts1, confidence, threshold, verbose=verbose)
        if verbose:
            print("Fitting TPS...")
        tps = fit_tps_torch(kpts0, kpts1, alpha=smoothing)
        if verbose:
            print("Warping Moving Image...")
        warped_moving_img = warp_tps_torch(tps, img_moving)
        if mask_moving is not None:
            if verbose:
                print("Warping Moving Mask...")
            warped_moving_mask = warp_tps_torch(tps, mask_moving, interpolation_mode='nearest')
    else:
        print("No enough matches for TPS found")
        warped_moving_img = None
        warped_moving_mask = None
        tps = None
    
    if return_tps:
        if mask_moving is not None:
            return warped_moving_img, warped_moving_mask, tps
        else:
            return warped_moving_img, tps
    else:
        if mask_moving is not None:
            return warped_moving_img, warped_moving_mask
        else:
            return warped_moving_img


# ----- Alternative TPS implementation using Skimage -----
# (Note that this implementation is significantly slower and doesn't expose a smoothing hyperparameter.
#   We hence moved away from this implementation, and it is thus not as polished as the up-to-date architecture)

def warp_tps_skimage(img, tps: ski.transform.ThinPlateSplineTransform, verbose: bool=False):
    """
    Applies TPS transform to warp image.

    Args:
        img: Image to warp. torch.Tensor or numpy image
        tps: Skimage TPS object
        verbose: Whether to produce detailed output (for diagnostic purposes)
    
    Returns:
        torch.Tensor: transformed image
    """
    # kornia and torch expect C x H x W, while skimage expects H x W x C
    if type(img) == torch.Tensor:
        img = K.tensor_to_image(img)

    if verbose:
        print("Transforming moving image...")
    warped = ski.transform.warp(img, tps) # warp uses inverse transform, i.e. img_mov -> img_fix

    return convert_image_to_tensor(warped)

def tps_skimage(keypts_fix: torch.Tensor, keypts_mov: torch.Tensor, img_mov=None, warp_moving: bool=True, verbose: bool=False):
    """
    Fits a TPS function to the keypoints. If a moving image is provided, applies the TPS transform, thus registering the moving image to the fixed image. 
    Expects keypoints to be filtered already. If too many keypoints are provided, memory limitations can occur.

    Args:
        keypts_fix: keypoints in fixed image
        keypts_mov: keypoitns in moving image
        img_mov: Optional moving image
        warp_moving: whether to transform the moving image
        verbose: Whether to produce detailed output (for diagnostic purposes)

    Returns: 
        torch.Tensor: transformed moving image
        ski.transform.ThinPlateSplineTransform: TPS transformation object
    """

    if verbose:
        print("Estimating TPS transform...")
    tps = ski.transform.ThinPlateSplineTransform.from_estimate(keypts_fix, keypts_mov)

    if warp_moving:
        warped = warp_tps_skimage(img_mov, tps, verbose=verbose)
        # warped = ski.transform.warp(img_mov_reordered, tps) # warp uses inverse transform, i.e. img_mov -> img_fix

        return warped, tps
    else:
        return None, tps

def tps_skimage_confidence(keypts_fix: torch.Tensor, keypts_mov: torch.Tensor, confidence: torch.Tensor, thrsld: float, img_mov, warp_moving: bool=True, verbose: bool=False):
    """
    Applies TPS to register moving image to fixed image. Keypoints are filtered by confidence.

    Args:
        keypts_fix: keypoints in fixed image
        keypts_mov: keypoitns in moving image
        confidence: confidences of matches
        thrshld: minimum confidence threshold
        img_mov: Optional moving image
        warp_moving: whether to transform the moving image
        verbose: Whether to produce detailed output (for diagnostic purposes)

    Returns: 
        torch.Tensor: transformed moving image
        ski.transform.ThinPlateSplineTransform: TPS transformation object
    """

    # filter keypoints by confidence
    img_fix_mks = keypts_fix[confidence > thrsld]
    img_mov_mks = keypts_mov[confidence > thrsld]
    if verbose and (len(img_fix_mks) > 500):
        print("Setting threshold..")
    while (len(img_fix_mks) > 500):
        thrsld += (1-thrsld)/5
        img_fix_mks = keypts_fix[confidence > thrsld]
        img_mov_mks = keypts_mov[confidence > thrsld]
    if verbose:
        print(f"Threshold set to {thrsld}")

    # fit and apply TPS
    return tps_skimage(img_fix_mks, img_mov_mks, img_mov=img_mov, warp_moving=warp_moving, verbose=verbose)


def compose_tps(transforms: List[ski.transform.ThinPlateSplineTransform]):
    """Composes TPS transforms"""
    def composed(coords):
        for t in transforms:
            coords = t(coords)
        return coords
    return composed

def register_loftr_tps_skimage(img_fixed: torch.Tensor, img_moving: torch.Tensor, threshold=0.5, mask_moving: torch.Tensor=None, verbose: bool=False, plot_loftr_matches: bool=False, warp_moving: bool=True, return_tps: bool=False):
    """
    Detects loftr matches between fixed and moving image, filters them by confidence, then uses skimage to fit and apply a TPS transform.

    Args:
        img_fixed: fixed image
        img_moving: moving image
        threshold: minimum match confidence threshold
        mask_moving: Optional mask of moving image
        verbose: Whether to produce detailed output (for diagnostic purposes)
        plot_loftr_matches: whether to plot figures showing distribution of matches 
        warp_moving: whether to return the warped moving image. If false, only the TPS is returned.
        return_tps: whether to return the fitted TPS object

    Returns:
        torch.Tensor: registered moving image. only returned if warp_moving==True
        torch.Tensor: mask of registered moving image. only returned if warp_moving==True and mask of moving image is provided.
        ski.transform.ThinPlateSplineTransform: TPS object used to register the image. only returned if return_tps==True.
    """
    if img_fixed is None or img_moving is None:
        if return_tps:
            if mask_moving is not None:
                return None, None, None
            else:
                return None, None
        else:
            if mask_moving is not None:
                return None, None
            else:
                return None

    # detect loftr matches
    mkpts0, mkpts1, confidence, _, n_matches = loftr_match(img_fixed, img_moving, verbose=verbose, return_n_matches=True)

    if plot_loftr_matches:
        fig, ax = plot_matches_conf(img_fixed, mkpts0, img_moving, mkpts1, confidence, N_show=50, vertical=True)
        fig.show()
        fig, axs = plot_match_coverage(img_fixed, mkpts0, img_moving, mkpts1, confidence)
        fig.show()
    
    if n_matches['conf_matches'] > 3:
        warped_moving_img, tps = tps_skimage_confidence(mkpts0, mkpts1, confidence, threshold, img_moving, warp_moving=warp_moving, verbose=verbose)
        warped_moving_img = convert_image_to_tensor(warped_moving_img)
        if not warp_moving:
            return tps
        elif mask_moving is not None:
            # converting mask to bool makes warp use nearest-neighbor interpolation
            warped_moving_mask = warp_tps_skimage(mask_moving.bool(), tps, verbose)
            warped_moving_mask = convert_image_to_tensor(warped_moving_mask)
    else:
        print("No enough matches for TPS found")
        warped_moving_img = None
        warped_moving_mask = None
        tps = None
        if not warp_moving:
            return tps
    
    if return_tps:
        if mask_moving is not None:
            return warped_moving_img, warped_moving_mask, tps
        else:
            return warped_moving_img, tps
    else:
        if mask_moving is not None:
            return warped_moving_img, warped_moving_mask
        else:
            return warped_moving_img


# ----- Filtering Matches -----

def filter_matches_by_confidence(mkpts0: torch.Tensor, mkpts1: torch.Tensor, confidence: torch.Tensor, threshold: float=0.5, n_max: int=500, verbose: bool=False):
    """
    Filters matches by confidence, such that a minimum confidence threshold is satisfied, then increasing the threshold until at most n_max matches remain.
    
    Args:
        mkpts0: keypoints in first/fixed image
        mkpts1: keypoints in second/moving image
        confidence: confidence of each match
        threshold: minimum confidence value accepted
        n_max: maximum number of matches to be returned
        verbose: Whether to produce detailed output (for diagnostic purposes)

    Returns:
        filtered mkpts0 and mkpts1
    """
    img_fix_mks = mkpts0[confidence > threshold]
    img_mov_mks = mkpts1[confidence > threshold]

    if verbose and (len(img_fix_mks) > n_max):
        print("Setting threshold..")
    while (len(img_fix_mks) > n_max):
        threshold += (1-threshold)/5
        img_fix_mks = mkpts0[confidence > threshold]
        img_mov_mks = mkpts1[confidence > threshold]
    if verbose:
        print(f"Threshold set to {threshold}")

    return img_fix_mks, img_mov_mks

def filter_matches_by_confidence_bin_search(mkpts0: torch.Tensor, mkpts1: torch.Tensor, confidence: torch.Tensor, n_target: int=500, tol: int=50, min_conf: float=0.5):
    """
    Filters points by confidence, adjusting the confidence threshold so that the resulting subset of points is within a tolerance of the targeted number of points.
    
    Args:
        mkpts0: keypoints in first/fixed image
        mkpts1: keypoints in second/moving image
        confidence: confidence of each match
        n_target: targeted number of matches to be returned
        tol: tolerated deviation of returned number of points from target
        min_conf: minimum confidence value accepted
        n_max: maximum number of matches to be returned

    Returns:
        filtered mkpts0 and mkpts1
    """
    
    best_0 = mkpts0[confidence > min_conf]
    best_1 = mkpts1[confidence > min_conf]

    if len(best_0) <= 3: 
        # if cutting of at confidence threshold leaves us with too few samples,
        # TPS is no longer possible => take 4 most confident matches
        k = min(len(mkpts0), 4)
        print(f"Too few confident matches. Using top {k} most confident matches.")
        top_ind = confidence.topk(k).indices
        return mkpts0[top_ind], mkpts1[top_ind]

    if len(best_0) <= n_target:
        return best_0, best_1  # nothing to do

    low = min_conf
    high = 1.0 

    for _ in range(20):  # enough iterations to achieve convergence
        mid = (low + high) / 2
        filtered0 = mkpts0[confidence > mid]
        filtered1 = mkpts1[confidence > mid]

        if len(filtered0) > n_target:
            # too many points -> increase threshold
            low = mid
        else:
            # too few points -> decrease threshold
            high = mid
            best_0 = filtered0
            best_1 = filtered1

        if abs(len(filtered0) - n_target) < tol:
            return filtered0, filtered1

    return best_0, best_1

def filter_matches_by_grid(mkpts0: torch.Tensor, mkpts1: torch.Tensor, confidence: torch.Tensor, img_width: IndentationError, threshold: float=0.5, cell_size: int=50):
    """
    Divides image into a grid and picks highest confidence match per grid cell.
    
    Args:
        mkpts0: keypoints in first/fixed image
        mkpts1: keypoints in second/moving image
        confidence: confidence of each match
        img_width: width of the underlying image
        threshold: minimum confidence value accepted
        cell_size: size of each grid cell in px

    Returns:
        filtered mkpts0 and mkpts1
    """
    num_cell_per_row =  math.ceil(img_width / cell_size)

    cell_x = mkpts0[:,0] // cell_size
    cell_y = mkpts0[:,1] // cell_size
    cell_id = cell_y * num_cell_per_row + cell_x

    # get indices of max per cell
    cell_max_indices = group_by_argmax(confidence, cell_id.long())

    cell_max_coord0 = mkpts0[cell_max_indices, :]
    cell_max_coord1 = mkpts1[cell_max_indices, :]

    if threshold is not None:
        cell_max_coord0 = cell_max_coord0[confidence[cell_max_indices] > threshold]
        cell_max_coord1 = cell_max_coord1[confidence[cell_max_indices] > threshold]
    return cell_max_coord0, cell_max_coord1

def filter_matches_by_grid_adaptive(mkpts0: torch.Tensor, mkpts1: torch.Tensor, confidence: torch.Tensor, img_shape: Tuple[int, int], n_target: int=500, tol: int=50, min_conf: float=0.5):
    """
    Iteratively filters points by grid, adjusting the cell size parameter so that the resulting subset of points is within a tolerance of the targeted number of points.
    
    Args:
        mkpts0: (N, 2) array/tensor of keypoints in image 0
        mkpts1: (N, 2) array/tensor of keypoints in image 1
        confidence: (N,) match confidence
        img_shape: shape of the underlying image(s)
        n_target: targeted number of matches
        tol: tolerance indicating by how much the number of matches may deviate from the target 
        threshold: confidence threshold. only matches with confidence above this threshold are considered

    Returns:
        filtered kpts0, filtered kpts1
    """
    
    best_0 = mkpts0[confidence > min_conf]
    best_1 = mkpts1[confidence > min_conf]

    if len(best_0) <= 3: 
        # if cutting of at confidence threshold leaves us with too few samples,
        # TPS is no longer possible => take 4 most confident matches
        k = min(len(mkpts0), 4)
        print(f"Too few confident matches. Using top {k} most confident matches.")
        top_ind = confidence.topk(k).indices
        return mkpts0[top_ind], mkpts1[top_ind]

    if len(best_0) <= n_target:
        return best_0, best_1  # nothing to do

    low = 1e-6  # very small cells → almost all points kept
    high = max(img_shape[-1], img_shape[-2])
    

    for _ in range(20):  # enough iterations to achieve convergence
        mid = (low + high) / 2
        filtered0, filtered1 = filter_matches_by_grid(mkpts0, mkpts1, confidence, img_width=img_shape[3], cell_size=mid, threshold=min_conf)

        if len(filtered0) > n_target:
            # too many points -> increase cell size (i.e. decrease number of cells)
            low = mid
        else:
            # too few points -> decrease cell size (i.e. increase number of cells)
            high = mid
            best_0 = filtered0
            best_1 = filtered1

        if abs(len(filtered0) - n_target) < tol:
            return filtered0, filtered1

    return best_0, best_1

def filter_matches_by_cluster(mkpts0: torch.Tensor, mkpts1: torch.Tensor, confidence: torch.Tensor, min_conf: float=0.5, n_clusters: int=400):
    """
    Clusters matches into clusters and picks the highest confidence match for each cluster.
    
    Args:
        mkpts0: keypoints in first/fixed image
        mkpts1: keypoints in second/moving image
        confidence: confidence of each match
        min_conf: minimum confidence value accepted
        n_clusters: number of clusters

    Returns:
        filtered mkpts0 and mkpts1
    """
    # only consider above threshold
    mkpts0_th = mkpts0[confidence > min_conf]
    mkpts1_th = mkpts1[confidence > min_conf]

    if len(mkpts0_th) <= 3: 
        # if cutting of at confidence threshold leaves us with too few samples,
        # TPS is no longer possible => take 4 most confident matches
        k = min(len(mkpts0), 4)
        print(f"Too few confident matches. Using top {k} most confident matches.")
        top_ind = confidence.topk(k).indices
        return mkpts0[top_ind], mkpts1[top_ind]

    if len(mkpts0_th) <= n_clusters:
        return mkpts0_th, mkpts1_th  # nothing to do

    # cluster remaining matches
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto").fit(mkpts0_th)
    
    # get indices of max per cell
    cluster_max_indices = group_by_argmax(confidence[confidence>min_conf], torch.Tensor(kmeans.labels_))
    cluster_max_coord0 = mkpts0_th[cluster_max_indices, :]
    cluster_max_coord1 = mkpts1_th[cluster_max_indices, :]

    return cluster_max_coord0, cluster_max_coord1

def filter_matches_by_min_distance(mkpts0: torch.Tensor, mkpts1: torch.Tensor, confidence: torch.Tensor, min_dist: float=20.0, max_points: int=None, threshold: float=0.5,):
    """
    Greedy minimum-distance filtering for LoFTR matches.

    Args:
        mkpts0: (N, 2) array/tensor of keypoints in image 0
        mkpts1: (N, 2) array/tensor of keypoints in image 1
        confidence: (N,) match confidence
        min_dist: minimum pixel spacing between selected keypoints
        max_points: optional cap on number of matches
        threshold: confidence threshold. only matches with confidence above this threshold are considered

    Returns:
        filtered_kpts0, filtered_kpts1
    """

    # Convert to numpy if torch
    if type(mkpts0) == torch.Tensor:
        mkpts0 = mkpts0.cpu().numpy()
    if type(mkpts1) == torch.Tensor:
        mkpts1 = mkpts1.cpu().numpy()
    if type(confidence) == torch.Tensor:
        confidence = confidence.cpu().numpy()

    # Sort by confidence descending
    idxs = np.argsort(-confidence)

    selected = []
    selected_points = []
    n_skip = 0

    for idx in idxs:
        # only consider matches satisfying the confidence threshold
        if confidence[idx] < threshold:
            break

        pt = mkpts0[idx]

        if len(selected_points) == 0:
            # first point can just add to list
            selected.append(idx)
            selected_points.append(pt)
        else:
            # compute distances to existing points
            dists = np.linalg.norm(np.array(selected_points) - pt, axis=1)

            # ensure minimum distance isn't violated
            if np.min(dists) >= min_dist:
                selected.append(idx)
                selected_points.append(pt)
            else:
                n_skip += 1

        if max_points is not None and len(selected) >= max_points:
            # print("max points reached")
            break

    selected = np.array(selected)
    # print(f"points skipped: {n_skip}")

    return torch.from_numpy(mkpts0[selected]), torch.from_numpy(mkpts1[selected]), #torch.from_numpy(confidence[selected])   )

def filter_matches_by_min_distance_adaptive(mkpts0: torch.Tensor, mkpts1: torch.Tensor, confidence: torch.Tensor, img_shape: Tuple[int, int], n_target: int=500, tol: int=50, min_conf: float=0.5, max_points: int=None):
    """
    Iteratively filters points by min distance, adjusting the min distance parameter so that the resulting subset of points is within a tolerance of the targeted number of points.
    
    Args:
        mkpts0: (N, 2) array/tensor of keypoints in image 0
        mkpts1: (N, 2) array/tensor of keypoints in image 1
        confidence: (N,) match confidence
        img_shape: shape of the underlying image(s)
        n_target: targeted number of matches
        tol: tolerance indicating by how much the number of matches may deviate from the target 
        min_conf: confidence threshold. only matches with confidence above this threshold are considered
        max_points: optional cap on number of matches

    Returns:
        filtered kpts0, filtered kpts1
    """
    
    best_0 = mkpts0[confidence > min_conf]
    best_1 = mkpts1[confidence > min_conf]

    if len(best_0) <= 3: 
        # if cutting of at confidence threshold leaves us with too few samples,
        # TPS is no longer possible => take 4 most confident matches
        k = min(len(mkpts0), 4)
        print(f"Too few confident matches. Using top {k} most confident matches.")
        top_ind = confidence.topk(k).indices
        return mkpts0[top_ind], mkpts1[top_ind]

    if len(best_0) <= n_target:
        return best_0, best_1  # nothing to do

    low = 0.0
    high = np.sqrt(img_shape[-1]**2 + img_shape[-2]**2)  # e.g. image diagonal
    

    for i in range(20):  # enough iterations to achieve convergence
        mid = (low + high) / 2
        filtered0, filtered1 = filter_matches_by_min_distance(mkpts0, mkpts1, confidence, min_dist=mid, max_points=max_points, threshold=min_conf)

        if len(filtered0) > n_target:
            # too many points -> increase distance
            low = mid
        else:
            # too few points -> decrease distance
            high = mid
            best_0 = filtered0
            best_1 = filtered1

        if abs(len(filtered0) - n_target) < tol:
            return filtered0, filtered1

    return best_0, best_1

def filter_matches(mkpts0: torch.Tensor, mkpts1: torch.Tensor, confidence: torch.Tensor, img_shape: Tuple[int, int], filtering_strategy: str="confidence", n_target: int=500, tol: int=50, min_conf: float=0.5):
    """
    Filter matches using the chosen strategy, resulting in n_target+/-tol matches.

    Args:
        mkpts0: (N, 2) array/tensor of keypoints in image 0
        mkpts1: (N, 2) array/tensor of keypoints in image 1
        confidence: (N,) match confidence
        img_shape: shape of the underlying image(s)
        filtering_strategy: which strategy to use for filtering. One of "confidence", "grid", "clusters", "min_distance"
        n_target: targeted number of matches
        tol: tolerance indicating by how much the number of matches may deviate from the target 
        min_conf: confidence threshold. only matches with confidence above this threshold are considered
        max_points: optional cap on number of matches

    Returns:
        filtered kpts0, filtered kpts1
    """
    if filtering_strategy == "confidence":
        return filter_matches_by_confidence_bin_search(mkpts0, mkpts1, confidence, n_target=n_target, tol=tol, min_conf=min_conf)
    elif filtering_strategy == "grid":
        return filter_matches_by_grid_adaptive(mkpts0, mkpts1, confidence, img_shape, n_target=n_target, tol=tol, min_conf=min_conf)
    elif filtering_strategy == "clusters":
        return filter_matches_by_cluster(mkpts0, mkpts1, confidence, min_conf=min_conf, n_clusters=n_target)
    elif filtering_strategy == "min_distance":
        return filter_matches_by_min_distance_adaptive(mkpts0, mkpts1, confidence, img_shape, n_target=n_target, tol=tol, min_conf=min_conf)
    else:
        raise ValueError(f"Unknown filtering strategy {strategy}. Expected on of 'confidence', 'grid', 'clusters', 'min_distance'.")

# ----- warp consistency -----

def nearest_neighbors(pts1: torch.Tensor, pts2: torch.Tensor):
    """
    Finds the nearest neighbor in pts2 of each point in pts1.

    Args:
        pts1: List of points, shape (N, D)
        pts2: List of points, shape (M, D)

    Returns:
        Tensor of indices, such that pts2[indices[i]] is the nearest neighbor of pts[i]
    """
    # pairwise distances: dists[i,j] = dist(pts1[i], pts2[j])
    dists = torch.cdist(pts1, pts2)  # (N, M)

    # nearest neighbor in pts2 for each pts1
    min_dists, indices = torch.min(dists, dim=1)

    return indices

def cycle_matches(kpts12_2: torch.Tensor, kpts23_2: torch.Tensor, kpts23_3: torch.Tensor, kpts31_3: torch.Tensor, kpts31_1: torch.Tensor, return_indices: bool=False):
    """
    Cycles from Img1 to Img2 to Img3, picking always the nearest neighbor from the new set of matches.

    Args:
        kpts12_2: matches in Img2 from match detection Img1 -> Img2
        kpts23_2: matches in Img2 from match detection Img2 -> Img3
        kpts23_3: matches in Img3 from match detection Img2 -> Img3
        kpts31_3: matches in Img3 from match detection Img3 -> Img1
        kpts31_1: matches in Img1 from match detection Img3 -> Img1
        return_indices: whether to return the indices of nearest neighbors 

    Returns:
        the matches in Img1 after they've made a full cycle (i.e. a subset of kpts31_1)
    """
    nearest_neighbors_img2 = nearest_neighbors(kpts12_2, kpts23_2)
    nearest_neighbors_img3 = nearest_neighbors(kpts23_3, kpts31_3)

    # kpts23_2 = kpts23_2[nearest_neighbors_img2]
    # kpts23_3 = kpts23_3[nearest_neighbors_img2]
    # kpts31_3 = kpts31_3[nearest_neighbors_img3[nearest_neighbors_img2]]
    kpts31_1 = kpts31_1[nearest_neighbors_img3[nearest_neighbors_img2]]
    if return_indices:
        return kpts31_1, nearest_neighbors_img2, nearest_neighbors_img3
    else:
        return kpts31_1

def plot_cycle_matches(img1: torch.Tensor, img2: torch.Tensor, img3: torch.Tensor, kpts12_1: torch.Tensor, kpts12_2: torch.Tensor, kpts23_2: torch.Tensor, kpts23_3: torch.Tensor, kpts31_3: torch.Tensor, kpts31_1: torch.Tensor, nearest_neighbors_img2, nearest_neighbors_img3, N_show: int=50):
    """
    Plots the three images of warp consistency, and the cycle through the nearest-neighbor matches and their displacements.

    Args:
        img1: first image (fixed image)
        img2: second image (moving image)
        img3: third image (warped moving image)
        kpts12_1: matches in Img1 from match detection Img1 -> Img2
        kpts12_2: matches in Img2 from match detection Img1 -> Img2
        kpts23_2: matches in Img2 from match detection Img2 -> Img3
        kpts23_3: matches in Img3 from match detection Img2 -> Img3
        kpts31_3: matches in Img3 from match detection Img3 -> Img1
        kpts31_1: matches in Img1 from match detection Img3 -> Img1
        nearest_neighbors_img2: indices of nearest neighbors between kpts12_2 and kpts23_2
        nearest_neighbors_img3: indices of nearest neighbors between kpts23_3 and kpts31_3
        N_show: number of matches to show
    """
    N_show = min(N_show, len(kpts12_1))
    show_idx = torch.randperm(len(kpts12_1))[:N_show]

    kpts12_1 = kpts12_1[show_idx]
    kpts12_2 = kpts12_2[show_idx]
    kpts23_2 = kpts23_2[nearest_neighbors_img2[show_idx]]
    kpts23_3 = kpts23_3[nearest_neighbors_img2[show_idx]]
    kpts31_3 = kpts31_3[nearest_neighbors_img3[nearest_neighbors_img2[show_idx]]]
    kpts31_1 = kpts31_1[nearest_neighbors_img3[nearest_neighbors_img2[show_idx]]]

    fig, ax = plt.subplots(figsize=(14,14))
    img_list = [img1, img2, img3, img1]

    max_width = max(img.shape[-1] for img in img_list)

    for i, img in enumerate(img_list):
        padder = K.augmentation.PadTo((img.shape[-2], max_width))
        img_list[i] = padder(img)

    img_set = np.concatenate([K.tensor_to_image(img) for img in img_list], axis=0)
    ax.imshow(img_set, cmap='gray')

    prev = 0
    for i in range(len(img_list)):
        ax.hlines(y=prev+img_list[i].shape[2], xmin=0, xmax=img_list[i].shape[3]-1, color='grey', linewidth=1)
        prev += img_list[i].shape[2]

    color = 'cyan'
    y_disp = 0
    for (x0, y0), (x1, y1) in zip(kpts12_1, kpts12_2):
        ax.scatter([x0, x1 ], [y0 + y_disp, y1 + y_disp + img1.shape[2]], color=color, s=7)
        ax.plot([x0, x1 ], [y0 + y_disp, y1 + y_disp + img1.shape[2]], color=color, linewidth=2)
    y_disp += img1.shape[2]

    for (x0, y0), (x1, y1) in zip(kpts12_2, kpts23_2):
        ax.plot([x0, x1 ], [y0 + y_disp, y1 + y_disp], color='white', linewidth=2)

    color = 'orange'
    for (x0, y0), (x1, y1) in zip(kpts23_2, kpts23_3):
        ax.scatter([x0, x1 ], [y0 + y_disp, y1 + y_disp + img2.shape[2]], color=color, s=7)
        ax.plot([x0, x1 ], [y0 + y_disp, y1 + y_disp + img2.shape[2]], color=color, linewidth=2)
    y_disp += img2.shape[2]

    for (x0, y0), (x1, y1) in zip(kpts23_3, kpts31_3):
        ax.plot([x0, x1 ], [y0 + y_disp, y1 + y_disp], color='white', linewidth=2)

    color = 'lime'
    for (x0, y0), (x1, y1) in zip(kpts31_3, kpts31_1):
        ax.scatter([x0, x1 ], [y0 + y_disp, y1 + y_disp + img3.shape[2]], color=color, s=7)
        ax.plot([x0, x1 ], [y0 + y_disp, y1 + y_disp + img3.shape[2]], color=color, linewidth=2)
    y_disp += img3.shape[2]

    for (x0, y0), (x1, y1) in zip(kpts31_1, kpts12_1):
        ax.plot([x0, x1 ], [y0 + y_disp, y1 + y_disp], color='white', linewidth=2)

    color = 'red'
    ax.scatter(kpts12_1[:,0], kpts12_1[:,1] + y_disp, color=color, s=7)

    ax.axis('off')
    plt.show()

def check_warp_consistency(
    img_fixed: torch.Tensor, 
    img_moving: torch.Tensor, 
    mask_fixed: torch.Tensor=None,
    mask_moving: torch.Tensor=None, 
    plot_matches: bool=False, 
    consistency_tolerance: float=10, 
    transform: dict={'type': 'rotation', 'params': {'rotation': -10}},
    verbose: bool=False
    ):
    """
    Detects matches between fixed and moving image and filters out inconsistent matches.
    To evaluate consistent, a third image is generated through a transformation of the moving image. LoFTR matches are then detected between image pair. the correspondences are then matched to their nearest neighbor amongst the correspondes between the next image pair, resulting in cycle of matches. finally the displacement between the first an last point is measured. if the displacement it too large, the correspondence is considered inconsistent.

    Args:
        img_fixed: fixed image 
        img_moving: moving image
        mask_fixed: Optional mask of fixed image
        mask_moving: Optional mask of moving image
        plot_matches: whether to plot figures showing distribution of matches and warp consistency cycles
        consistency_tolerance: threshold for displacement after traveling through the cycle. matches with larger displacement are discarded as inconsistent
        transform: dictionary specifying the transformation used to generate the third image. by default 'type'='rotation'. alternatively, 'type'='homography' is also supported.
        verbose: Whether to produce detailed output (for diagnostic purposes)

    Returns
        torch.Tensor: consistent matches in fixed image
        torch.Tensor: consistent matches in moving image
        torch.Tensor: confidence of consistent matches
    """
        

    img1 = img_fixed
    img1_mask = mask_fixed
    img2 = img_moving
    img2_mask = mask_moving
    # create third image
    if transform["type"] == "rotation":
        out = affine_warp_expand(img2, img2_mask, rot_angle_deg=transform['params']['rotation'])
        img3 = out["imgs"]
        if img2_mask is None:
            img3_mask = None
        else:
            img3_mask = out["masks"]
    elif transform["type"] == "homography":
        hom = RandomHomography(img_fixed.shape[2], img_fixed.shape[3], distortion_scale=transform['params']['distortion_scale'])
        img3 = hom.warp_image(img_moving)
        img3_mask = hom.warp_mask(mask_moving)
    else:
        raise ValueError(f"Unknown transform_type {transform_type}. Expected 'rotation' or 'homography'.")

    # detect loftr matches
    if verbose:
        print(f"Detecting LoFTR Matches...")
    # 1 -> 2
    mkpts12_1, mkpts12_2, confidence_12, _,= loftr_match(img1, img2, img1_mask, img2_mask, verbose=verbose, return_n_matches=False)
    # 2 -> 3
    mkpts23_2, mkpts23_3, confidence_23, _,= loftr_match(img2, img3, img2_mask, img3_mask, verbose=verbose, return_n_matches=False)
    # 3 -> 1
    # mkpts31_3, mkpts31_1, confidence_31, _,= loftr_match(img3, img1, verbose=verbose, return_n_matches=False)
    mkpts13_1, mkpts13_3, confidence_13, _ = loftr_match(img1, img3, img1_mask, img3_mask, verbose=verbose, return_n_matches=False)
    if verbose:
        print(f"Img1 -> Img2: {len(mkpts12_1)} Matches")
        print(f"Img2 -> Img3: {len(mkpts23_2)} Matches")
        print(f"Img3 -> Img1: {len(mkpts13_3)} Matches")

    if plot_matches:
        _ = plot_match_coverage(img1, mkpts12_1, img2, mkpts12_2, confidence_12)
        _ = plot_match_coverage(img2, mkpts23_2, img3, mkpts23_3, confidence_23)
        # _ = plot_match_coverage(img3, mkpts31_3, img1, mkpts31_1, confidence_31)
        _ = plot_match_coverage(img3, mkpts13_3, img1, mkpts13_1, confidence_13)


    if plot_matches:
        # cycled_31_1, nn_ind_2, nn_ind_3 = cycle_matches(mkpts12_2, mkpts23_2, mkpts23_3, mkpts31_3, mkpts31_1, return_indices=True)
        # plot_cycle_matches(img1, img2, img3, mkpts12_1, mkpts12_2, mkpts23_2, mkpts23_3, mkpts31_3, mkpts31_1, nn_ind_2, nn_ind_3, N_show=30)
        cycled_31_1, nn_ind_2, nn_ind_3 = cycle_matches(mkpts12_2, mkpts23_2, mkpts23_3, mkpts13_3, mkpts13_1, return_indices=True)
        plot_cycle_matches(img1, img2, img3, mkpts12_1, mkpts12_2, mkpts23_2, mkpts23_3, mkpts13_3, mkpts13_1, nn_ind_2, nn_ind_3, N_show=30)
    else:
        # find cycle through matches

        # cycled_31_1 = cycle_matches(mkpts12_2, mkpts23_2, mkpts23_3, mkpts31_3, mkpts31_1)
        cycled_31_1 = cycle_matches(mkpts12_2, mkpts23_2, mkpts23_3, mkpts13_3, mkpts13_1)
    dists = torch.norm(mkpts12_1 - cycled_31_1, dim=1)
    is_consistent = (dists < consistency_tolerance)

    if verbose:
        print(f"Number of consistent matches: {int(is_consistent.sum())}")
        print(f"Ratio of consistent matches: {is_consistent.to(torch.float32).mean():.3f}")
        print(f"Least confident of consistent matches: {confidence_12[is_consistent].min():.3f}")
    if plot_matches:
        _ = plot_match_coverage(img1, mkpts12_1[is_consistent], img2, mkpts12_2[is_consistent], confidence_12[is_consistent])
        # _ = plot_match_coverage(img1, cycled_31_1[is_consistent], img1, mkpts12_1[is_consistent], is_consistent[is_consistent])

    return mkpts12_1[is_consistent], mkpts12_2[is_consistent], confidence_12[is_consistent]


def fetch_keypoints(
    img_fixed: torch.Tensor,
    img_moving: torch.Tensor, 
    mask_fixed: torch.Tensor=None,
    mask_moving: torch.Tensor=None,
    warp_consistency: dict=CONSISTENCY_DEFAULT,
    match_filtering: dict=FILTERING_DEFAULT,
    verbose: bool=False,
):
    """
    Detects matches between fixed and moving, filters them by warp consistency, then reduces the number of matches to the target amount. 
    Also tests for a 180° degree rotation between fixed and moving image.
    
    Args:
        img_fixed: fixed image 
        img_moving: moving image
        mask_fixed: Optional mask of fixed image
        mask_moving: Optional mask of moving image
        warp_consistency: dictionary specifying parameters for warp consistency. to disable warp consistency, set it to None.
        match_filtering: dictionary specifying parameters for match filtering/subsampling, such as filtering strategy, target number of landmarks, and minimum confidence threshold.
        verbose: Whether to produce detailed output (for diagnostic purposes)

    Returns:
        dictionary:
            - mkpts0: consistent matches in fixed image
            - mkpts1: consistent matches in moving image
            - confidence: confidences of consistend matches
            - mkpts0_filtered: filtered consistent matches in fixed image
            - mkpts1_filtered: filtered consistent matches in moving image
            - rotated_moving_img: moving image rotate by 180°. only present if it was determined that the moving image is rotated by 180° relative to the fixed image.
            - rotated_moving_mask: mask moving image rotate by 180°. only present if it was determined that the moving image is rotated by 180° relative to the fixed image.
    """
    out_dict = {}

    # find loftr matches
    if warp_consistency is not None: 
        # use only consistent matches
        mkpts0, mkpts1, confidence = check_warp_consistency(img_fixed, img_moving, mask_fixed, mask_moving, plot_matches=False, verbose=verbose, **warp_consistency)
    else:
        mkpts0, mkpts1, confidence, _ = loftr_match(img_fixed, img_moving, mask_fixed, mask_moving, verbose=verbose, return_n_matches=False)

    # check for 180 degree rotations between images
    if not check_orientation(mkpts0, mkpts1):
        # images are likely in different orientations -> loftr struggles

        # rotate moving image        
        img_moving_rot = affine(img_moving, rot_angle_deg=180)
        mask_moving_rot = affine(mask_moving, rot_angle_deg=180, interpolation_mode='nearest')

        # detect matches to rotated image
        if warp_consistency is not None: 
            # use only consistent matches
            mkpts0_rot, mkpts1_rot, confidence_rot = check_warp_consistency(img_fixed, img_moving_rot, mask_fixed, mask_moving_rot, plot_matches=False, verbose=verbose, **warp_consistency)
        else:
            mkpts0_rot, mkpts1_rot, confidence_rot, _ = loftr_match(img_fixed, img_moving_rot, mask_fixed, mask_moving_rot, verbose=verbose, return_n_matches=False)

        # more matches with rotated image => stick with rotated moving image
        if len(mkpts0_rot) > len(mkpts0):
            print("Flipped Orientation confirmed.")
            img_moving = img_moving_rot
            mask_moving = mask_moving_rot
            mkpts0 = mkpts0_rot
            mkpts1 = mkpts1_rot
            confidence = confidence_rot

            out_dict.update({"rotated_moving_img": img_moving_rot})
            out_dict.update({"rotated_moving_mask": mask_moving_rot})
        else:
            print("Flipped Orientation dismissed.")

    out_dict.update({"mkpts0": mkpts0})
    out_dict.update({"mkpts1": mkpts1})
    out_dict.update({"confidence": confidence})

    # reduce number of matches
    filtering_mapped = { # rename arguments
        "filtering_strategy": match_filtering["filtering_strategy"],
        "n_target": match_filtering["n_landmarks"],
        "tol": match_filtering["n_landmarks_tol"],
        "min_conf": match_filtering["min_conf"],
    }
    mkpts0_filtered, mkpts1_filtered = filter_matches(mkpts0, mkpts1, confidence, img_fixed.shape, **filtering_mapped)

    out_dict.update({"mkpts0_filtered": mkpts0_filtered})
    out_dict.update({"mkpts1_filtered": mkpts1_filtered})
    

    # return mkpts0, mkpts1, confidence, mkpts0_filtered, mkpts1_filtered, img_moving, mask_moving
    return out_dict

