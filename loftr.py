import os
import cv2
import kornia as K
import kornia.feature as KF
import numpy as np
import torch
import skimage as ski
import math
from sklearn.cluster import KMeans
from torch_tps import ThinPlateSpline
from utils import convert_image_to_tensor, group_by_argmax
from plotting import plot_matches_conf, plot_match_coverage


def load_resize_image(img_path: str, H: int=375, W: int=600):
    """
    Loads and resizes image at path to desired dimensions

    Args:
        img_path: path to image
        H: desired height of output image
        D: desired height of output image
    """

    assert os.path.exists(img_path), "Invalid path to images"

    img = K.io.load_image(img_path, K.io.ImageLoadType.RGB32)[None, ...]
    img = K.geometry.resize(img, (H, W), antialias=True)

    return img


def loftr_match(img_fix, img_mov, verbose: bool=True, return_n_matches: bool=False):
    """
    Detects Feature matches between fixed and moving images using LoFTR

    Returns:
        Keypoints in fixed image
        Keypoints in moving image
        Classification of inliers, using RANSAC/Fundamental matrix
        Confidence of matches
    """

    # match with LoFTR
    matcher = KF.LoFTR(pretrained="outdoor")

    # Add batch dim if needed
    if img_fix.dim() == 3:
        img_fix = img_fix.unsqueeze(0)
    if img_mov.dim() == 3:
        img_mov = img_mov.unsqueeze(0)

    input_dict = {
        "image0": K.color.rgb_to_grayscale(img_fix),  # LofTR works on grayscale images only
        "image1": K.color.rgb_to_grayscale(img_mov),
    }

    with torch.inference_mode():
        correspondences = matcher(input_dict)
    

    # select inliers
    mkpts0 = correspondences["keypoints0"]#.cpu().numpy()
    mkpts1 = correspondences["keypoints1"]#.cpu().numpy()
    confidence = correspondences["confidence"]
    # _, inliers = cv2.findFundamentalMat(mkpts0.cpu().numpy(), mkpts1.cpu().numpy(), cv2.USAC_MAGSAC, 0.5, 0.999, 100000)
    # inliers = inliers > 0

    if mkpts0.shape[0] < 8 or mkpts1.shape[0] < 8:
        print("Not enough points to perform inlier detection.")
        n_inliers = None
        inliers = None
    else:
        _, inliers = cv2.findFundamentalMat(mkpts0.cpu().numpy(), mkpts1.cpu().numpy(), cv2.USAC_MAGSAC, 0.5, 0.999, 100000)
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



# -------------------------------- torch tps -----------------------------------------

def fit_tps_torch(target_keypts, moving_keypts, alpha=0.5):
    """
    keypoints have format (x,y)
    """

    target_keypts = target_keypts[..., [1, 0]] # meshgrid builder expects (y,x) format
    moving_keypts = moving_keypts[..., [1, 0]]

    # Fit the thin plate spline from output to input
    tps = ThinPlateSpline(alpha)
    tps.fit(target_keypts, moving_keypts)

    return tps

def warp_tps_points_torch(tps, points):
    points = points[..., [1, 0]] # tps was trained in (y,x) format
    return tps.transform(points)[..., [1, 0]] # return output back in (x,y) format

def warp_tps_torch(tps_list: list, image, interpolation_mode='bilinear'):
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

    # grid_x = (2 * x + 1) / W - 1
    # grid_y = (2 * y + 1) / H - 1
    # grid = torch.stack((grid_x, grid_y), dim=-1)
    return torch.nn.functional.grid_sample(image, grid[None], mode=interpolation_mode, align_corners=False)

def torch_tps(target_keypts, moving_keypts, moving_img, alpha: float=0.5, verbose: bool=False):
    if verbose:
        print("Fitting TPS...")
    tps = fit_tps_torch(target_keypts, moving_keypts, alpha=alpha)
    if verbose:
        print("Warping Moving Image...")
    warped = warp_tps_torch(tps, moving_img)
    return warped

def register_loftr_tps(img_fixed, img_moving, threshold: float=0.5, smoothing: float=0.5, mask_moving: torch.Tensor=None, verbose: bool=False, plot_loftr_matches: bool=False, return_tps: bool=False):
    """
    uses loftr to detect matches between the fixed and moving image, filters the matches by confidence, then uses TPS to transform the moving image
    if a mask of the moving image is provided, it is also warped.
    optionally the TPS transform can be returned
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

# ------------------- skimage tps ----------------------------------------------------------

def warp_tps_skimage(img, tps, verbose=False):
    # kornia and torch expect C x H x W, while skimage expects H x W x C
    if type(img) == torch.Tensor:
        img = K.tensor_to_image(img)

    if verbose:
        print("Transforming moving image...")
    warped = ski.transform.warp(img, tps) # warp uses inverse transform, i.e. img_mov -> img_fix

    return convert_image_to_tensor(warped)

def tps_skimage(keypts_fix, keypts_mov, img_mov=None, warp_moving: bool=True, verbose: bool=False):
    """
    Applies TPS to register moving image to fixed image. Expects keypoints to be filtered already.

    Returns: transformed moving image and transform function
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


def compose_tps(transforms):
    def composed(coords):
        for t in transforms:
            coords = t(coords)
        return coords
    return composed

def register_loftr_tps_skimage(img_fixed, img_moving, threshold=0.5, mask_moving: torch.Tensor=None, verbose: bool=False, plot_loftr_matches: bool=False, warp_moving: bool=True, return_tps: bool=False):
    """
    if `warp_moving` is False, the moving image is not warped and only the tps transform is returned
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



# Filtering -------------------------------

def tps_skimage_confidence(keypts_fix, keypts_mov, confidence, thrsld, img_mov, warp_moving: bool=True, verbose: bool=False):
    """
    Applies TPS to register moving image to fixed image. Keypoints are filtered by confidence.

    Returns: transformed moving image and transform function
    """

    # kornia and torch expect C x H x W, while skimage expects H x W x C
    # img_mov_reordered = K.tensor_to_image(img_mov)

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

    # if verbose:
    #     print("Estimating TPS transform...")
    # tps = ski.transform.ThinPlateSplineTransform.from_estimate(img_fix_mks, img_mov_mks)

    return tps_skimage(img_fix_mks, img_mov_mks, img_mov=img_mov, warp_moving=warp_moving, verbose=verbose)

    # if warp_moving:
    #     if verbose:
    #         print("Transforming moving image...")
    #     warped = ski.transform.warp(img_mov_reordered, tps) # warp uses inverse transform, i.e. img_mov -> img_fix

    #     return warped, tps
    # else:
    #     return None, tps

def filter_matches_by_confidence(mkpts0, mkpts1, confidence, threshold: float=0.5, n_max: int=500, verbose: bool=False):
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

def filter_matches_by_grid(mkpts0, mkpts1, confidence, img_width, threshold: float=0.5, cell_size: int=50):
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

def filter_matches_by_cluster(mkpts0, mkpts1, confidence, threshold: float=0.5, n_clusters: int=400):

    # only consider above threshold
    mkpts0_th = mkpts0[confidence > threshold]
    mkpts1_th = mkpts1[confidence > threshold]

    # cluster remaining matches
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto").fit(mkpts0_th)
    
    # get indices of max per cell
    cluster_max_indices = group_by_argmax(confidence[confidence>threshold], torch.Tensor(kmeans.labels_))
    cluster_max_coord0 = mkpts0_th[cluster_max_indices, :]
    cluster_max_coord1 = mkpts1_th[cluster_max_indices, :]

    return cluster_max_coord0, cluster_max_coord1

def filter_matches_by_min_distance(mkpts0, mkpts1, confidence, min_dist: float=20.0, max_points: int=None, threshold: float=0.5,):
    """
    Greedy minimum-distance filtering for LoFTR matches.

    Args:
        kpts0: (N, 2) array/tensor of keypoints in image 0
        kpts1: (N, 2) array/tensor of keypoints in image 1
        confidence: (N,) match confidence
        min_dist: minimum pixel spacing between selected keypoints
        max_points: optional cap on number of matches

    Returns:
        filtered_kpts0, filtered_kpts1, filtered_conf
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

    



