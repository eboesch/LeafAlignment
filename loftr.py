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


def loftr_match(img_fix: torch.Tensor, img_mov: torch.Tensor, mask_fix: torch.Tensor=None, mask_mov: torch.Tensor=None, verbose: bool=True, return_n_matches: bool=False):
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
    if (mask_fix is not None) and (mask_mov is not None):
        if mask_fix.dim() == 2:
            mask_fix = mask_fix.unsqueeze(0)
        if mask_mov.dim() == 2:
            mask_mov = mask_mov.unsqueeze(0)
        if mask_fix.dim() == 4:
            mask_fix = mask_fix.squeeze(0)
        if mask_mov.dim() == 4:
            mask_mov = mask_mov.squeeze(0)

    # if (mask_fix is not None) and (mask_mov is not None):
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

def filter_matches_by_confidence_bin_search(mkpts0, mkpts1, confidence, n_target: int=500, tol: int=50, min_conf: float=0.5):
    """
    Filters points by confidence, adjusting the confidence threshold so that the resulting subset of points is within a tolerance of the targeted number of points.
    """
    
    best_0 = mkpts0[confidence > min_conf]
    best_1 = mkpts1[confidence > min_conf]

    if len(best_0) <= n_target:
        return best_0, best_1  # nothing to do

    low = min_conf
    high = 1.0 

    for _ in range(20):  # enough iterations to achieve convergence
        mid = (low + high) / 2
        filtered0 = mkpts0[confidence > mid]
        filtered1 = mkpts1[confidence > mid]

        if len(filtered0) > n_target:
            # too many points → increase threshold
            low = mid
        else:
            # too few points → decrease threshold
            high = mid
            best_0 = filtered0
            best_1 = filtered1

        if abs(len(filtered0) - n_target) < tol:
            return filtered0, filtered1

    return best_0, best_1

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

def filter_matches_by_grid_adaptive(mkpts0, mkpts1, confidence, img_shape, n_target: int=500, tol: int=50, threshold: float=0.5):
    """
    Iteratively filters points by min grid, adjusting the cell size parameter so that the resulting subset of points is within a tolerance of the targeted number of points.
    
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
    
    best_0 = mkpts0[confidence > threshold]
    best_1 = mkpts1[confidence > threshold]

    if len(best_0) <= n_target:
        return best_0, best_1  # nothing to do

    low = 1e-6  # very small cells → almost all points kept
    high = max(img_shape[-1], img_shape[-2])
    

    for _ in range(20):  # enough iterations to achieve convergence
        mid = (low + high) / 2
        filtered0, filtered1 = filter_matches_by_grid(mkpts0, mkpts1, confidence, img_width=img_shape[3], cell_size=mid, threshold=threshold)

        if len(filtered0) > n_target:
            # too many points → increase cell size (i.e. decrease number of cells)
            low = mid
        else:
            # too few points → decrease cell size (i.e. increase number of cells)
            high = mid
            best_0 = filtered0
            best_1 = filtered1

        if abs(len(filtered0) - n_target) < tol:
            return filtered0, filtered1

    return best_0, best_1


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


def filter_matches_by_min_distance_adaptive(mkpts0, mkpts1, confidence, img_shape, n_target: int=500, tol: int=50, max_points: int=500, threshold: float=0.5):
    """
    Iteratively filters points by min distance, adjusting the min distance parameter so that the resulting subset of points is within a tolerance of the targeted number of points.
    
    Args:
        mkpts0: (N, 2) array/tensor of keypoints in image 0
        mkpts1: (N, 2) array/tensor of keypoints in image 1
        confidence: (N,) match confidence
        img_shape: shape of the underlying image(s)
        n_target: targeted number of matches
        tol: tolerance indicating by how much the number of matches may deviate from the target 
        threshold: confidence threshold. only matches with confidence above this threshold are considered
        max_points: optional cap on number of matches

    Returns:
        filtered kpts0, filtered kpts1
    
    """
    
    best_0 = mkpts0[confidence > threshold]
    best_1 = mkpts1[confidence > threshold]

    if len(best_0) <= n_target:
        return best_0, best_1  # nothing to do

    low = 0.0
    high = np.sqrt(img_shape[-1]**2 + img_shape[-2]**2)  # e.g. image diagonal
    

    for i in range(20):  # enough iterations to achieve convergence
        mid = (low + high) / 2
        filtered0, filtered1 = filter_matches_by_min_distance(mkpts0, mkpts1, confidence, min_dist=mid, max_points=max_points, threshold=threshold)

        if len(filtered0) > n_target:
            # too many points → increase distance
            low = mid
        else:
            # too few points → decrease distance
            high = mid
            best_0 = filtered0
            best_1 = filtered1

        if abs(len(filtered0) - n_target) < tol:
            return filtered0, filtered1

    return best_0, best_1


# warp consistency ---------------

def nearest_neighbors(pts1, pts2):
    """
    pts1: (N, D)
    pts2: (M, D)

    Returns:
        Tensor of indices, such that pts2[indices[i]] is the nearest neighbor of pts[i]
    """
    # pairwise distances: dists[i,j] = dist(pts1[i], pts2[j])
    dists = torch.cdist(pts1, pts2)  # (N, M)

    # nearest neighbor in pts2 for each pts1
    min_dists, indices = torch.min(dists, dim=1)

    return indices#, min_dists

def cycle_matches(kpts12_2, kpts23_2, kpts23_3, kpts31_3, kpts31_1, return_indices: bool=False):
    """
    Cycles from Img1 to Img2 to Img3, picking always to nearest neighbor from the new set of matches.

    Returns the matches in Img1 after they've made a full cycle
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

def plot_cycle_matches(img1, img2, img3, kpts12_1, kpts12_2, kpts23_2, kpts23_3, kpts31_3, kpts31_1, nearest_neighbors_img2, nearest_neighbors_img3, N_show=50):
    N_show = min(N_show, len(kpts12_1))
    show_idx = torch.randperm(len(kpts12_1))[:N_show]

    kpts12_1 = kpts12_1[show_idx]
    kpts12_2 = kpts12_2[show_idx]
    kpts23_2 = kpts23_2[nearest_neighbors_img2[show_idx]]
    kpts23_3 = kpts23_3[nearest_neighbors_img2[show_idx]]
    kpts31_3 = kpts31_3[nearest_neighbors_img3[nearest_neighbors_img2[show_idx]]]
    kpts31_1 = kpts31_1[nearest_neighbors_img3[nearest_neighbors_img2[show_idx]]]

    fig, ax = plt.subplots(figsize=(8,6))
    img_list = [img1, img2, img3, img1]
    img_set = np.concatenate([K.tensor_to_image(img1), K.tensor_to_image(img2), K.tensor_to_image(img3), K.tensor_to_image(img1)], axis=0)
    ax.imshow(img_set, cmap='gray')

    prev = 0
    for i in range(len(img_list)):
        ax.hlines(y=prev+img_list[i].shape[2], xmin=0, xmax=img_list[i].shape[3]-1, color='grey', linewidth=1)
        prev += img_list[i].shape[2]

    color = 'cyan'
    y_disp = 0
    for (x0, y0), (x1, y1) in zip(kpts12_1, kpts12_2):
        ax.scatter([x0, x1 ], [y0 + y_disp, y1 + y_disp + img1.shape[2]], color=color, s=2)
        ax.plot([x0, x1 ], [y0 + y_disp, y1 + y_disp + img1.shape[2]], color=color, linewidth=1)
    y_disp += img1.shape[2]

    for (x0, y0), (x1, y1) in zip(kpts12_2, kpts23_2):
        ax.plot([x0, x1 ], [y0 + y_disp, y1 + y_disp], color='white', linewidth=1)

    color = 'orange'
    for (x0, y0), (x1, y1) in zip(kpts23_2, kpts23_3):
        ax.scatter([x0, x1 ], [y0 + y_disp, y1 + y_disp + img2.shape[2]], color=color, s=2)
        ax.plot([x0, x1 ], [y0 + y_disp, y1 + y_disp + img2.shape[2]], color=color, linewidth=1)
    y_disp += img2.shape[2]

    for (x0, y0), (x1, y1) in zip(kpts23_3, kpts31_3):
        ax.plot([x0, x1 ], [y0 + y_disp, y1 + y_disp], color='white', linewidth=1)

    color = 'lime'
    for (x0, y0), (x1, y1) in zip(kpts31_3, kpts31_1):
        ax.scatter([x0, x1 ], [y0 + y_disp, y1 + y_disp + img3.shape[2]], color=color, s=2)
        ax.plot([x0, x1 ], [y0 + y_disp, y1 + y_disp + img3.shape[2]], color=color, linewidth=1)
    y_disp += img3.shape[2]

    for (x0, y0), (x1, y1) in zip(kpts31_1, kpts12_1):
        ax.plot([x0, x1 ], [y0 + y_disp, y1 + y_disp], color='white', linewidth=1)

    color = 'red'
    ax.scatter(kpts12_1[:,0], kpts12_1[:,1] + y_disp, color=color, s=2)

    ax.axis('off')
    plt.show()

def warp_consistency(img_fixed, img_moving, plot_matches: bool=False, distortion_scale=0.4, tolerance=50, verbose: bool=False):
    hom = RandomHomography(img_fixed.shape[2], img_fixed.shape[3], distortion_scale=distortion_scale)

    img1 = img_fixed
    img2 = img_moving
    img3 =  hom.warp_image(img_moving)

    if verbose:
        print(f"Detecting LoFTR Matches...")
    # 1 -> 2
    mkpts12_1, mkpts12_2, confidence_12, _,= loftr_match(img1, img2, verbose=verbose, return_n_matches=False)
    # 2 -> 3
    mkpts23_2, mkpts23_3, confidence_23, _,= loftr_match(img2, img3, verbose=verbose, return_n_matches=False)
    # 3 -> 1
    mkpts31_3, mkpts31_1, confidence_31, _,= loftr_match(img3, img1, verbose=verbose, return_n_matches=False)
    if verbose:
        print(f"Img1 -> Img2: {len(mkpts12_1)} Matches")
        print(f"Img2 -> Img3: {len(mkpts23_2)} Matches")
        print(f"Img3 -> Img1: {len(mkpts31_3)} Matches")

    if plot_matches:
        _ = plot_match_coverage(img1, mkpts12_1, img2, mkpts12_2, confidence_12)
        _ = plot_match_coverage(img2, mkpts23_2, img3, mkpts23_3, confidence_23)
        _ = plot_match_coverage(img3, mkpts31_3, img1, mkpts31_1, confidence_31)


    if plot_matches:
        cycled_31_1, nn_ind_2, nn_ind_3 = cycle_matches(mkpts12_2, mkpts23_2, mkpts23_3, mkpts31_3, mkpts31_1, return_indices=True)
        plot_cycle_matches(img1, img2, img3, mkpts12_1, mkpts12_2, mkpts23_2, mkpts23_3, mkpts31_3, mkpts31_1, nn_ind_2, nn_ind_3, N_show=30)
    else:
        cycled_31_1 = cycle_matches(mkpts12_2, mkpts23_2, mkpts23_3, mkpts31_3, mkpts31_1)
    dists = torch.norm(mkpts12_1 - cycled_31_1, dim=1)
    is_consistent = (dists < tolerance)

    if plot_matches:
        print(f"Number of consistent matches: {int(is_consistent.sum())}")
        print(f"Ratio of consistent matches: {is_consistent.to(torch.float32).mean():.3f}")
        print(f"Least confident of consistent matches: {confidence_12[is_consistent].min():.3f}")
        _ = plot_match_coverage(img1, mkpts12_1[is_consistent], img2, mkpts12_2[is_consistent], confidence_12[is_consistent])
        # _ = plot_match_coverage(img1, cycled_31_1[is_consistent], img1, mkpts12_1[is_consistent], is_consistent[is_consistent])

    return mkpts12_1[is_consistent], mkpts12_2[is_consistent], confidence_12[is_consistent]


