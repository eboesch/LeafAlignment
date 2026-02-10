import os
import cv2
import kornia as K
import kornia.feature as KF
import numpy as np
import torch
import skimage as ski
from utils import convert_image_to_tensor
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
        Confidence of matches
        Classification of inliers, using RANSAC/Fundamental matrix
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


def tps_skimage(keypts_fix, keypts_mov, confidence, thrsld, img_mov, verbose=False):
    """
    Applies TPS to register moving image to fixed image. Keypoints are filtered by confidence.

    Returns: transformed moving image and transform function
    """

    # kornia and torch expect C x H x W, while skimage expects H x W x C
    img_mov_reordered = K.tensor_to_image(img_mov)

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

    # tps = ski.transform.ThinPlateSplineTransform()
    if verbose:
        print("Estimating TPS transform...")
    # tps.estimate(img_fix_mks, img_mov_mks) # estimate transform from img_fix -> img_mov
    tps = ski.transform.ThinPlateSplineTransform.from_estimate(img_fix_mks, img_mov_mks)
    if verbose:
        print("Transforming moving image...")
    warped = ski.transform.warp(img_mov_reordered, tps) # warp uses inverse transform, i.e. img_mov -> img_fix

    return warped, tps

def warp_tps(img, tps, verbose=False):
    # kornia and torch expect C x H x W, while skimage expects H x W x C
    img = K.tensor_to_image(img)

    if verbose:
        print("Transforming moving image...")
    warped = ski.transform.warp(img, tps) # warp uses inverse transform, i.e. img_mov -> img_fix

    return warped


def register_loftr_tps(img_fixed, img_moving, threshold=0.5, mask_moving: torch.Tensor=None, verbose=False, plot_loftr_matches=False, return_tps=False):
    mkpts0, mkpts1, confidence, _, n_matches = loftr_match(img_fixed, img_moving, verbose=verbose, return_n_matches=True)

    if plot_loftr_matches:
        fig, ax = plot_matches_conf(img_fixed, mkpts0, img_moving, mkpts1, confidence, N_show=50, vertical=True)
        fig.show()
        fig, axs = plot_match_coverage(img_fixed, mkpts0, img_moving, mkpts1, confidence)
        fig.show()
    
    if n_matches['conf_matches'] > 3:
        warped_moving_img, tps = tps_skimage(mkpts0, mkpts1, confidence, threshold, img_moving, verbose=verbose)
        warped_moving_img = convert_image_to_tensor(warped_moving_img)
        if mask_moving is not None:
            # warped_moving_mask, tps = tps_skimage(mkpts0, mkpts1, confidence, threshold, mask_moving, verbose=False)
            warped_moving_mask = warp_tps(mask_moving, tps, verbose)
            warped_moving_mask = convert_image_to_tensor(warped_moving_mask)
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



