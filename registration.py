# import os
# import cv2
import kornia as K
import numpy as np
import torch
from tqdm import tqdm
from skimage.transform import AffineTransform

# from utils import crop_img, convert_img_tensor_to_numpy, crop_coords_zero_borders, undo_rotation
# from plotting import plot_matches, plot_matches_conf, plot_match_coverage
# from masking import keypoints_roi_to_image, scale_image, mask_leaf, erode_crop_leaf, crop_ROI_erode_leaf, 
from utils import convert_image_to_tensor, match_sizes_resize, match_sizes_resize_batch, invert_list
from masking import fetch_image_mask_pair
from loftr import loftr_match, tps_skimage, tps_skimage_confidence, register_loftr_tps, register_loftr_tps_skimage, warp_tps_skimage, warp_tps_torch, fit_tps_torch, compose_tps, filter_matches_by_confidence, filter_matches_by_min_distance, filter_matches, check_warp_consistency
from plotting import plot_image_pair, plot_overlay, plot_matches_conf, plot_match_coverage
from DatasetTools.LeafImageSeries import LeafDataset

PREPROCESSING_DEFAULT = {'img_scale': 'full', 'pre_rotate': False, 'erase_markers': {'type': 'pixel_erosion'}}
CONSISTENCY_DEFAULT = None # {'consistency_tolerance': 10, 'transform': {'type': 'rotation', 'params': {'rotation': -10}}}
FILTERING_DEFAULT = {'filtering_strategy': 'confidence', 'n_landmarks': 500, 'n_landmarks_tol': 50, 'min_conf': 0.5}
CRITERION_DEFAULT = {'type': 'coverage', 'params': {'dist_threshold': 40}}



def fetch_preregistered_leaf(leaf, ind):
    img = convert_image_to_tensor(leaf.target_images[ind])
    mask = convert_image_to_tensor(leaf.target_masks[ind])
    if (img is None) or (mask is None): 
        print(f"Error: missing data for leaf {leaf.leaf_uid} at index {ind}")
        return None, None
    mask[mask != 0] = 1

    if ind == 0:
        # first image in sequence still has background
        img = img * mask

    return img, mask

def fetch_preregistered_leaf_seq(leaf):
    imgs = [convert_image_to_tensor(leaf.target_images[ind]) for ind in range(leaf.n_leaves)]
    masks = [convert_image_to_tensor(leaf.target_masks[ind]) for ind in range(leaf.n_leaves)]
    for i, mask in enumerate(masks):
        if mask is None:
            print(f"Error: missing data for leaf {leaf.leaf_uid} at index {i}")
            continue
        mask[mask != 0] = 1
        masks[i] = mask


    # first image in sequence still has background
    imgs[0] = imgs[0] * masks[0]

    return imgs, masks


def fetch_registered_image_mask_pair(leaf, fixed_img_ind, moving_img_ind, method, plot_masked_images=False, plot_loftr_matches=False):
    """
    for the given index pair, fetches registered fixed and moving image plus matching masks.

    Args:
        leaf: leaf sequence to retrieve data from
        fixed_img_ind: index of the fixed image
        moving_img_ind: index of the moving image
        method: registration to utilize
            "Piecewise Affine": Jonas' pre-existing method
            "LoFTR + TPS Full": TPS based on LoFTR matches on full leaf
            "LoFTR + TPS Full with Markers": TPS based on LoFTR matches on full leaf, without eroding away markers
            "LoFTR + TPS ROI": TPS based on LoFTR matches only on ROI
            "LoFTR + TPS ROI with Markers": TPS based on LoFTR matches only on ROI, without eroding away markers
            "LoFTR + TPS ROI Pre-Rotated": TPS based on LoFTR matches only on ROI, where ROI is already rotated  to align with the image borders
            "LoFTR + TPS ROI Pre-Rotated with Markers": TPS based on LoFTR matches only on pre-rotated ROI, without eroding away markers
        plot_masked_images: if True, displays images & masks after masking, before registration
        plot_loftr_matches: if True, displays diagnostic images of matches detected by LoFTR

    Returns:
        fixed image
        registered moving image
        mask for fixed image
        registered moving image

    """
    if method == "Piecewise Affine":
        img_fixed, mask_fixed = fetch_preregistered_leaf(leaf, fixed_img_ind)
        img_moving, mask_moving = fetch_preregistered_leaf(leaf, moving_img_ind)
        return img_fixed, img_moving, mask_fixed, mask_moving
        
    else:
        
        if method == "LoFTR + TPS ROI":
            img_fixed, mask_fixed = fetch_image_mask_pair(leaf, fixed_img_ind, img_scale="roi", erase_markers=True, pre_rotate=False)
            img_moving, mask_moving = fetch_image_mask_pair(leaf, moving_img_ind, img_scale="roi", erase_markers=True, pre_rotate=False)
        elif method == "LoFTR + TPS ROI with Markers":
            img_fixed, mask_fixed = fetch_image_mask_pair(leaf, fixed_img_ind, img_scale="roi", erase_markers=False, pre_rotate=False)
            img_moving, mask_moving = fetch_image_mask_pair(leaf, moving_img_ind, img_scale="roi", erase_markers=False, pre_rotate=False)
        elif method == "LoFTR + TPS ROI Pre-Rotated":
            img_fixed, mask_fixed = fetch_image_mask_pair(leaf, fixed_img_ind, img_scale="roi", erase_markers=True, pre_rotate=True)
            img_moving, mask_moving = fetch_image_mask_pair(leaf, moving_img_ind, img_scale="roi", erase_markers=True, pre_rotate=True)
        elif method == "LoFTR + TPS ROI Pre-Rotated with Markers":
            img_fixed, mask_fixed = fetch_image_mask_pair(leaf, fixed_img_ind, img_scale="roi", erase_markers=False, pre_rotate=True)
            img_moving, mask_moving = fetch_image_mask_pair(leaf, moving_img_ind, img_scale="roi", erase_markers=False, pre_rotate=True)
        elif method == "LoFTR + TPS Full":
            img_fixed, mask_fixed = fetch_image_mask_pair(leaf, fixed_img_ind, img_scale="full", erase_markers=True)
            img_moving, mask_moving = fetch_image_mask_pair(leaf, moving_img_ind, img_scale="full", erase_markers=True)
        elif method == "LoFTR + TPS Full with Markers":
            img_fixed, mask_fixed = fetch_image_mask_pair(leaf, fixed_img_ind, img_scale="full", erase_markers=False)
            img_moving, mask_moving = fetch_image_mask_pair(leaf, moving_img_ind, img_scale="full", erase_markers=False)
        else:
            raise ValueError(f'Unknown registration method {method}')

        # resize
        img_fixed, img_moving, mask_fixed, mask_moving = match_sizes_resize(img_fixed, img_moving, mask_fixed, mask_moving)

        if plot_masked_images:
            fig, ax = plot_image_pair(img_fixed, img_moving, fixed_img_ind, moving_img_ind, title="Masked out input images", title_offset=0.7)
            fig.show()
            fig, ax = plot_image_pair(mask_fixed, mask_moving, fixed_img_ind, moving_img_ind, title="corresponding masks", title_offset=0.7)
            fig.show()

        # register
        warped_moving_img, warped_moving_mask = register_loftr_tps(img_fixed, img_moving, mask_moving=mask_moving, verbose=False, plot_loftr_matches=plot_loftr_matches, return_tps=False)
        
        return img_fixed, warped_moving_img, mask_fixed, warped_moving_mask


def register_single_image(
    img_fixed,
    img_moving, 
    mask_fixed: torch.Tensor=None,
    mask_moving: torch.Tensor=None,
    smoothing: float=0.0,     
    return_tps: bool=False,
    plot_loftr_matches: bool=False, 
    warp_consistency: dict=CONSISTENCY_DEFAULT,
    match_filtering: dict=FILTERING_DEFAULT,
    verbose: bool=False,
    ):
    
    """
    uses loftr to detect matches between the fixed and moving image, filters the matches by confidence, then uses TPS to transform the moving image
    if a mask of the moving image is provided, it is also warped.
    optionally the TPS transform can be returned
    """

    # handle missing data cases
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

    mkpts0, mkpts1, confidence, _ = loftr_match(img_fixed, img_moving, mask_fixed, mask_moving, verbose=verbose, return_n_matches=False)

    if warp_consistency is not None: # filter out inconsistent matches
        mkpts0, mkpts1, confidence = check_warp_consistency(img_fixed, img_moving, mask_moving, plot_matches=False, verbose=verbose, **warp_consistency)

    # reduce number of matches
    filtering_mapped = { # rename arguments
        "filtering_strategy": match_filtering["filtering_strategy"],
        "n_target": match_filtering["n_landmarks"],
        "tol": match_filtering["n_landmarks_tol"],
        "min_conf": match_filtering["min_conf"],
    }
    mkpts0_filtered, mkpts1_filtered = filter_matches(mkpts0, mkpts1, confidence, img_fixed.shape, **filtering_mapped)

    if plot_loftr_matches:
        fig, ax = plot_matches_conf(img_fixed, mkpts0, img_moving, mkpts1, confidence, N_show=50, vertical=True)
        fig.show()
        fig, axs = plot_match_coverage(img_fixed, mkpts0, img_moving, mkpts1, confidence)
        fig.show()
    
    if len(mkpts0_filtered) > 3: # ensure there are enough keypts to compute TPS
        
        # fit tps
        if verbose:
            print("Fitting TPS...")
        tps = fit_tps_torch(mkpts0_filtered, mkpts1_filtered, alpha=smoothing)
        
        # warp image
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


def register_leaf_seq(
    leaf: LeafDataset, 
    smoothing: float=0.0, 
    return_masks: bool=True,
    use_skimage: bool=False, 
    image_preprocessing: dict=PREPROCESSING_DEFAULT,
    warp_consistency: dict=CONSISTENCY_DEFAULT,
    match_filtering: dict=FILTERING_DEFAULT,
    verbose: bool=False,    
    ):
    
    # retrieve images
    if verbose:
        print("Fetching leaves...")
    imgs = []
    if return_masks:
        masks = []

    for ind in range(leaf.n_leaves):
        img, mask = img_moving, mask_moving = fetch_image_mask_pair(leaf, ind, **image_preprocessing)
        imgs.append(img)
        if return_masks:
            masks.append(mask)

    # resize
    imgs, masks = match_sizes_resize_batch(imgs, masks)
    
    registered_imgs = [imgs[0]]
    if return_masks:
        registered_masks = [masks[0]]
    
    moving_indices = np.arange(1, leaf.n_leaves)
    for ind in tqdm(moving_indices, "Registering Individually"):
        
        # handle missing data cases
        if imgs[ind] is None:
            print(f"No image data for index {ind}")
            registered_imgs.append(None)
            if return_masks:
                registered_masks.append(None)
            continue

        # register
        if use_skimage:
            img_moving, mask_moving = register_loftr_tps_skimage(imgs[0], imgs[ind], mask_moving=masks[ind], verbose=verbose, plot_loftr_matches=False, return_tps=False)    
        else:
            img_moving, mask_moving = register_single_image(imgs[0], imgs[ind], mask_fixed=masks[0], mask_moving=masks[ind], smoothing=smoothing, return_tps=False, plot_loftr_matches=False, warp_consistency=warp_consistency, match_filtering=match_filtering, verbose=verbose)

        registered_imgs.append(img_moving)
        if return_masks:
            registered_masks.append(mask_moving)

    if return_masks:
        return registered_imgs, registered_masks
    else:
        return registered_imgs


def register_leaf_seq_sequential(
    leaf: LeafDataset, 
    smoothing: float=0.0, 
    return_masks: bool=True,
    use_skimage: bool=False, 
    image_preprocessing: dict=PREPROCESSING_DEFAULT,
    warp_consistency: dict=CONSISTENCY_DEFAULT,
    match_filtering: dict=FILTERING_DEFAULT,
    verbose: bool=False,
    ):
    
    # retrieve images
    imgs = []
    if return_masks:
        masks = []

    if verbose:
        print("Fechting images...")
    for ind in range(leaf.n_leaves):
        img, mask = img_moving, mask_moving = fetch_image_mask_pair(leaf, ind, **image_preprocessing)
        imgs.append(img)
        if return_masks:
            masks.append(mask)

    # resize
    imgs, masks = match_sizes_resize_batch(imgs, masks)
    
    tps = [None]*leaf.n_leaves
    # registered_imgs = [imgs[0]]*leaf.n_leaves
    registered_imgs = [imgs[0]]
    if return_masks:
        registered_masks = [masks[0]]
    moving_indices = np.arange(1, leaf.n_leaves)
    for ind in tqdm(moving_indices, "Registering Sequentially"):
        
        # get TPS transform from current image to previous
        
        if imgs[ind] is None: # if image data is missing, add identity transform to stack
            registered_imgs.append(None)
            if use_skimage:
                tps[ind] = AffineTransform() # identity transform
            else:
                tps[ind] = None
            if return_masks:
                registered_masks.append(None)
            continue

        j = 1
        while imgs[ind-j] is None: # register to latest image that *isn't* missing
            j += 1
        
        if use_skimage:
            tps[ind] = register_loftr_tps_skimage(imgs[ind-j], imgs[ind], threshold=0.5, verbose=verbose, plot_loftr_matches=False, return_tps=True)
            tps_chain = invert_list(tps, ind) # get inverted list of tps transforms
            coord_map = compose_tps(tps_chain)

            # warp images
            registered_imgs.append( warp_tps_skimage(imgs[ind], coord_map, verbose=verbose) )
            if return_masks:
                # converting mask to bool makes warp use nearest-neighbor interpolation
                registered_masks.append( warp_tps_skimage(masks[ind].bool(), coord_map, verbose=verbose) )
        
        else:
            mkpts0, mkpts1, confidence, _ = loftr_match(imgs[ind-j], imgs[ind], masks[ind-j], masks[ind], verbose=verbose, return_n_matches=False)
            
            if warp_consistency is not None: # filter out inconsistent matches
                mkpts0, mkpts1, confidence = check_warp_consistency(imgs[ind-j], imgs[ind], masks[ind], plot_matches=False, verbose=verbose, **warp_consistency)

            # reduce number of matches
            filtering_mapped = { # rename arguments
                "filtering_strategy": match_filtering["filtering_strategy"],
                "n_target": match_filtering["n_landmarks"],
                "tol": match_filtering["n_landmarks_tol"],
                "min_conf": match_filtering["min_conf"],
            }
            mkpts0_filtered, mkpts1_filtered = filter_matches(mkpts0, mkpts1, confidence, imgs[ind].shape, **filtering_mapped)
            
            # fit tps
            tps[ind] = fit_tps_torch(mkpts0_filtered, mkpts1_filtered, alpha=smoothing)

            # warp images
            if verbose:
                print("Warping Moving Image...")
            registered_imgs.append( warp_tps_torch(tps[:ind+1], imgs[ind]) )
            if return_masks:
                registered_masks.append( warp_tps_torch(tps[:ind+1], masks[ind], interpolation_mode='nearest') )

    if return_masks:
        return registered_imgs, registered_masks
    else:
        return registered_imgs


def register_leaf_seq_sequential_filtered(leaf: LeafDataset, smoothing: float=0.5, img_scale: str="full", pre_rotate: bool=False, erase_markers: bool=True, return_masks: bool=True, use_scaling_erosion: bool=False, use_skimage: bool=False, verbose: bool=False):
    
    # retrieve images
    imgs = []
    if return_masks:
        masks = []

    for ind in range(leaf.n_leaves):
        img, mask = img_moving, mask_moving = fetch_image_mask_pair(leaf, ind, img_scale=img_scale, pre_rotate=pre_rotate, erase_markers=erase_markers, use_scaling_erosion=use_scaling_erosion)
        imgs.append(img)
        if return_masks:
            masks.append(mask)

    # resize
    imgs, masks = match_sizes_resize_batch(imgs, masks)
    
    tps = [None]*leaf.n_leaves
    registered_imgs = [imgs[0]]
    if return_masks:
        registered_masks = [masks[0]]
    moving_indices = np.arange(1, leaf.n_leaves)
    for ind in tqdm(moving_indices, "Registering Filtered Sequentially"):

        if imgs[ind] is None:
            registered_imgs.append(None)
            if use_skimage:
                tps[ind] = AffineTransform() # identity transform
            else:
                tps[ind] = None
            if return_masks:
                registered_masks.append(None)
            continue

        j = 1
        while imgs[ind-j] is None:
            j += 1
        
        mkpts0, mkpts1, confidence, _ = loftr_match(imgs[ind-j], imgs[ind], verbose=verbose, return_n_matches=False)
        
        min_dist = 80
        max_points=None
        threshold=0.5
        dist_matches_fix, dist_matches_mov = filter_matches_by_min_distance(mkpts0, mkpts1, confidence, min_dist=min_dist, max_points=max_points, threshold=threshold)

        if use_skimage:
            _, tps_func = tps_skimage(dist_matches_fix, dist_matches_mov, warp_moving=False, verbose=verbose)
            tps[ind] = tps_func
            tps_chain = invert_list(tps, ind) # get inverted list of tps transforms
            coord_map = compose_tps(tps_chain)

            # warp images
            registered_imgs.append( warp_tps_skimage(imgs[ind], coord_map, verbose=verbose) )
            if return_masks:
                # converting mask to bool makes warp use nearest-neighbor interpolation
                registered_masks.append( warp_tps_skimage(masks[ind].bool(), coord_map, verbose=verbose) )

        else:
            tps[ind] = fit_tps_torch(dist_matches_fix, dist_matches_mov, alpha=smoothing)

            # warp images
            registered_imgs.append( warp_tps_torch(tps[:ind+1], imgs[ind]) )
            if return_masks:
                registered_masks.append( warp_tps_torch(tps[:ind+1], masks[ind], interpolation_mode='nearest') )

    if return_masks:
        return registered_imgs, registered_masks
    else:
        return registered_imgs


def conf_matches_amount(confidence, conf_threshold: float=0.5, n_threshold=400):
    # if we have more than n_threshold confident matches, return True
    out =  (torch.sum(confidence > conf_threshold) > n_threshold)
    # print(f"condition: {out}")
    return out


def keypoint_coverage(mask, keypoints, dist_threshold: float=40, quantile: float=0.975):

    H, W = mask.shape[2:]

    # create point image: 1 everywhere, 0 at points
    point_img = torch.zeros((1, 1, H, W), dtype=torch.float32)
    for x, y in keypoints:
        point_img[0, 0, int(y), int(x)] = 1.0

    
    # distance transform
    dist = K.contrib.distance_transform(point_img)#, kernel_size=11)

    # keep only foreground
    dist_fg = dist * mask

    y = torch.linspace(0, 1, H).view(H, 1).expand(H, W)
    x = torch.linspace(0, 1, W).view(1, W).expand(H, W)

    # distance to horizontal center (penalize left/right less)
    x_weight = 1.0 - 0.7 * (2 * torch.abs(x - 0.5))  # strong reduction at edges

    # distance to vertical center (penalize top/bottom slightly less)
    y_weight = 1.0 - 0.3 * (2 * torch.abs(y - 0.5))  # mild reduction

    weight = x_weight * y_weight
    weight = weight.clamp(min=0.1)  # avoid zeroing things out

    weighted_dist = dist_fg * weight

    # max_gap = weighted_dist.max()
    masked_weighted_dist = weighted_dist[mask > 0]
    quantile_val = torch.quantile(masked_weighted_dist, quantile)

    return (quantile_val <= dist_threshold)



def semi_seq_criterion(criterion_type: str="coverage", params: dict={'dist_threshold': 40}, *args, **kwargs):
    if criterion_type == "coverage":
        return keypoint_coverage(*args, **kwargs, **params)
    elif criterion_type == "num_conf_matches":
        return conf_matches_amount(*args, **kwargs, **params)
    else:
        raise ValueError(f"Unknown criterion type '{criterion_type}'. Expected one of 'coverage' or 'num_conf_matches'.")  




def register_leaf_seq_semi_sequential(
    leaf: LeafDataset, 
    smoothing: float=0.0, 
    return_masks: bool=True,
    use_skimage: bool=False, 
    image_preprocessing: dict=PREPROCESSING_DEFAULT,
    warp_consistency: dict=CONSISTENCY_DEFAULT,
    match_filtering: dict=FILTERING_DEFAULT,
    semi_sequential_criterion: dict=CRITERION_DEFAULT,
    verbose: bool=False,
    ):
    
    # retrieve images
    imgs = []
    if return_masks:
        masks = []

    if verbose:
        print("Fechting images...")
    for ind in range(leaf.n_leaves):
        img, mask = fetch_image_mask_pair(leaf, ind, **image_preprocessing)
        imgs.append(img)
        if return_masks:
            masks.append(mask)
            

    # resize
    imgs, masks = match_sizes_resize_batch(imgs, masks)
    
    tps = [None]*leaf.n_leaves
    registered_imgs = [imgs[0]]
    if return_masks:
        registered_masks = [masks[0]]
    moving_indices = np.arange(1, leaf.n_leaves)
    anchor = [0]
    threshold = 0.5
    sanity = [None]*leaf.n_leaves

    for ind in tqdm(moving_indices, "Registering Semi-Sequentially"):

        # skip images with missing data
        if imgs[ind] is None:
            print(f"No image data for index {ind}")
            registered_imgs.append(None)
            if return_masks:
                registered_masks.append(None)
            continue

        if use_skimage:
            mkpts0, mkpts1, confidence, _, n_matches = loftr_match(imgs[anchor[-1]], imgs[ind], masks[anchor[-1]], masks[ind], verbose=verbose, return_n_matches=True)
            # warped_moving_img, warped_moving_mask, tps[ind] = register_loftr_tps(imgs[ind-1], imgs[ind], threshold=0.5, verbose=False, plot_loftr_matches=False, warp_moving=True, return_tps=True)
            
            if warp_consistency is not None: # filter out inconsistent matches
                mkpts0, mkpts1, confidence = check_warp_consistency(imgs[anchor[-1]], imgs[ind], masks[ind], plot_matches=False, verbose=verbose, **warp_consistency)

            if condition(confidence, conf_threshold=0.8):
                # if condition is satisfied, warp moving image

                _, tps[ind] = tps_skimage(mkpts0, mkpts1, confidence, threshold, imgs[ind], warp_moving=False, verbose=verbose)
                
            elif ind != 1: # otherwise, register to a more recent image
                
                # make sure we don't link back to an empty picture
                j = ind-1
                while imgs[j] is None: # look for most recent non-None image
                    j -= 1
                anchor.append(j) # set new anchor

                # register to new anchor
                mkpts0, mkpts1, confidence, _, n_matches = loftr_match(imgs[anchor[-1]], imgs[ind], masks[anchor[-1]], masks[ind], verbose=verbose, return_n_matches=True)
                
                _, tps[ind] = tps_skimage(mkpts0, mkpts1, confidence, threshold, imgs[ind], warp_moving=False, verbose=verbose)
            else:
                print(f"Warning! Only few matches found between first and second image of sequence")
                _, tps[ind] = tps_skimage(mkpts0, mkpts1, confidence, threshold, imgs[ind], warp_moving=False, verbose=verbose)


            # sanity check
            # print(f"----- Index {ind} -----------")
            # sanity[ind] = f"{anchor[-1]}-{ind}"
            # relevant_sanity = [sanity[i] for i in anchor + [ind]]
            # sanity_chain = invert_list(relevant_sanity, -1)
            # print(f" anchors: {anchor}")
            # print(f"full chain: {sanity}")
            # print(f"sliced chain: {relevant_sanity}")
            # print(f"inverted chain: {sanity_chain}")

            
            relevant_tps = [tps[i] for i in anchor + [ind]] # pick out transforms for relevant steps
            tps_chain = invert_list(relevant_tps, -1) # invert the list
            # coord_map = compose_tps(tps_chain) # compose the transforms
            # print(len(tps_chain))
            if len(tps_chain) > 1:
                coord_map = compose_tps(tps_chain) # compose the transforms
            else:
                coord_map = tps_chain[0]
            

            # warp images
            registered_imgs.append( warp_tps_skimage(imgs[ind], coord_map, verbose=verbose) )
            if return_masks:
                # converting mask to bool makes warp use nearest-neighbor interpolation
                registered_masks.append( warp_tps_skimage(masks[ind].bool(), coord_map, verbose=verbose) )

        else:
            mkpts0, mkpts1, confidence, _, n_matches = loftr_match(imgs[anchor[-1]], imgs[ind], masks[anchor[-1]], masks[ind], verbose=verbose, return_n_matches=True)
                        
            if warp_consistency is not None: # filter out inconsistent matches
                mkpts0, mkpts1, confidence = check_warp_consistency(imgs[anchor[-1]], imgs[ind], masks[ind], plot_matches=False, verbose=verbose, **warp_consistency)

            # reduce number of matches
            filtering_mapped = { # rename arguments
                "filtering_strategy": match_filtering["filtering_strategy"],
                "n_target": match_filtering["n_landmarks"],
                "tol": match_filtering["n_landmarks_tol"],
                "min_conf": match_filtering["min_conf"],
            }
            mkpts0_filtered, mkpts1_filtered = filter_matches(mkpts0, mkpts1, confidence, imgs[ind].shape, **filtering_mapped)
            
            # print(semi_sequential_criterion)
            # evaluate quality criterion 
            if semi_seq_criterion(mask=masks[ind], keypoints=mkpts1, **semi_sequential_criterion):
            # if semi_seq_criterion(confidence=confidence, **semi_sequential_criterion):
                
                # if condition is satisfied, warp moving image
                tps[ind] = fit_tps_torch(mkpts0_filtered, mkpts1_filtered, alpha=smoothing)
                
            elif ind != 1: # otherwise, register to a more recent image
                
                # make sure we don't link back to an empty picture
                j = ind-1
                while imgs[j] is None: # look for most recent non-None image
                    j -= 1
                anchor.append(j) # set new anchor

                # register to new anchor
                mkpts0, mkpts1, confidence, _ = loftr_match(imgs[anchor[-1]], imgs[ind], masks[anchor[-1]], masks[ind], verbose=verbose, return_n_matches=False)

                if warp_consistency is not None: # filter out inconsistent matches
                    mkpts0, mkpts1, confidence = check_warp_consistency(imgs[anchor[-1]], imgs[ind], masks[ind], plot_matches=False, verbose=verbose, **warp_consistency)

                # reduce number of matches
                mkpts0_filtered, mkpts1_filtered = filter_matches(mkpts0, mkpts1, confidence, imgs[ind].shape, **filtering_mapped)
                
                tps[ind] = fit_tps_torch(mkpts0_filtered, mkpts1_filtered, alpha=smoothing)
            else:
                print(f"Warning! Only few matches found between first and second image of sequence")
                tps[ind] = fit_tps_torch(mkpts0_filtered, mkpts1_filtered, alpha=smoothing)


            # sanity check
            # print(f"----- Index {ind} -----------")
            # sanity[ind] = f"{anchor[-1]}-{ind}"
            # relevant_sanity = [sanity[i] for i in anchor + [ind]]
            # sanity_chain = invert_list(relevant_sanity, -1)
            # print(f" anchors: {anchor}")
            # print(f"full chain: {sanity}")
            # print(f"sliced chain: {relevant_sanity}")
            # print(f"inverted chain: {sanity_chain}")

            
            relevant_tps = [tps[i] for i in anchor + [ind]] # pick out transforms for relevant steps          

            # warp images
            if verbose:
                print("Warping Moving Image...")
            registered_imgs.append( warp_tps_torch(relevant_tps[:ind+1], imgs[ind]) )
            if return_masks:
                registered_masks.append( warp_tps_torch(relevant_tps[:ind+1], masks[ind], interpolation_mode='nearest') )
    
    if return_masks:
        return registered_imgs, registered_masks
    else:
        return registered_imgs


def fetch_registered_image_mask_seq(leaf, registration_method, config):#, plot_masked_images=False, plot_loftr_matches=False):
    """
    for the given index pair, fetches registered fixed and moving image plus matching masks.

    Args:
        leaf: leaf sequence to retrieve data from
        fixed_img_ind: index of the fixed image
        moving_img_ind: index of the moving image
        method: registration to utilize
            "Piecewise Affine": Jonas' pre-existing method
            "Full Leaf": TPS based on LoFTR matches on full leaf
            "Full Leaf with Markers": TPS based on LoFTR matches on full leaf, without eroding away markers
            "Leaf ROI": TPS based on LoFTR matches only on ROI
            "Leaf ROI with Markers": TPS based on LoFTR matches only on ROI, without eroding away markers
            "Leaf ROI Pre-Rotated": TPS based on LoFTR matches only on ROI, where ROI is already rotated  to align with the image borders
            "Leaf ROI Pre-Rotated with Markers": TPS based on LoFTR matches only on pre-rotated ROI, without eroding away markers
        plot_masked_images: if True, displays images & masks after masking, before registration
        plot_loftr_matches: if True, displays diagnostic images of matches detected by LoFTR

    Returns:
        List of registered images
        List of corresponding registered masks

    """
    if registration_method == "Piecewise Affine":
        imgs, masks = fetch_preregistered_leaf_seq(leaf)
        return imgs, masks
        
    else:
        
        # if leaf_style == "Leaf ROI":
        #     leaf_kwargs = {"img_scale": "roi", "erase_markers": True, "pre_rotate": False}
        # elif leaf_style == "Leaf ROI with Markers":
        #     leaf_kwargs = {"img_scale": "roi", "erase_markers": False, "pre_rotate": False}
        # elif leaf_style == "Leaf ROI Pre-Rotated":
        #     leaf_kwargs = {"img_scale": "roi", "erase_markers": True, "pre_rotate": True}
        # elif leaf_style == "Leaf ROI Pre-Rotated with Markers":
        #     leaf_kwargs = {"img_scale": "roi", "erase_markers": False, "pre_rotate": True}
        # elif leaf_style == "Full Leaf":
        #     leaf_kwargs = {"img_scale": "false", "erase_markers": True}
        # elif leaf_style == "Full Leaf with Markers":
        #     leaf_kwargs = {"img_scale": "false", "erase_markers": False}
        # else:
        #     raise ValueError(f'Unknown leaf style {leaf_style}')

        # register
        if registration_method == "LoFTR + TPS Individual":
            if 'semi_sequential_criterion' in config:
                config.pop('semi_sequential_criterion')
            imgs, masks = register_leaf_seq(leaf, **config)
        elif registration_method == "LoFTR + TPS Semi-Sequential":
            imgs, masks = register_leaf_seq_semi_sequential(leaf, **config)
        elif registration_method == "LoFTR + TPS Sequential":
            if 'semi_sequential_criterion' in config:
                config.pop('semi_sequential_criterion')
            imgs, masks = register_leaf_seq_sequential(leaf, **config)
        # elif registration_method == "LoFTR + TPS Filtered Sequential":
        #     imgs, masks = register_leaf_seq_sequential_filtered(leaf, smoothing=smoothing, **leaf_kwargs)
        else:
            raise ValueError(f'Unknown registration method {registration_method}')
        
        return imgs, masks
