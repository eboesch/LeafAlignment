# import os
# import cv2
import kornia as K
import numpy as np
import torch
from tqdm import tqdm
# from utils import crop_img, convert_img_tensor_to_numpy, crop_coords_zero_borders, undo_rotation
# from plotting import plot_matches, plot_matches_conf, plot_match_coverage
# from masking import keypoints_roi_to_image, scale_image, mask_leaf, erode_crop_leaf, crop_ROI_erode_leaf, 
from utils import convert_image_to_tensor, match_sizes_resize, match_sizes_resize_batch, invert_list
from masking import fetch_image_mask_pair
from loftr import loftr_match, tps_skimage, register_loftr_tps, warp_tps, compose_tps
from plotting import plot_image_pair, plot_overlay
from DatasetTools.LeafImageSeries import LeafDataset


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

def register_leaf_seq(leaf: LeafDataset, img_scale: str="full", pre_rotate: bool=False, erase_markers: bool=True, return_masks: bool=True, use_scaling_erosion: bool=False, verbose: bool=False):
    # img_fixed_og, mask_fixed_og = fetch_image_mask_pair(leaf, 0, img_scale=img_scale, pre_rotate=pre_rotate, erase_markers=erase_markers, use_scaling_erosion=use_scaling_erosion)

    # retrieve images
    if verbose:
        print("Fetching leaves...")
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
    
    registered_imgs = [imgs[0]]
    if return_masks:
        registered_masks = [masks[0]]
    
    moving_indices = np.arange(1, leaf.n_leaves)
    for ind in tqdm(moving_indices, "Registering Individually"):
        # img_moving, mask_moving = fetch_image_mask_pair(leaf, ind, img_scale=img_scale, pre_rotate=pre_rotate, erase_markers=erase_markers, use_scaling_erosion=use_scaling_erosion)

        # resize
        # img_fixed, img_moving, mask_fixed, mask_moving = match_sizes_resize(img_fixed_og, img_moving, mask_fixed_og, mask_moving)

        if imgs[ind] is None:
            print(f"No image data for index {ind}")
            registered_imgs.append(None)
            if return_masks:
                registered_masks.append(None)
            continue

        # register
        img_moving, mask_moving = register_loftr_tps(imgs[0], imgs[ind], mask_moving=masks[ind], verbose=verbose, plot_loftr_matches=False, return_tps=False)

        registered_imgs.append(img_moving)
        if return_masks:
            registered_masks.append(mask_moving)

    if return_masks:
        return registered_imgs, registered_masks
    else:
        return registered_imgs


def register_leaf_seq_sequential(leaf: LeafDataset, img_scale: str="full", pre_rotate: bool=False, erase_markers: bool=True, return_masks: bool=True, use_scaling_erosion: bool=False):
    
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
    # registered_imgs = [imgs[0]]*leaf.n_leaves
    registered_imgs = [imgs[0]]
    if return_masks:
        registered_masks = [masks[0]]
    moving_indices = np.arange(1, leaf.n_leaves)
    for ind in tqdm(moving_indices, "Registering Sequentially"):

        # mkpts0, mkpts1, confidence, _, n_matches = loftr_match(img_fixed, img_moving, verbose=verbose, return_n_matches=True)
        
        # if n_matches['conf_matches'] > 3:
        #     tps = tps_skimage(mkpts0, mkpts1, confidence, threshold, img_moving, warp_moving=False, verbose=verbose)
        # else:
        #     print("No enough matches for TPS found")
        #     tps = None
        
        # get TPS transform from current image to previous
        tps[ind] = register_loftr_tps(imgs[ind-1], imgs[ind], threshold=0.5, verbose=False, plot_loftr_matches=False, warp_moving=False, return_tps=True)
        
        # if ind > 1:
        #     tps_chain = [tps[ind], tps[ind-1]]
        #     coord_map = compose_tps(tps_chain)
        # else:
        #     coord_map = tps[ind]

        # TODO: what if None?
        tps_chain = invert_list(tps, ind) # get inverted list of tps transforms
        coord_map = compose_tps(tps_chain)

        # warp images
        registered_imgs.append( warp_tps(imgs[ind], coord_map, verbose=True) )
        if return_masks:
            # converting mask to bool makes warp use nearest-neighbor interpolation
            registered_masks.append( warp_tps(masks[ind].bool(), coord_map, verbose=False) )

    if return_masks:
        return registered_imgs, registered_masks
    else:
        return registered_imgs


def stop_condition(confidence, conf_threshold: float=0.5, n_threshold=400):
    # if we have more than 200 confident matches, return True
    out =  (torch.sum(confidence > conf_threshold) > n_threshold)
    # print(f"condition: {out}")
    return out

def register_leaf_seq_semi_sequential(leaf: LeafDataset, condition=stop_condition, img_scale: str="full", pre_rotate: bool=False, erase_markers: bool=True, return_masks: bool=True, use_scaling_erosion: bool=False, verbose=False):
    
    # retrieve images
    imgs = []
    if return_masks:
        masks = []

    if verbose:
        print("Fechting images...")
    for ind in range(leaf.n_leaves):
        img, mask = fetch_image_mask_pair(leaf, ind, img_scale=img_scale, pre_rotate=pre_rotate, erase_markers=erase_markers, use_scaling_erosion=use_scaling_erosion)
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

        if imgs[ind] is None:
            print(f"No image data for index {ind}")
            registered_imgs.append(None)
            if return_masks:
                registered_masks.append(None)
            continue

        mkpts0, mkpts1, confidence, _, n_matches = loftr_match(imgs[anchor[-1]], imgs[ind], verbose=verbose, return_n_matches=True)
        # warped_moving_img, warped_moving_mask, tps[ind] = register_loftr_tps(imgs[ind-1], imgs[ind], threshold=0.5, verbose=False, plot_loftr_matches=False, warp_moving=True, return_tps=True)
        
        if condition(confidence, conf_threshold=0.8):
            # if condition is satisfied, warp moving image

            _, tps[ind] = tps_skimage(mkpts0, mkpts1, confidence, threshold, imgs[ind], warp_moving=False, verbose=verbose)
            
        elif ind != 1:
            # make sure we don't link back to an empty picture
            j = ind-1
            while imgs[j] is None:
                j -= 1
            anchor.append(j)
            mkpts0, mkpts1, confidence, _, n_matches = loftr_match(imgs[anchor[-1]], imgs[ind], verbose=verbose, return_n_matches=True)
            
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
        registered_imgs.append( warp_tps(imgs[ind], coord_map, verbose=verbose) )
        if return_masks:
            # converting mask to bool makes warp use nearest-neighbor interpolation
            registered_masks.append( warp_tps(masks[ind].bool(), coord_map, verbose=verbose) )

    
    if return_masks:
        return registered_imgs, registered_masks
    else:
        return registered_imgs


def fetch_registered_image_mask_seq(leaf, registration_method, leaf_style, plot_masked_images=False, plot_loftr_matches=False):
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
        
        if leaf_style == "Leaf ROI":
            leaf_kwargs = {"img_scale": "roi", "erase_markers": True, "pre_rotate": False}
        elif leaf_style == "Leaf ROI with Markers":
            leaf_kwargs = {"img_scale": "roi", "erase_markers": False, "pre_rotate": False}
        elif leaf_style == "Leaf ROI Pre-Rotated":
            leaf_kwargs = {"img_scale": "roi", "erase_markers": True, "pre_rotate": True}
        elif leaf_style == "Leaf ROI Pre-Rotated with Markers":
            leaf_kwargs = {"img_scale": "roi", "erase_markers": False, "pre_rotate": True}
        elif leaf_style == "Full Leaf":
            leaf_kwargs = {"img_scale": "false", "erase_markers": True}
        elif leaf_style == "Full Leaf with Markers":
            leaf_kwargs = {"img_scale": "false", "erase_markers": False}
        else:
            raise ValueError(f'Unknown leaf style {leaf_style}')

        # register
        if registration_method == "LoFTR + TPS Individual":
            imgs, masks = register_leaf_seq(leaf, **leaf_kwargs)
        elif registration_method == "LoFTR + TPS Semi-Sequential":
            imgs, masks = register_leaf_seq_semi_sequential(leaf, **leaf_kwargs)
        elif registration_method == "LoFTR + TPS Sequential":
            imgs, masks = register_leaf_seq_sequential(leaf, **leaf_kwargs)
        else:
            raise ValueError(f'Unknown registration method {registration_method}')
        
        return imgs, masks
