import kornia as K
import numpy as np
import torch
from tqdm import tqdm
from skimage.transform import AffineTransform

from utils import convert_image_to_tensor, match_sizes_resize, match_sizes_resize_batch, invert_list, affine_warp_expand, check_orientation, RandomHomography
from masking import fetch_image_mask_pair, fetch_masked_image_seq
from loftr import loftr_match, warp_tps_torch, fit_tps_torch, fetch_keypoints
from loftr import loftr_match, tps_skimage_confidence, register_loftr_tps_skimage, warp_tps_skimage, compose_tps,
from plotting import plot_image_pair, plot_overlay, plot_matches_conf, plot_match_coverage
from DatasetTools.LeafImageSeries import LeafDataset

PREPROCESSING_DEFAULT = {'img_scale': 'full', 'pre_rotate': False, 'erase_markers': {'type': 'pixel_erosion', 'params': {}}}
CONSISTENCY_DEFAULT = None # {'consistency_tolerance': 10, 'transform': {'type': 'rotation', 'params': {'rotation': -10}}}
FILTERING_DEFAULT = {'filtering_strategy': 'confidence', 'n_landmarks': 500, 'n_landmarks_tol': 50, 'min_conf': 0.5}
CRITERION_DEFAULT = {'criterion_type': 'coverage', 'params': {'dist_threshold': 40}}



def fetch_preregistered_leaf(leaf: LeafDataset, ind: int):
    """
    Fetches ROI registered with Piecewise Affine registration at a specified time point, and the corresponding mask.

    Args:
        leaf: leaf data of the desired leaf
        ind: Index (i.e. point in time series) of the specific image to be fetch.

    Returns:
        torch.Tensor: image of ROI registered with Piecewise Affine registration
        torch.Tensor: corresponding mask
    """
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

def fetch_preregistered_leaf_seq(leaf: LeafDataset):
    """
    Fetches ROI registered with Piecewise Affine registration at all time points, and the corresponding masks.

    Args:
        leaf: leaf data of the desired leaf

    Returns:
        List[torch.Tensor]: image of ROI registered with Piecewise Affine registration
        List[torch.Tensor]: corresponding mask
    """
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


def register_single_image(
    img_fixed: torch.Tensor,
    img_moving: torch.Tensor, 
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
    Registers moving image to fixed image.
    Uses LoFTR to detect matches between the fixed and moving image, filters by warp consistency, reduces number of matches as specified by match_filtering, then uses TPS to transform the moving image
    If a mask of the moving image is provided, it is also warped.
    Optionally, the TPS transform can be returned.

    Args:
        img_fixed: fixed image
        img_moving: moving image
        mask_fixed: Optional mask for fixed image
        mask_moving: Optional mask for moving image
        smoothing: smoothing hyperparameter. higher values lead to more "rigid" transforms
        return_tps: whether to return the fitted TPS object
        plot_loftr_matches: whether to plot figures showing distribution of matches 
        warp_consistency: dictionary specifying parameters for warp consistency. to disable warp consistency, set it to None.
        match_filtering: dictionary specifying parameters for match filtering/subsampling, such as filtering strategy, target number of landmarks, and minimum confidence threshold.
        verbose: Whether to produce detailed output (for diagnostic purposes)

    Returns:
        torch.Tensor: registered moving image
        torch.Tensor: mask for registered moving image. Only returned if a mask for the original moving image is provided
        (ThinPlateSpline: fitted TPS object. only returned if return_tps==True)
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

    # fetch keypoints
    out = fetch_keypoints(
        img_fixed,
        img_moving, 
        mask_fixed,
        mask_moving,
        warp_consistency,
        match_filtering,
        verbose=verbose
    )
    if 'rotated_moving_img' in out:
        # print("Rotation check")
        img_moving = out['rotated_moving_img']
        mask_moving = out['rotated_moving_mask']

    if plot_loftr_matches:
        mkpts0 = out['mkpts0']
        mkpts1 = out['mkpts1']
        confidence = out['confidence']
        fig, ax = plot_matches_conf(img_fixed, mkpts0, img_moving, mkpts1, confidence, N_show=50, vertical=True)
        fig.show()
        fig, axs = plot_match_coverage(img_fixed, mkpts0, img_moving, mkpts1, confidence)
        fig.show()
    
    mkpts0_filtered = out['mkpts0_filtered']
    mkpts1_filtered = out['mkpts1_filtered']

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
        print("Not enough matches for TPS found")
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

def fetch_registered_image_mask_pair(
    leaf: LeafDataset, 
    fixed_img_ind: int, 
    moving_img_ind: int, 
    method: str, 
    smoothing: float=0.0, 
    warp_consistency: dict=CONSISTENCY_DEFAULT, 
    match_filtering: dict=FILTERING_DEFAULT, 
    plot_masked_images: bool=False, 
    plot_loftr_matches: bool=False,
    verbose: bool=False, 
    ):
    """
    For the given index pair, fetches registered fixed and moving image plus matching masks.

    Args:
        leaf: leaf sequence to retrieve data from
        fixed_img_ind: index of the fixed image
        moving_img_ind: index of the moving image
        method: registration to utilize
            "Piecewise Affine": Jonas' pre-existing method
            "LoFTR + TPS Full": TPS based on LoFTR matches on full leaf
            "LoFTR + TPS Full with Markers": TPS based on LoFTR matches on full leaf, without eroding away markers
            "LoFTR + TPS Full Pre-Rotated": TPS based on LoFTR matches on full leaf, where the leaf is pre-rotated to align with the image borders
            "LoFTR + TPS Full Pre-Rotated with Markers": TPS based on LoFTR matches on full leaf, without eroding away markers,  where the leaf is pre-rotated to align with the image borders
            "LoFTR + TPS ROI": TPS based on LoFTR matches only on ROI
            "LoFTR + TPS ROI with Markers": TPS based on LoFTR matches only on ROI, without eroding away markers
            "LoFTR + TPS ROI Pre-Rotated": TPS based on LoFTR matches only on ROI, where ROI is already rotated  to align with the image borders
            "LoFTR + TPS ROI Pre-Rotated with Markers": TPS based on LoFTR matches only on pre-rotated ROI, without eroding away markers
        smoothing: smoothing hyperparameter. higher values lead to more "rigid" transforms
        warp_consistency: dictionary specifying parameters for warp consistency. to disable warp consistency, set it to None.
        match_filtering: dictionary specifying parameters for match filtering/subsampling, such as filtering strategy, target number of landmarks, and minimum confidence threshold.
        plot_masked_images: if True, displays images & masks after masking, before registration
        plot_loftr_matches: if True, displays diagnostic images of matches detected by LoFTR
        verbose: Whether to produce detailed output (for diagnostic purposes)

    Returns:
        torch.Tensor: fixed image
        torch.Tensor: registered moving image
        torch.Tensor: mask for fixed image
        torch.Tensor: mask for registered moving image

    """
    if method == "Piecewise Affine":
        img_fixed, mask_fixed = fetch_preregistered_leaf(leaf, fixed_img_ind)
        img_moving, mask_moving = fetch_preregistered_leaf(leaf, moving_img_ind)
        return img_fixed, img_moving, mask_fixed, mask_moving
        
    else:
        # fetch images

        erosion = {"type": "pixel_erosion", 'params': {}}
        no_erosion = None

        if method == "LoFTR + TPS ROI":
            img_fixed, mask_fixed = fetch_image_mask_pair(leaf, fixed_img_ind, img_scale="roi", erase_markers=erosion, pre_rotate=False)
            img_moving, mask_moving = fetch_image_mask_pair(leaf, moving_img_ind, img_scale="roi", erase_markers=erosion, pre_rotate=False)
        elif method == "LoFTR + TPS ROI with Markers":
            img_fixed, mask_fixed = fetch_image_mask_pair(leaf, fixed_img_ind, img_scale="roi", erase_markers=no_erosion, pre_rotate=False)
            img_moving, mask_moving = fetch_image_mask_pair(leaf, moving_img_ind, img_scale="roi", erase_markers=no_erosion, pre_rotate=False)
        elif method == "LoFTR + TPS ROI Pre-Rotated":
            img_fixed, mask_fixed = fetch_image_mask_pair(leaf, fixed_img_ind, img_scale="roi", erase_markers=erosion, pre_rotate=True)
            img_moving, mask_moving = fetch_image_mask_pair(leaf, moving_img_ind, img_scale="roi", erase_markers=erosion, pre_rotate=True)
        elif method == "LoFTR + TPS ROI Pre-Rotated with Markers":
            img_fixed, mask_fixed = fetch_image_mask_pair(leaf, fixed_img_ind, img_scale="roi", erase_markers=no_erosion, pre_rotate=True)
            img_moving, mask_moving = fetch_image_mask_pair(leaf, moving_img_ind, img_scale="roi", erase_markers=no_erosion, pre_rotate=True)
        elif method == "LoFTR + TPS Full":
            img_fixed, mask_fixed = fetch_image_mask_pair(leaf, fixed_img_ind, img_scale="full", erase_markers=erosion, pre_rotate=False)
            img_moving, mask_moving = fetch_image_mask_pair(leaf, moving_img_ind, img_scale="full", erase_markers=erosion, pre_rotate=False)
        elif method == "LoFTR + TPS Full with Markers":
            img_fixed, mask_fixed = fetch_image_mask_pair(leaf, fixed_img_ind, img_scale="full", erase_markers=no_erosion, pre_rotate=False)
            img_moving, mask_moving = fetch_image_mask_pair(leaf, moving_img_ind, img_scale="full", erase_markers=no_erosion, pre_rotate=False)
        elif method == "LoFTR + TPS Full Pre-Rotated":
            img_fixed, mask_fixed = fetch_image_mask_pair(leaf, fixed_img_ind, img_scale="full", erase_markers=erosion, pre_rotate=True)
            img_moving, mask_moving = fetch_image_mask_pair(leaf, moving_img_ind, img_scale="full", erase_markers=erosion, pre_rotate=True)
        elif method == "LoFTR + TPS Full Pre-Rotated with Markers":
            img_fixed, mask_fixed = fetch_image_mask_pair(leaf, fixed_img_ind, img_scale="full", erase_markers=no_erosion, pre_rotate=True)
            img_moving, mask_moving = fetch_image_mask_pair(leaf, moving_img_ind, img_scale="full", erase_markers=no_erosion, pre_rotate=True)
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
        warped_moving_img, warped_moving_mask = register_single_image(img_fixed, img_moving, mask_fixed=mask_fixed, mask_moving=mask_moving, smoothing=smoothing, return_tps=False, plot_loftr_matches=False, warp_consistency=warp_consistency, match_filtering=match_filtering, verbose=verbose)
        
        return img_fixed, warped_moving_img, mask_fixed, warped_moving_mask


def register_leaf_seq_individual(
    leaf: LeafDataset, 
    smoothing: float=0.0, 
    return_masks: bool=True,
    image_preprocessing: dict=PREPROCESSING_DEFAULT,
    warp_consistency: dict=CONSISTENCY_DEFAULT,
    match_filtering: dict=FILTERING_DEFAULT,
    verbose: bool=False,    
    ):
    """
    For the given leaf, registers all leaves using individual registration.

    Args:
        leaf: leaf sequence to register
        smoothing: smoothing hyperparameter. higher values lead to more "rigid" transforms
        return_masks: whether to return masks of registered images
        image_preprocessing: dictionary specifying parameters for image preprocessing, such as image scale, whether to pre-rotate, and parameters of marker erosion
        warp_consistency: dictionary specifying parameters for warp consistency. to disable warp consistency, set it to None.
        match_filtering: dictionary specifying parameters for match filtering/subsampling, such as filtering strategy, target number of landmarks, and minimum confidence threshold.
        verbose: Whether to produce detailed output (for diagnostic purposes)

    Returns:
        List[torch.Tensor]: list of registered images
        (List[torch.Tensor]: list of masks for registered images. Only returned if return_masks==True.)
    """
    
    # retrieve images
    if verbose:
        print("Fetching leaves...")
    imgs = []
    if return_masks:
        masks = []

    for ind in range(leaf.n_leaves):
        img, mask = fetch_image_mask_pair(leaf, ind, **image_preprocessing)
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
    image_preprocessing: dict=PREPROCESSING_DEFAULT,
    warp_consistency: dict=CONSISTENCY_DEFAULT,
    match_filtering: dict=FILTERING_DEFAULT,
    verbose: bool=False,
    ):
    """
    For the given leaf, registers all leaves using sequential registration.

    Args:
        leaf: leaf sequence to register
        smoothing: smoothing hyperparameter. higher values lead to more "rigid" transforms
        return_masks: whether to return masks of registered images
        image_preprocessing: dictionary specifying parameters for image preprocessing, such as image scale, whether to pre-rotate, and parameters of marker erosion
        warp_consistency: dictionary specifying parameters for warp consistency. to disable warp consistency, set it to None.
        match_filtering: dictionary specifying parameters for match filtering/subsampling, such as filtering strategy, target number of landmarks, and minimum confidence threshold.
        verbose: Whether to produce detailed output (for diagnostic purposes)

    Returns:
        List[torch.Tensor]: list of registered images
        (List[torch.Tensor]: list of masks for registered images. Only returned if return_masks==True.)
    """
    
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
    for ind in tqdm(moving_indices, "Registering Sequentially"):
        
        # get TPS transform from current image to previous
        
        if imgs[ind] is None: # if image data is missing, add identity transform to stack
            registered_imgs.append(None)
            tps[ind] = None
            if return_masks:
                registered_masks.append(None)
            continue

        j = 1
        while imgs[ind-j] is None: # register to latest image that *isn't* missing
            j += 1
        

        out = fetch_keypoints(
            imgs[ind-j],
            imgs[ind], 
            masks[ind-j],
            masks[ind],
            warp_consistency,
            match_filtering,
            verbose=verbose
        )
        if 'rotated_moving_img' in out:
            # print("Rotation check")
            imgs[ind] = out['rotated_moving_img']
            masks[ind] = out['rotated_moving_mask']
        

        mkpts0_filtered = out['mkpts0_filtered']
        mkpts1_filtered = out['mkpts1_filtered']

        if len(mkpts0_filtered) > 3: # ensure there are enough keypts to compute TPS
            # fit tps
            tps[ind] = fit_tps_torch(mkpts0_filtered, mkpts1_filtered, alpha=smoothing)

            # warp images
            if verbose:
                print("Warping Moving Image...")
            registered_imgs.append( warp_tps_torch(tps[:ind+1], imgs[ind]) )
            if return_masks:
                registered_masks.append( warp_tps_torch(tps[:ind+1], masks[ind], interpolation_mode='nearest') )
        else:
            print(f"Not enough matches for TPS found at index {ind}")
            imgs[ind] = None # ensures that subsequent images skip this one
            masks[ind] = None
            registered_imgs.append(None)
            tps[ind] = None
            if return_masks:
                registered_masks.append(None)

    if return_masks:
        return registered_imgs, registered_masks
    else:
        return registered_imgs


def conf_matches_amount(confidence: torch.Tensor, conf_threshold: float=0.5, n_threshold: int=400):
    """Check whether more than n_threshold matches have confidence greater than conf_threshold"""
    # if we have more than n_threshold confident matches, return True
    out =  (torch.sum(confidence > conf_threshold) > n_threshold)
    # print(f"condition: {out}")
    return out

def keypoint_coverage(mask: torch.Tensor, keypoints: torch.Tensor, dist_threshold: float=40, quantile: float=0.975):
    """
    Investigates how well the keypoints cover the region of interest, by computing distances to the nearest keypoint for all pixels.

    Args:
        mask: mask of Region of Interest
        keypoints: list of keypoints
        dist_threshold: distance threshold value. if the specified quantile is above this value, we consider the keypoints to have poor coverage 
        quantile: percentile to use

    Returns:
        bool: Indicates whether the keypoints have good coverage. True if the distance quantile is <= dist_threshold
    """

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
    """
    Wrapper function for resetting criterion for semi-sequential registration.
    
    Args: 
        criterion_type: Which criterion to use. 'coverage' or 'num_conf_matches'.
        params: parameters for criterion
        *args, **kwargs: for arguments needed for the selected criterion, e.g. mask, keypoints, or confidence.

    Returns:
        bool: True if registration is considered successful (i.e. if criterion is satisfied).
    """
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
    image_preprocessing: dict=PREPROCESSING_DEFAULT,
    warp_consistency: dict=CONSISTENCY_DEFAULT,
    match_filtering: dict=FILTERING_DEFAULT,
    semi_sequential_criterion: dict=CRITERION_DEFAULT,
    verbose: bool=False,
    ):
    """
    For the given leaf, registers all leaves using individual registration.

    Args:
        leaf: leaf sequence to register
        smoothing: smoothing hyperparameter. higher values lead to more "rigid" transforms
        return_masks: whether to return masks of registered images
        image_preprocessing: dictionary specifying parameters for image preprocessing, such as image scale, whether to pre-rotate, and parameters of marker erosion
        warp_consistency: dictionary specifying parameters for warp consistency. to disable warp consistency, set it to None.
        match_filtering: dictionary specifying parameters for match filtering/subsampling, such as filtering strategy, target number of landmarks, and minimum confidence threshold.
        verbose: Whether to produce detailed output (for diagnostic purposes)

    Returns:
        List[torch.Tensor]: list of registered images
        (List[torch.Tensor]: list of masks for registered images. Only returned if return_masks==True.)
    """

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

    for ind in tqdm(moving_indices, "Registering Semi-Sequentially"):

        # skip images with missing data
        if imgs[ind] is None:
            print(f"No image data for index {ind}")
            registered_imgs.append(None)
            if return_masks:
                registered_masks.append(None)
            continue

        
        # fetch keypoints to anchor image    
        out = fetch_keypoints(
            imgs[anchor[-1]],
            imgs[ind], 
            masks[anchor[-1]],
            masks[ind],
            warp_consistency,
            match_filtering,
            verbose=verbose
        )
        if 'rotated_moving_img' in out:
            # print("Rotation check")
            imgs[ind] = out['rotated_moving_img']
            masks[ind] = out['rotated_moving_mask']


        # evaluate quality criterion 
        if semi_seq_criterion(mask=masks[ind], keypoints=out['mkpts1'], **semi_sequential_criterion):
        # if semi_seq_criterion(confidence=out['confidence'], **semi_sequential_criterion):
            
            # if condition is satisfied, fit TPS
            mkpts0_filtered = out['mkpts0_filtered']
            mkpts1_filtered = out['mkpts1_filtered']
            tps[ind] = fit_tps_torch(mkpts0_filtered, mkpts1_filtered, alpha=smoothing)
            
        elif ind != 1: # otherwise, register to a more recent image
            
            # make sure we don't link back to an empty picture
            j = ind-1
            while imgs[j] is None: # look for most recent non-None image
                j -= 1
            
            if j != anchor[-1]: # new anchor isn't just the old one again
                anchor.append(j) # set new anchor

                # register to new anchor
                out = fetch_keypoints(
                    imgs[anchor[-1]],
                    imgs[ind], 
                    masks[anchor[-1]],
                    masks[ind],
                    warp_consistency,
                    match_filtering,
                    verbose=verbose
                )
                if 'rotated_moving_img' in out:
                    # print("Rotation check")
                    imgs[ind] = out['rotated_moving_img']
                    masks[ind] = out['rotated_moving_mask']

            mkpts0_filtered = out['mkpts0_filtered']
            mkpts1_filtered = out['mkpts1_filtered']

            # check if we have enough matches for tps
            if len(mkpts0_filtered) > 3:
                tps[ind] = fit_tps_torch(mkpts0_filtered, mkpts1_filtered, alpha=smoothing)
            else:
                print(f"Not enough matches for TPS found at index {ind}")
                imgs[ind] = None # to prevent other images from back-linking to this
                masks[ind] = None
                tps[ind] = None
                registered_imgs.append(None)                    
                if return_masks:
                    registered_masks.append(None)
                continue
        else:
            # condition failed between first and second image => no anchor we can reset to
            mkpts0_filtered = out['mkpts0_filtered']
            mkpts1_filtered = out['mkpts1_filtered']
            print(f"Warning! Poor distribution of matches found between first and second image of sequence. Using {len(mkpts0_filtered)} Matches.")

            # check if we have enough matches for tps
            if len(mkpts0_filtered) > 3:
                tps[ind] = fit_tps_torch(mkpts0_filtered, mkpts1_filtered, alpha=smoothing)
            else:
                print(f"Not enough matches for TPS found at index {ind}")
                imgs[ind] = None # to prevent other images from back-linking to this
                masks[ind] = None
                tps[ind] = None
                registered_imgs.append(None)                    
                if return_masks:
                    registered_masks.append(None)
                continue

        # pick out transforms for relevant steps          
        relevant_tps = [tps[i] for i in anchor + [ind]] 

        # warp images
        if verbose:
            print("Warping Moving Image...")
        registered_imgs.append( warp_tps_torch(relevant_tps[:ind+1], imgs[ind]) )
        if return_masks:
            registered_masks.append( warp_tps_torch(relevant_tps[:ind+1], masks[ind], interpolation_mode='nearest') )
    
    if verbose:
        print(f"Anchors: {[int(a) for a in anchor]}")
    if return_masks:
        return registered_imgs, registered_masks
    else:
        return registered_imgs


def fetch_registered_image_mask_seq(leaf: LeafDataset, registration_method: str, config: dict):
    """
    For the given leaf, registers all leaves using the specified method

    Args:
        leaf: leaf sequence to register
        registration_method: specifies which registration method to use
            "Piecewise Affine": Jonas' pre-existing piecewise affine method
            "LoFTR + TPS Individual": TPS based on LoFTR matches. each leaf is registered directly to the first image of the sequence.
            "LoFTR + TPS Sequential": TPS based on LoFTR matches. each leaf is registered to the preceeding image in the sequence.
            "LoFTR + TPS Semi-Sequential": TPS based on LoFTR matches. each leaf is registered directly to the first image of the sequence.
            "Baseline": Baseline method. Leaf images arent truly registered, only preprocessed and rotated to be aligned with image axes.
        config: dictionary specifying preprocessing configuration, warp consistency parameters, match filtering configurations, and the criterion for semi-sequential registration

    Returns:
        List[torch.Tensor]: List of registered images
        List[torch.Tensor]: List of corresponding registered masks

    """
    if registration_method == "Piecewise Affine":
        imgs, masks = fetch_preregistered_leaf_seq(leaf)
        return imgs, masks
        
    else:
        if registration_method == "LoFTR + TPS Individual":
            # if 'semi_sequential_criterion' in config:
            #     config.pop('semi_sequential_criterion')
            cfg = config.copy()
            cfg.pop('semi_sequential_criterion', None)
            imgs, masks = register_leaf_seq_individual(leaf, **cfg)
        elif registration_method == "LoFTR + TPS Semi-Sequential":
            imgs, masks = register_leaf_seq_semi_sequential(leaf, **config)
        elif registration_method == "LoFTR + TPS Sequential":
            # if 'semi_sequential_criterion' in config:
            #     config.pop('semi_sequential_criterion')
            cfg = config.copy()
            cfg.pop('semi_sequential_criterion', None)
            imgs, masks = register_leaf_seq_sequential(leaf, **cfg)
        elif registration_method == "Baseline":
            # TODO: test
            cfg = config.copy()
            cfg['image_preprocessing']['pre_rotate'] = True 
            imgs, masks = fetch_masked_image_seq(leaf, return_masks=True, image_preprocessing=cfg['image_preprocessing'])
        else:
            raise ValueError(f'Unknown registration method {registration_method}')
        
        return imgs, masks





# ----- Alternative implementations of registration functions using skimage -----
# (Note that this implementation is significantly slower and was thus discarded.
#  We leave it here for reference and potential future use, but note that not all recenet developments are incorporate.)

def register_leaf_seq_individual_skimage(
    leaf: LeafDataset,  
    return_masks: bool=True,
    image_preprocessing: dict=PREPROCESSING_DEFAULT,
    warp_consistency: dict=CONSISTENCY_DEFAULT,
    match_filtering: dict=FILTERING_DEFAULT,
    verbose: bool=False,    
    ):
    """
    For the given leaf, registers all leaves using individual registration and Skimage for TPS.

    Args:
        leaf:return_masks: whether to return masks of registered images
        image_preprocessing: dictionary specifying parameters for image preprocessing, such as image scale, whether to pre-rotate, and parameters of marker erosion
        warp_consistency: dictionary specifying parameters for warp consistency. to disable warp consistency, set it to None.
        match_filtering: dictionary specifying parameters for match filtering/subsampling, such as filtering strategy, target number of landmarks, and minimum confidence threshold.
        verbose: Whether to produce detailed output (for diagnostic purposes)

    Returns:
        List[torch.Tensor]: list of registered images
        (List[torch.Tensor]: list of masks for registered images. Only returned if return_masks==True.)
    """
    
    # retrieve images
    if verbose:
        print("Fetching leaves...")
    imgs = []
    if return_masks:
        masks = []

    for ind in range(leaf.n_leaves):
        img, mask = fetch_image_mask_pair(leaf, ind, **image_preprocessing)
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
        img_moving, mask_moving = register_loftr_tps_skimage(imgs[0], imgs[ind], mask_moving=masks[ind], verbose=verbose, plot_loftr_matches=False, return_tps=False)    


        registered_imgs.append(img_moving)
        if return_masks:
            registered_masks.append(mask_moving)

    if return_masks:
        return registered_imgs, registered_masks
    else:
        return registered_imgs


def register_leaf_seq_sequential_skimage(
    leaf: LeafDataset, 
    return_masks: bool=True,
    image_preprocessing: dict=PREPROCESSING_DEFAULT,
    warp_consistency: dict=CONSISTENCY_DEFAULT,
    match_filtering: dict=FILTERING_DEFAULT,
    verbose: bool=False,
    ):
    """
    For the given leaf, registers all leaves using sequential registration and using Skimage for TPS.
    Note that with Skimage composing TPS transforms is very timing consuming.

    Args:
        leaf: leaf sequence to register
        return_masks: whether to return masks of registered images
        image_preprocessing: dictionary specifying parameters for image preprocessing, such as image scale, whether to pre-rotate, and parameters of marker erosion
        warp_consistency: dictionary specifying parameters for warp consistency. to disable warp consistency, set it to None.
        match_filtering: dictionary specifying parameters for match filtering/subsampling, such as filtering strategy, target number of landmarks, and minimum confidence threshold.
        verbose: Whether to produce detailed output (for diagnostic purposes)

    Returns:
        List[torch.Tensor]: list of registered images
        (List[torch.Tensor]: list of masks for registered images. Only returned if return_masks==True.)
    """
    
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
    for ind in tqdm(moving_indices, "Registering Sequentially"):
        
        # get TPS transform from current image to previous
        
        if imgs[ind] is None: # if image data is missing, add identity transform to stack
            registered_imgs.append(None)
            tps[ind] = AffineTransform() # identity transform
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
        

    if return_masks:
        return registered_imgs, registered_masks
    else:
        return registered_imgs

def register_leaf_seq_semi_sequential_skimage(
    leaf: LeafDataset, 
    return_masks: bool=True,
    image_preprocessing: dict=PREPROCESSING_DEFAULT,
    warp_consistency: dict=CONSISTENCY_DEFAULT,
    match_filtering: dict=FILTERING_DEFAULT,
    semi_sequential_criterion: dict=CRITERION_DEFAULT,
    verbose: bool=False,
    ):
    """
    For the given leaf, registers all leaves using individual registration.

    Args:
        leaf: leaf sequence to register
        return_masks: whether to return masks of registered images
        image_preprocessing: dictionary specifying parameters for image preprocessing, such as image scale, whether to pre-rotate, and parameters of marker erosion
        warp_consistency: dictionary specifying parameters for warp consistency. to disable warp consistency, set it to None.
        match_filtering: dictionary specifying parameters for match filtering/subsampling, such as filtering strategy, target number of landmarks, and minimum confidence threshold.
        verbose: Whether to produce detailed output (for diagnostic purposes)

    Returns:
        List[torch.Tensor]: list of registered images
        (List[torch.Tensor]: list of masks for registered images. Only returned if return_masks==True.)
    """

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

    for ind in tqdm(moving_indices, "Registering Semi-Sequentially"):

        # skip images with missing data
        if imgs[ind] is None:
            print(f"No image data for index {ind}")
            registered_imgs.append(None)
            if return_masks:
                registered_masks.append(None)
            continue

        # fetch keypoints to anchor image
        mkpts0, mkpts1, confidence, _, n_matches = loftr_match(imgs[anchor[-1]], imgs[ind], masks[anchor[-1]], masks[ind], verbose=verbose, return_n_matches=True)
        
        if semi_seq_criterion(mask=masks[ind], keypoints=out['mkpts1'], **semi_sequential_criterion):
            # if condition is satisfied, fit TPS
            _, tps[ind] = tps_skimage_confidence(mkpts0, mkpts1, confidence, threshold, imgs[ind], warp_moving=False, verbose=verbose)
            
        elif ind != 1: # otherwise, register to a more recent image
            
            # make sure we don't link back to an empty picture
            j = ind-1
            while imgs[j] is None: # look for most recent non-None image
                j -= 1
            anchor.append(j) # set new anchor

            # register to new anchor
            mkpts0, mkpts1, confidence, _, n_matches = loftr_match(imgs[anchor[-1]], imgs[ind], masks[anchor[-1]], masks[ind], verbose=verbose, return_n_matches=True)
            
            _, tps[ind] = tps_skimage_confidence(mkpts0, mkpts1, confidence, threshold, imgs[ind], warp_moving=False, verbose=verbose)
        else:
            # condition failed between first and second image => no anchor we can reset to
            print(f"Warning! Poor distribution of matches found between first and second image of sequence.")
            _, tps[ind] = tps_skimage_confidence(mkpts0, mkpts1, confidence, threshold, imgs[ind], warp_moving=False, verbose=verbose)

        
        # compose TPS transforms
        relevant_tps = [tps[i] for i in anchor + [ind]] # pick out transforms for relevant steps
        tps_chain = invert_list(relevant_tps, -1) # invert the list
        if len(tps_chain) > 1:
            coord_map = compose_tps(tps_chain) # compose the transforms
        else:
            coord_map = tps_chain[0]
        

        # warp images
        registered_imgs.append( warp_tps_skimage(imgs[ind], coord_map, verbose=verbose) )
        if return_masks:
            # converting mask to bool makes warp use nearest-neighbor interpolation
            registered_masks.append( warp_tps_skimage(masks[ind].bool(), coord_map, verbose=verbose) )

    
    if verbose:
        print(f"Anchors: {[int(a) for a in anchor]}")
    if return_masks:
        return registered_imgs, registered_masks
    else:
        return registered_imgs
