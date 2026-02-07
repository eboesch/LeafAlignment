import os
import cv2
import kornia as K
# import kornia.feature as KF
# import kornia.geometry.transform as KT
# import matplotlib.pyplot as plt
# import matplotlib.cm as cm
import numpy as np
import torch
# from kornia_moons.viz import draw_LAF_matches
# import skimage as ski
from utils import crop_img, convert_image_to_tensor

# ----------------- masking ------------------------------------------


def keypoints_roi_to_image(kp_roi: np.ndarray, roi: dict):
    """
    kp_crop: (N,2) keypoints in crop coordinates (TXT)
    roi: dict with rotation_matrix (2x3) and bounding_box
    Returns: kp_full (N,2) in original image coordinates
    """
    if kp_roi is None:
        return None
    kp_crop = kp_roi.astype(np.float64)

    # 1) shift by top-left of bounding box to get coordinates in rotated image
    box = roi["bounding_box"]
    R = roi["rotation_matrix"]
    if (box is None) or (R is None):
        return None
    box = np.asarray(box, dtype=np.float64)
    bbox_min = box.min(axis=0)  # [x_min, y_min]
    kp_rot_img = kp_crop + bbox_min  # coordinates in rotated image

    # 2) invert rotation to map back to original image
    R = np.asarray(R, dtype=np.float64)
    rot = R[:, :2]
    trans = R[:, 2:]
    rot_inv = np.linalg.inv(rot)
    kp_full = (kp_rot_img - trans.T) @ rot_inv.T

    return kp_full

def mask_leaf(img: torch.Tensor, keypts: np.ndarray, erode_px: int = 0, return_center: bool=True, return_bounds: bool=False):
    img = convert_image_to_tensor(img)

    B,C,H,W = img.shape
    # Computes the convex hull of the keypoints.
    hull = cv2.convexHull(keypts.astype(np.int32)) # Returns ordered list of points forming a polygon that encloses all the keypoints.
    # center = np.mean(hull, axis=0)
    mins = np.min(hull, axis=0)
    maxs = np.max(hull, axis=0)
    center = (maxs + mins)/2

    # initialize empty mask
    mask = np.zeros((H,W), dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull.reshape(-1,2), 1) # fill in convex hull with value 1
    mask_t = torch.from_numpy(mask).float().to(img.device).unsqueeze(0) # convert mask to tensor and add channel dim
    
    if erode_px > 0:
        kernel = torch.ones((5,5), dtype=torch.float32, device=mask_t.device)
        for _ in range(int(erode_px/5)):
            # unsqueeze mask to add batch dim
            mask_t = K.morphology.erosion(mask_t.unsqueeze(0), kernel, border_type='constant').squeeze(0) 
    
    masked_img = img * mask_t
    if return_center and return_bounds:
        return masked_img, mask_t, center[0], [mins[0], maxs[0]]
    elif return_center:
        return masked_img, mask_t, center[0]
    elif return_bounds:
        return masked_img, mask_t, [mins[0], maxs[0]]
    else:
        return masked_img, mask_t

def scale_image(img: torch.Tensor, scale: float, center: np.array=None):
    """
    img: (C,H,W) or (B,C,H,W) torch tensor
    scale: float >1 to enlarge
    """
    # Add batch dimension if necessary
    if img.dim() == 3:
        img = img.unsqueeze(0)  # (1,C,H,W)

    B,C,H,W = img.shape
    device = img.device
    dtype = img.dtype

    # Scale tensor: (B,2) float32 on same device
    scale_tensor = torch.tensor([[scale, scale]], dtype=dtype, device=device).repeat(B,1)

    if center is not None:
        # can choose custom center through which to scale, default is the center of the tensor
        center = torch.tensor(center, dtype=dtype, device=device).repeat(B,1)
        img_scaled = K.geometry.transform.scale(img, scale_factor=scale_tensor, center=center)
    else:
        img_scaled = K.geometry.transform.scale(img, scale_factor=scale_tensor)

    # Remove batch dim if needed
    if img_scaled.shape[0] == 1:
        img_scaled = img_scaled.squeeze(0)

    return img_scaled

def erode_mask_by_scaling(mask: torch.Tensor, scale: float):
    """
    mask: (1,H,W) or (B,1,H,W) torch tensor
    scale: <1 to shrink the mask (simulate erosion)
    Returns: resized mask with same original image size
    """
    # Add batch dim if needed
    if mask.dim() == 3:
        mask = mask.unsqueeze(0)  # (1,1,H,W)

    B, C, H, W = mask.shape
    device = mask.device
    dtype = mask.dtype

    # Resize mask: shrink by scale factor
    new_H = int(H * scale)
    new_W = int(W * scale)
    mask_small = F.interpolate(mask, size=(new_H, new_W), mode='nearest')

    # Pad back to original size and center
    pad_top = (H - new_H) // 2
    pad_bottom = H - new_H - pad_top
    pad_left = (W - new_W) // 2
    pad_right = W - new_W - pad_left

    mask_eroded = F.pad(mask_small, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)

    # Remove batch dim if needed
    if mask_eroded.shape[0] == 1:
        mask_eroded = mask_eroded.squeeze(0)

    return mask_eroded


def erode_leaf_keypoints(leaf, index, scale=1.2, return_mask=False):
    """
    erode leaf background based on keypoints
    """
    # img = available_data['images'][index]
    kpts_img = keypoints_roi_to_image(leaf.keypoints[index], leaf.rois[index])
    masked_img, mask_t, center = mask_leaf(leaf.images[index], kpts_img, erode_px=0, return_center=True, return_bounds=False)
    img_scaled = scale_image(masked_img, scale, center)
    masked_scaled_img = img_scaled * mask_t 
    if return_mask:
        return masked_scaled_img, mask_t
    else:
        return masked_scaled_img,


def erode_crop_leaf(leaf, index, scale=1.2, return_mask=False):
    kpts_img = keypoints_roi_to_image(leaf.keypoints[index], leaf.rois[index])
    if kpts_img is None:
        if return_mask:
            return None, None  
        else: 
            return None

    masked_img, mask_t, center, bounds = mask_leaf(leaf.images[index], kpts_img, erode_px=0, return_center=True, return_bounds=True)
    x_min, y_min = bounds[0]
    x_max, y_max = bounds[1]
    cropped_img, new_center = crop_img(masked_img, x_min, x_max, y_min, y_max, center)
    cropped_mask = crop_img(mask_t, x_min, x_max, y_min, y_max)
    img_scaled = scale_image(cropped_img, scale, new_center)
    masked_scaled_img = img_scaled * cropped_mask 
    if return_mask:
        return masked_scaled_img, cropped_mask
    else:
        return masked_scaled_img

def erode_leaf(img, mask, scale=1.2, erode_px=60, return_mask=True):
    masked_img = img * mask

    if erode_px > 0:
        kernel = torch.ones((5,5), dtype=torch.float32, device=mask.device)
        for _ in range(int(erode_px/5)):
            # unsqueeze mask to add batch dim
            mask = K.morphology.erosion(mask, kernel, border_type='constant')
        out_img = masked_img * mask
    else:
        scaled_img = scale_image(masked_img, scale)
        out_img = scaled_img * mask

    if return_mask:
        return out_img, mask
    else:
        return out_img

def crop_ROI_erode_leaf(leaf, ind, scale=1.2, erode_px=60, return_mask=True):
    img = convert_image_to_tensor(leaf.images[ind])
    H, W = img.shape[2], img.shape[3]
    roi = leaf.rois[ind]
    rot_mat = roi["rotation_matrix"]
    bbox = roi["bounding_box"]
    keypoints = leaf.keypoints[ind]
    if rot_mat is None or bbox is None or keypoints is None:
        print(f"Error: missing data for leaf {leaf.leaf_uid}")
        return None, None
    rot_mat = np.asarray(rot_mat)
    bbox = np.asarray(bbox)

    # rotate and crop to ROI
    img = K.geometry.transform.warp_affine(img, torch.Tensor(rot_mat).unsqueeze(0), (H, W)) #, align_corners=True)
    img = crop_img(img, bbox[:,0].min(), bbox[:,0].max()-1, bbox[:,1].min(), bbox[:,1].max()-1)

    # generate mask via keypoints
    masked_img, mask_t, center = mask_leaf(img, keypoints, erode_px=erode_px, return_center=True, return_bounds=False)

    if erode_px == 0:    
        img_scaled = scale_image(masked_img, scale, center)
        masked_img = img_scaled * mask_t 
    if return_mask:
        return masked_img, mask_t
    else:
        return masked_img

def fetch_leaves(indices: list, leaf, background_type: str='Original'):
    if background_type == "Original":
        img = [convert_image_to_tensor(leaf.images[index]) for index in indices]
    elif background_type == "Eroded":
        img = [erode_leaf(leaf, index=index) for index in indices]
    elif background_type == "Eroded+Cropped":
        img = [erode_crop_leaf(leaf, index=index) for index in indices]
    else:
        raise ValueError(f"Unknown background type '{background_type}'")

    return img

def fetch_image_mask_pair(leaf, fixed_img_ind, moving_img_ind, method, plot_masked_images=False, plot_loftr_matches=False, old=False):
    if method == "Pairwise Affine":
        img_fixed = convert_image_to_tensor(leaf.target_images[fixed_img_ind])
        mask_fixed = convert_image_to_tensor(leaf.target_masks[fixed_img_ind])
        if (img_fixed is None) or (mask_fixed is None): 
            print(f"Error: missing data for leaf {leaf.leaf_uid}")
            return img_fixed, None, mask_fixed, None
        mask_fixed[mask_fixed != 0] = 1

        img_moving = convert_image_to_tensor(leaf.target_images[moving_img_ind])
        mask_moving = convert_image_to_tensor(leaf.target_masks[moving_img_ind])
        if (img_moving is None) or (mask_moving is None): 
            print(f"Error: missing data for leaf {leaf.leaf_uid}")
            return img_fixed, img_moving, mask_fixed, mask_moving
        mask_moving[mask_moving != 0] = 1
        return img_fixed, img_moving, mask_fixed, mask_moving
        
    else:
        
        if method in ("LoFTR + TPS ROI", "LoFTR + TPS ROI with Markers"):
            # if we want to keep the markers, don't rescale the image
            if old:
                img_scale = {"scale": 1} if method == "LoFTR + TPS ROI with Markers" else {}
            erosion = {"scale": 1, "erode_px": 0} if method == "LoFTR + TPS ROI with Markers" else {"erode_px": 150}


            # fixed image
            if fixed_img_ind == 0:
                img_fixed = convert_image_to_tensor(leaf.images[fixed_img_ind])
                mask_fixed = convert_image_to_tensor(leaf.leaf_masks[fixed_img_ind])

                # if plot_masked_images:
                #     fig, ax = plot_image_pair(img_fixed, mask_fixed, fixed_img_ind, fixed_img_ind, title="Input for fixed image")
                #     fig.show()

                H, W = img_fixed.shape[2], img_fixed.shape[3]
                roi = leaf.rois[fixed_img_ind]
                rot_mat = roi["rotation_matrix"]
                bbox = roi["bounding_box"]
                if rot_mat is None or bbox is None:
                    print(f"Error: missing data for leaf {leaf.leaf_uid}")
                    return None, None, None, None
                rot_mat = np.asarray(rot_mat)
                bbox = np.asarray(bbox)

                img_fixed = K.geometry.transform.warp_affine(img_fixed, torch.Tensor(rot_mat).unsqueeze(0), (H, W)) #, align_corners=True)
                img_fixed = crop_img(img_fixed, bbox[:,0].min(), bbox[:,0].max()-1, bbox[:,1].min(), bbox[:,1].max()-1)
                # img_fixed = img_fixed * mask_fixed
                if old:
                    img_fixed = erode_leaf(img_fixed, mask_fixed, return_mask=False, **img_scale)
                else:
                    img_fixed, mask_fixed = erode_leaf(img_fixed, mask_fixed, return_mask=True, **erosion)

            else:
                if old:
                    img_fixed, mask_fixed = erode_crop_leaf(leaf, fixed_img_ind, return_mask=True, **img_scale)
                else:
                    img_fixed, mask_fixed = crop_ROI_erode_leaf(leaf, fixed_img_ind, return_mask=True, **erosion)
            
            if (img_fixed is None) or (mask_fixed is None): 
                print(f"Error: missing data for leaf {leaf.leaf_uid}")
                return img_fixed, None, mask_fixed, None

            # moving image
            if old:
                img_moving, mask_moving = erode_crop_leaf(leaf, moving_img_ind, return_mask=True, **img_scale)
            else:
                img_moving, mask_moving = crop_ROI_erode_leaf(leaf, moving_img_ind, return_mask=True, **erosion)
            if (img_moving is None) or (mask_moving is None): 
                print(f"Error: missing data for leaf {leaf.leaf_uid}")
                return img_fixed, img_moving, mask_fixed, mask_moving
            size_factor = 2

        elif method in ("LoFTR + TPS Full", "LoFTR + TPS Full with Markers"):
            # if we want to keep the markers, don't rescale the image
            img_scale = {"scale": 1} if method == "LoFTR + TPS ROI with Markers" else {}

            # fixed image
            img_fixed = convert_image_to_tensor(leaf.cropped_images[fixed_img_ind])
            mask_fixed = convert_image_to_tensor(leaf.seg_masks[fixed_img_ind])
            if (img_fixed is None) or (mask_fixed is None): 
                print(f"Error: missing data for leaf {leaf.leaf_uid}")
                return img_fixed, None, mask_fixed, None
            mask_fixed[mask_fixed!=0] = 1
            rmin, rmax, cmin, cmax = crop_coords_zero_borders(mask_fixed)
            img_fixed = crop_img(img_fixed, cmin, cmax, rmin, rmax)
            mask_fixed = crop_img(mask_fixed, cmin, cmax, rmin, rmax)
            img_fixed = erode_leaf(img_fixed, mask_fixed, **img_scale)

            # moving image
            img_moving = convert_image_to_tensor(leaf.cropped_images[moving_img_ind])
            mask_moving = convert_image_to_tensor(leaf.seg_masks[moving_img_ind])
            if (img_moving is None) or (mask_moving is None): 
                print(f"Error: missing data for leaf {leaf.leaf_uid}")
                return img_fixed, img_moving, mask_fixed, mask_moving
            mask_moving[mask_moving!=0] = 1
            rmin, rmax, cmin, cmax = crop_coords_zero_borders(mask_moving)
            img_moving = crop_img(img_moving, cmin, cmax, rmin, rmax)
            mask_moving = crop_img(mask_moving, cmin, cmax, rmin, rmax)
            img_moving = erode_leaf(img_moving, mask_moving, **img_scale)

            size_factor = 4
         
        else:
            raise ValueError(f'Unknown registration method {method}')

        # resize
        height = max(img_fixed.shape[-2], img_moving.shape[-2])
        width = max(img_fixed.shape[-1], img_moving.shape[-1])

        padder = K.augmentation.PadTo((height, width))

        img_fixed = padder(img_fixed)
        img_moving = padder(img_moving)
        mask_fixed = padder(mask_fixed)
        mask_moving = padder(mask_moving)        

        
        H = int(height/size_factor)
        W = int(width/size_factor) 

        img_fixed = K.geometry.resize(img_fixed, (H, W), antialias=True)
        mask_fixed = K.geometry.resize(mask_fixed, (H, W), antialias=False, interpolation='nearest')
        img_moving = K.geometry.resize(img_moving, (H, W), antialias=True)
        mask_moving = K.geometry.resize(mask_moving, (H, W), antialias=False, interpolation='nearest')

        if plot_masked_images:
            fig, ax = plot_image_pair(img_fixed, img_moving, fixed_img_ind, moving_img_ind, title="Masked out input images", title_offset=0.7)
            fig.show()
            fig, ax = plot_image_pair(mask_fixed, mask_moving, fixed_img_ind, moving_img_ind, title="corresponding masks", title_offset=0.7)
            fig.show()


        mkpts0, mkpts1, confidence, _, n_matches = loftr_match(img_fixed, img_moving, verbose=False, return_n_matches=True)

        if plot_loftr_matches:
            # fig, ax = plot_matches(img_fixed, mkpts0, img_moving, mkpts1, inliers, inliers_only=False)
            # fig.show()
            fig, ax = plot_matches_conf(img_fixed, mkpts0, img_moving, mkpts1, confidence, N_show=50, vertical=True)
            fig.show()
            fig, axs = plot_match_coverage(img_fixed, mkpts0, img_moving, mkpts1, confidence)
            fig.show()

        thrsld = 0.5
        
        if n_matches['conf_matches'] > 3:
            warped_moving_img, tps = tps_skimage(mkpts0, mkpts1, confidence, thrsld, img_moving, verbose=False)
            warped_moving_mask, tps = tps_skimage(mkpts0, mkpts1, confidence, thrsld, mask_moving, verbose=False)
        else:
            print("No enough matches for TPS found")
            warped_moving_img = None
            warped_moving_mask = None

        return img_fixed, warped_moving_img, mask_fixed, warped_moving_mask
