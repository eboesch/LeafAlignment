import os
import cv2
import kornia as K
import kornia.feature as KF
import kornia.geometry.transform as KT
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import torch
from kornia_moons.viz import draw_LAF_matches
import skimage as ski

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

def plot_matches(img_fix, keypts_fix, img_mov, keypts_mov, inliers, N_show: int=100, inliers_only: bool=True, vertical: bool=True):
    """
    Plots matches between images.

    Args:
        img_fix:    fixed image
        keypts_fix: coordinates of keypoints in fixed image 
        img_mov:    moving image
        keypts_mov: coordinates of keypoints in moving image 
        inliers:     array containing classification as inlier of each match
        N_show (int):   how many matches to plot 
        inliers_only (bool):    whether to plot only matches classified as inliers
        vertical (bool):    whether to plot images vertically stacked.
    """
    # select points to plot
    if inliers_only:
        inliers_t = torch.as_tensor(inliers).flatten() # flatten inliers array and convert to tensor
        inlier_idx = torch.nonzero(inliers_t).flatten() # extract indices of True values

        N_show = min(N_show, inliers.sum())
        rand_idx = torch.randperm(len(inlier_idx))[:N_show]  
        show_idx = inlier_idx[rand_idx] # plot subset of inliers
    else:
        N_show = min(N_show, len(keypts_fix))
        show_idx = torch.randperm(len(keypts_fix))[:N_show] # plot subset of *all* matches
    
    keypts_fix_show = keypts_fix[show_idx]
    keypts_mov_show = keypts_mov[show_idx]
    inliers_show = inliers[show_idx]

    # plot
    fig, ax = draw_LAF_matches(
        KF.laf_from_center_scale_ori(
            keypts_fix_show.view(1, -1, 2),
            torch.ones(keypts_fix_show.shape[0]).view(1, -1, 1, 1),
            torch.ones(keypts_fix_show.shape[0]).view(1, -1, 1),
        ),
        KF.laf_from_center_scale_ori(
            keypts_mov_show.view(1, -1, 2),
            torch.ones(keypts_mov_show.shape[0]).view(1, -1, 1, 1),
            torch.ones(keypts_mov_show.shape[0]).view(1, -1, 1),
        ),
        torch.arange(keypts_fix_show.shape[0]).view(-1, 1).repeat(1, 2),
        K.tensor_to_image(img_fix),
        K.tensor_to_image(img_mov),
        inliers_show,
        draw_dict={"inlier_color": (0.2, 1, 0.2), "tentative_color": None, "feature_color": (0.2, 0.5, 1), "vertical": vertical},
        return_fig_ax=True
    )

    return fig, ax
    # fig.savefig('LeafAlignment/loftr_test.png')
    # fig.show()


def plot_matches_conf(img_fix, keypts_fix, img_mov, keypts_mov, confidence, N_show: int=100, vertical: bool=True):
    """
    Plots matches between images, colored by confidence.

    Args:
        img_fix:    fixed image
        keypts_fix: coordinates of keypoints in fixed image 
        img_mov:    moving image
        keypts_mov: coordinates of keypoints in moving image 
        inliers:     array containing classification as inlier of each match
        N_show (int):   how many matches to plot 
        inliers_only (bool):    whether to plot only matches classified as inliers
        vertical (bool):    whether to plot images vertically stacked.
    """

    if type(img_fix) == torch.Tensor:
        img_fix = K.color.rgb_to_grayscale(img_fix)
        img_fix = K.tensor_to_image(img_fix)
    if type(img_mov) == torch.Tensor:
        img_mov = K.color.rgb_to_grayscale(img_mov)
        img_mov = K.tensor_to_image(img_mov)

    img_pair = np.concatenate([img_fix, img_mov], axis=0)


    # Prepare figure
    if vertical:
        fig, ax = plt.subplots(figsize=(8,8)) # for vertical plot
    else:
        fig, ax = plt.subplots(figsize=(12,6)) # for horizontal plot
    
    ax.imshow(img_pair, cmap='gray')

    # select matches to show
    N_show = min(N_show, len(keypts_fix))
    show_idx = torch.randperm(len(keypts_fix))[:N_show] # plot subset of *all* matches
    
    keypts_fix_show = keypts_fix[show_idx]
    keypts_mov_show = keypts_mov[show_idx]
    conf_show = confidence[show_idx]

    # Normalize confidence and map to colormap
    conf_norm = (conf_show - conf_show.min()) / (conf_show.max() - conf_show.min())
    colors = cm.viridis(conf_norm.cpu().numpy())

    # Draw matches
    if vertical:
        for (x0, y0), (x1, y1), color in zip(keypts_fix_show, keypts_mov_show, colors):
            ax.scatter([x0, x1 ], [y0, y1 + img_fix.shape[0]], color=color, s=4)
            ax.plot([x0, x1 ], [y0, y1+ img_fix.shape[0]], color=color, linewidth=1)
    else:
        for (x0, y0), (x1, y1), color in zip(keypts_fix_show, keypts_mov_show, colors):
            ax.scatter([x0, x1 + img_fix.shape[1]], [y0, y1], color=color, s=2)
            ax.plot([x0, x1 + img_fix.shape[1]], [y0, y1], color=color, linewidth=1)

    ax.axis('off')

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=conf_show.min(), vmax=conf_show.max()))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('LoFTR Confidence')

    # plt.show()
    return fig, ax



def plot_match_coverage(img_fix, keypts_fix, img_mov, keypts_mov, confidence):
    """
    Args:
        img_fix:    fixed image
        keypts_fix: coordinates of keypoints in fixed image 
        img_mov:    moving image
        keypts_mov: coordinates of keypoints in moving image 
        confidence: confidence of each match
    """
    
    # kornia and torch expect C x H x W, while skimage & matplotlib expect H x W x C
    if type(img_fix) == torch.Tensor:
    #     img_fix = change_dimension_order(img_fix)
        img_fix = K.tensor_to_image(img_fix)
    if type(img_mov) == torch.Tensor:
    #     img_mov = change_dimension_order(img_mov)
        img_mov = K.tensor_to_image(img_mov)


    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=[12, 6], sharex=True, sharey=True)

    cmap = 'viridis'

    ax1.imshow(img_fix)
    ax1.scatter(keypts_fix[:,0], keypts_fix[:,1], s=2, c=confidence, cmap=cmap, vmin=0, vmax=1)
    ax1.set_title('Keypoints on Image 0')

    ax2.imshow(img_mov)
    sc2 = ax2.scatter(keypts_mov[:,0], keypts_mov[:,1], s=2, c=confidence, cmap=cmap, vmin=0, vmax=1)
    ax2.set_title('Keypoints on Image 1')

    cbar = fig.colorbar(sc2, ax=[ax1, ax2], orientation='horizontal', fraction=0.046, pad=0.1)
    cbar.set_label("Confidence")
    # fig.colorbar(im, ax=ax2)

    return fig, (ax1, ax2)

# def change_dimension_order(img):
#     # kornia and torch expect C x H x W, while skimage & matplotlib expect H x W x C
#     return img[0].permute(1, 2, 0)

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

    tps = ski.transform.ThinPlateSplineTransform()
    if verbose:
        print("Estimating TPS transform...")
    tps.estimate(img_fix_mks, img_mov_mks) # estimate transform from img_fix -> img_mov
    if verbose:
        print("Transforming moving image...")
    warped = ski.transform.warp(img_mov_reordered, tps) # warp uses inverse transform, i.e. img_mov -> img_fix

    return warped, tps

def plot_img_transform(img_mov, img_mov_warped, plot_keypts: bool=False, keypts_mov=None, keypts_warped=None):
    """
    plots original moving image and transformed moving image
    """

    # kornia and torch expect C x H x W, while skimage & matplotlib expect H x W x C
    if type(img_mov) == torch.Tensor:
        # img_mov = change_dimension_order(img_mov)
        img_mov = K.tensor_to_image(img_mov)
    if type(img_mov_warped) == torch.Tensor:
        # img_mov_warped = change_dimension_order(img_mov_warped)
        img_mov_warped = K.tensor_to_image(img_mov_warped)


    fig, axs = plt.subplots(1, 2, figsize=(12,6))

    axs[0].imshow(img_mov, cmap='gray')
    if plot_keypts:
        axs[0].scatter(keypts_mov[:, 0], keypts_mov[:, 1], s=2, color='cyan')
    axs[0].set_title("Original Moving Image")
    axs[0].axis('off')
    axs[1].imshow(img_mov_warped, cmap='gray')
    if plot_keypts:
        axs[1].scatter(keypts_warped[:, 0], keypts_warped[:, 1], s=2, color='cyan')
    axs[1].set_title("Transformed Moving Image")
    axs[1].axis('off')

    return fig, axs


def plot_overlay(img_fix, img_mov):
    """
    ideally the fixed image is in gray scale
    """
    # kornia and torch expect C x H x W, while skimage & matplotlib expect H x W x C
    if type(img_fix) == torch.Tensor:
    #     img_fix = change_dimension_order(img_fix)
        img_fix = K.color.rgb_to_grayscale(img_fix)
        img_fix = K.tensor_to_image(img_fix)
    if type(img_mov) == torch.Tensor:
    #     img_mov = change_dimension_order(img_mov)
        img_mov = K.tensor_to_image(img_mov)

    fig = plt.figure(figsize=(12,6))
    plt.imshow(img_fix, cmap='gray')
    plt.imshow(img_mov, cmap='hot', alpha=0.5)
    plt.title("Overlay: fixed (gray) + moving (hot)")
    plt.axis('off')
    plt.show()

    return fig

def plot_image_pair(img1, img2, img1_ind: int=None, img2_ind: int = 2, title: str=None):
    if img1_ind is None:
        img1_ind = 1
    if img2_ind is None:
        img2_ind = 2

    fig, axs = plt.subplots(1, 2, figsize=(12,6))
    axs[0].imshow(K.tensor_to_image(img1))
    axs[0].set_title(f"Image {img1_ind}")
    # axs[0].axes('off')

    axs[1].imshow(K.tensor_to_image(img2))
    axs[1].set_title(f"Image {img2_ind}")
    if title is not None:
        fig.suptitle(title, fontsize=22, y=0.86)
    plt.tight_layout()
    
    return fig, axs

# ----------------- masking ------------------------------------------

  
def crop_margins(img, top, bottom, left, right):
    B, C, H, W = img.shape
    y1 = top
    y2 = H - bottom
    x1 = left
    x2 = W - right

    # top-left, top-right, bottom-right and bottom-left
    tl = [x1, y1]
    tr = [x2, y1]
    bl = [x1, y2]
    br = [x2, y2]

    # build index tensor
    # each ROI: [y1, y2, x1, x2]
    boxes = torch.tensor([[tl, tr, br, bl]], device=img.device)
    # print(boxes, boxes.shape)
    # boxes = boxes.repeat(B, 3)
    # print(boxes, boxes.shape)

    return KT.crop_by_indices(img, boxes)

def crop_img(img, x_min, x_max, y_min, y_max, center=None):
    # add batch dimension if necessary
    if img.dim() == 3:
        img = img.unsqueeze(0)

    B, C, H, W = img.shape

    # top-left, top-right, bottom-right and bottom-left
    tl = [x_min, y_min]
    tr = [x_max, y_min]
    bl = [x_min, y_max]
    br = [x_max, y_max]

    # build index tensor
    boxes = torch.tensor([[tl, tr, br, bl]], device=img.device)

    cropped_img = KT.crop_by_indices(img, boxes)

    if center is not None:
        new_center = center - [x_min, y_min]
        return cropped_img, new_center
    else:
        return cropped_img 

def pil_to_kornia(pil_img):
    np_img = np.array(pil_img)
    tensor_img = K.image_to_tensor(np_img).float() / 255.0 
    return tensor_img.unsqueeze(0)

def keypoints_roi_to_image(kp_roi: np.ndarray, roi: dict):
    """
    kp_crop: (N,2) keypoints in crop coordinates (TXT)
    roi: dict with rotation_matrix (2x3) and bounding_box
    Returns: kp_full (N,2) in original image coordinates
    """
    kp_crop = kp_roi.astype(np.float64)

    # 1) shift by top-left of bounding box to get coordinates in rotated image
    box = np.asarray(roi["bounding_box"], dtype=np.float64)
    bbox_min = box.min(axis=0)  # [x_min, y_min]
    kp_rot_img = kp_crop + bbox_min  # coordinates in rotated image

    # 2) invert rotation to map back to original image
    R = np.asarray(roi["rotation_matrix"], dtype=np.float64)
    rot = R[:, :2]
    trans = R[:, 2:]
    rot_inv = np.linalg.inv(rot)
    kp_full = (kp_rot_img - trans.T) @ rot_inv.T

    return kp_full

def mask_leaf(img: torch.Tensor, keypts: np.ndarray, erode_px: int = 0, return_center: bool=True, return_bounds: bool=False):
    img = pil_to_kornia(img)

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
        # FIXME
        print("erosion not yet functional")
        # kernel = torch.ones((erode_px,erode_px), dtype=torch.float32, device=mask_t.device)
        # unsqueeze mask to add batch dim
        # mask_t = morph.erosion(mask_t.unsqueeze(0), kernel, engine='convolution').squeeze(0) 
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


def erode_leaf(available_data, index):
    # img = available_data['images'][index]
    kpts_img = keypoints_roi_to_image(available_data['keypoints'][index], available_data['rois'][index])
    masked_img, mask_t, center = mask_leaf(available_data['images'][index], kpts_img, erode_px=0, return_center=True, return_bounds=False)
    img_scaled = scale_image(masked_img, 1.2, center)
    masked_scaled_img = img_scaled * mask_t 
    return masked_scaled_img

def erode_crop_leaf(available_data, index):
    kpts_img = keypoints_roi_to_image(available_data['keypoints'][index], available_data['rois'][index])
    masked_img, mask_t, center, bounds = mask_leaf(available_data['images'][index], kpts_img, erode_px=0, return_center=True, return_bounds=True)
    x_min, y_min = bounds[0]
    x_max, y_max = bounds[1]
    cropped_img, new_center = crop_img(masked_img, x_min, x_max, y_min, y_max, center)
    cropped_mask = crop_img(mask_t, x_min, x_max, y_min, y_max)
    img_scaled = scale_image(cropped_img, 1.2, new_center)
    masked_scaled_img = img_scaled * cropped_mask 
    return masked_scaled_img

def fetch_leaves(indices: list, available_data, background_type: str='Original'):
    if background_type == "Original":
        img = [pil_to_kornia(available_data['images'][index]) for index in indices]
    elif background_type == "Eroded":
        img = [erode_leaf(available_data, index=index) for index in indices]
    elif background_type == "Eroded+Cropped":
        img = [erode_crop_leaf(available_data, index=index) for index in indices]
    else:
        raise ValueError(f"Unknown background type '{background_type}'")

    return img

# ------------- transforms ---------------------------------------

def affine(img, rot_angle_deg=0, fx=0, fy=0, scale=1.0):
    # add batch dim if necessary
    if img.dim() == 3:
        img = img.unsqueeze(0)
    
    B,C,H,W = img.shape
    tx = W*fx/100
    ty = H*fy/100
    angle = torch.tensor([rot_angle_deg], dtype=img.dtype, device=img.device).repeat(B)
    scale = torch.tensor([scale, scale], dtype=img.dtype, device=img.device).repeat(B,1)
    center = torch.tensor([[img.shape[-1]/2, img.shape[-2]/2]], dtype=img.dtype, device=img.device).repeat(B,1)
    translation = torch.tensor([[tx, ty]], dtype=img.dtype, device=img.device).repeat(B,1)
    matrix = K.geometry.transform.get_affine_matrix2d(translation, center, scale, angle)
    return K.geometry.transform.warp_affine(img, matrix[:,:2,:], dsize=(H, W))

def adjust_color(img, brightness=0.0, contrast=0.0, saturation=1.0):
    # brightness ∈ [-1, 1], contrast > 0
    img = K.enhance.adjust_brightness(img, brightness)
    img = K.enhance.adjust_contrast(img, contrast+1e-6)
    img = K.enhance.adjust_saturation(img, saturation)
    K.enhance.adjust_gamma()
    return img

def gaussian_blur(img, kernel_size=0.0, sigma=0.0):
    # add batch dim if necessary
    if img.dim() == 3:
        img = img.unsqueeze(0)

    blur = K.filters.GaussianBlur2d((kernel_size, kernel_size), (sigma, sigma))

    return blur(img)

def add_gaussian_noise(img, sigma=0.0):
    noise = torch.randn_like(img)  # Generating on GPU is fastest with `torch.randn_like(...)`
    if sigma != 1.0:  # `if` is cheaper than multiplication
        noise *= sigma
    return (img + noise).clamp(0,1)


def transform_img(img, transform_name, magnitude):
    # geometric
    if transform_name == "Rotation":
        return affine(img, rot_angle_deg=magnitude, fx=0, fy=0, scale=1.0)
    if transform_name == "Translation":
        return affine(img, rot_angle_deg=0, fx=magnitude, fy=magnitude, scale=1.0)
    if transform_name == "Scale":
        return affine(img, rot_angle_deg=0, fx=0, fy=0, scale=magnitude)

    # intensity
    if transform_name == "Brightness":
        return K.enhance.adjust_brightness(img, magnitude)
    if transform_name == "Contrast":
        return K.enhance.adjust_contrast(img, magnitude)
    if transform_name == "Saturation":
        return K.enhance.adjust_saturation(img, magnitude)
    if transform_name == "Gamma":
        return K.enhance.adjust_gamma(img, magnitude)
    if transform_name == "Hue":
        return K.enhance.adjust_hue(img, magnitude)

    # noise / blur
    if transform_name == "Gaussian Noise":
        return add_gaussian_noise(img, sigma=magnitude)
    if transform_name == "Gaussian Blur":
        return gaussian_blur(img, kernel_size=magnitude, sigma=magnitude)
   
    # add more as needed
    return img

# ------------- metrics ------------------------------------------

def ncc(img1, img2):

    if type(img1) == np.ndarray:
        img1 = K.image_to_tensor(img1)
    if type(img2) == np.ndarray:
        img2 = K.image_to_tensor(img2)

    img1 = K.color.rgb_to_grayscale(img1)
    img2 = K.color.rgb_to_grayscale(img2)
    

    img1 = img1 - img1.mean()
    img2 = img2 - img2.mean()
    std_img1 = torch.std(img1)
    std_img2 = torch.std(img2)
    return torch.sum(img1*img2) / (std_img1 * std_img2 * torch.numel(img1) + 1e-13)



def histogram2d_scatter(img1, img2, bins=64, eps=1e-8):
    """
    a, b: (H,W) or flattened tensors of equal shape
    bins: number of histogram bins per dimension
    """

    # flatten
    img1 = img1.reshape(-1)
    img2 = img2.reshape(-1)

    # normalize into [0, bins-1]
    img1_min, img1_max = img1.min(), img1.max()
    img2_min, img2_max = img2.min(), img2.max()

    img1_scaled = (img1 - img1_min) / (img1_max - img1_min + eps)
    img2_scaled = (img2 - img2_min) / (img2_max - img2_min + eps)

    img1_i = torch.clamp((img1_scaled * (bins - 1)).long(), 0, bins - 1)
    img2_i = torch.clamp((img2_scaled * (bins - 1)).long(), 0, bins - 1)

    # 2D histogram via scatter_add
    hist = torch.zeros((bins, bins), device=img1.device)
    hist.index_put_((img1_i, img2_i), torch.ones_like(img1_i, dtype=hist.dtype),
                    accumulate=True)

    return hist



def mutual_information(img1, img2, bins=100):
    if type(img1) == np.ndarray:
        img1 = K.image_to_tensor(img1)
    if type(img2) == np.ndarray:
        img2 = K.image_to_tensor(img2)

    img1 = K.color.rgb_to_grayscale(img1)
    img2 = K.color.rgb_to_grayscale(img2)

    # min_val = 0 # min(img1.min(), img2.min())
    # max_val = 1 # max(img1.max(), img2.max())

    # hist1 = torch.histc(img1, bins=bins, min=min_val, max=max_val)
    # hist2 = torch.histc(img2, bins=bins, min=min_val, max=max_val)

    # imgs = torch.cat([img1, img2])
    # hist12 = torch.histc(imgs, bins=bins, min=min_val, max=max_val)

    # non_zero = hist12 > 0

    # hist1 /= torch.sum(hist1)
    # hist2 /= torch.sum(hist2)
    # hist12 /= torch.sum(hist12)
    # print(hist1.min())
    # print(hist2.min())
    # print(hist12.min())
    # print(torch.max(hist1[non_zero]*hist2[non_zero]))
    # print(torch.max(hist12[non_zero] / (hist1[non_zero] * hist2[non_zero])))
    # print(torch.max(torch.log( hist12[non_zero] / (hist1[non_zero] * hist2[non_zero]))))  

    # mi = torch.sum(hist12[non_zero] * torch.log( hist12[non_zero] / (hist1[non_zero] * hist2[non_zero]) ))


    hist2d = histogram2d_scatter(img1, img2, bins=bins)

    # convert to distribution
    p12 = hist2d / hist2d.sum()
    p1 = p12.sum(1)     # (bins,)
    p2 = p12.sum(0)     # (bins,)

    # compute MI
    p1_p2 = p1[:, None] * p2[None, :]
    non_zero = p12 > 0

    mi = torch.sum(p12[non_zero] * torch.log(p12[non_zero] / (p1_p2[non_zero] + 1e-13)))

    return mi
