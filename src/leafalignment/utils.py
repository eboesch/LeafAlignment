import os
import cv2
import kornia as K
import kornia.geometry.transform as KT
import numpy as np
import torch
from typing import List, Tuple
import yaml
from pathlib import Path
from itertools import combinations
 
def load_resize_image(img_path: str, H: int=375, W: int=600):
    """
    Loads and resizes image at path to desired dimensions

    Args:
        img_path: path to image
        H: desired height of output image
        W: desired height of output image

    Returns:
        torch.Tensor: image
    """

    assert os.path.exists(img_path), "Invalid path to images"

    img = K.io.load_image(img_path, K.io.ImageLoadType.RGB32)[None, ...]
    img = K.geometry.resize(img, (H, W), antialias=True)

    return img

def load_config(config_path: str = "../configs/default_s100.yaml"):
    """
    Loads and parses a YAML configuration file.
    
    Args:
        config_path (str): path to the config file to be loaded.

    Returns:
        dict: dictionary of the configuration
    """
    try:
        # Use Path to handle file paths cross-platform
        with open(Path(config_path), "r") as f:
            # safe_load() parses YAML into Python dict/list/primitives
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        raise RuntimeError(f"Config file not found at: {config_path}")
    except yaml.YAMLError as e:
        raise RuntimeError(f"Failed to parse YAML: {e}")


# ----- Cropping & Preprocessing -----

def crop_coords_zero_borders(mask: torch.Tensor):
    """
    Determines the min/max rows/columns containing foreground based on the mask.

    Args:
        mask (torch.Tensor): values 0/1 (or nonzero = foreground)

    Returns: 
        rmin, rmax, cmin, cmax: indices of min/max row/column
    """

    if mask.dim() == 4:
        mask.squeeze(0)
        # Collapse channels
        mask = torch.any(mask != 0, dim=0)
        mask = mask.squeeze(0)
    elif mask.dim() == 3:
        # Collapse channels
        mask = torch.any(mask != 0, dim=0)
        mask = mask.squeeze(0)

    # Rows / cols containing foreground
    rows = torch.any(mask, dim=1) 
    cols = torch.any(mask, dim=0) 

    if rows.sum() == 0 or cols.sum() == 0:
        print("Warning! Mask empty!")
        rmin = rmax = cmin = cmax = 0
        return rmin, rmax, cmin, cmax

    # fetch indices
    r_idx = torch.where(rows)[0]
    c_idx = torch.where(cols)[0]

    rmin, rmax = r_idx[0], r_idx[-1]
    cmin, cmax = c_idx[0], c_idx[-1]

    return rmin, rmax, cmin, cmax

def crop_coords_zero_borders_batch(mask: torch.Tensor):
    """
    Determines the min/max rows/columns containing foreground based on the mask per batch element.

    Args:
        mask (torch.Tensor): shape (B, C, H, W), values 0/1 (or nonzero = foreground)

    Returns: 
        list of indices of min/max foreground row/column, one per batch element
    """

    B, C, H, W = mask.shape

    # Collapse channels → foreground per pixel
    fg = torch.any(mask != 0, dim=1)  # (B, H, W)

    # Rows / cols containing foreground
    rows = torch.any(fg, dim=2)  # (B, H)
    cols = torch.any(fg, dim=1)  # (B, W)

    crop_coords = []

    for b in range(B):
        if rows[b].sum() == 0 or cols[b].sum() == 0:
            print("Warning! Mask empty!")
            rmin = rmax = cmin = cmax = 0
            crop_coords.append([rmin, rmax, cmin, cmax])
            continue

        r_idx = torch.where(rows[b])[0]
        c_idx = torch.where(cols[b])[0]

        rmin, rmax = r_idx[0], r_idx[-1]
        cmin, cmax = c_idx[0], c_idx[-1]

        crop_coords.append([rmin, rmax, cmin, cmax])

    return crop_coords

def crop_margins(img: torch.Tensor, top: int=0, bottom: int=0, left: int=0, right: int=0):
    """
    Crops margins of the provided magnitude off each side of the image.

    Args:
        img (torch.Tensor): image to be cropped. shape (B, C, H, W)
        top (int): margin to crop off the top
        bottom (int): margin to crop off the bottom
        left (int): margin to crop off the left
        right (int): margin to crop off the right

    Returns:
        torch.Tensor: cropped image 
    """
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
    boxes = torch.tensor([[tl, tr, br, bl]], device=img.device)

    return KT.crop_by_indices(img, boxes)

def crop_img(img: torch.Tensor, x_min: int, x_max: int, y_min: int, y_max: int, center: Tuple[int, int]=None):
    """
    Crops the image the provided min/max x/y coordinates.
    If a center of the Region of Interest is provided, its coordinates in the cropped image are also returned.

    Args:
        img (torch.Tensor): image to crop, shape (B, C, H, W)
        x_min (int): minimum x coordinate to crop to
        x_max (int): maximum x coordinate to crop to
        y_min (int): minimum y coordinate to crop to
        y_max (int): maximum y coordinate to crop to
        center (List[int, int]): Optional, coordinates of center of ROI

    Returns:
        torch.Tensor: the cropped image
        if a center is provided, its coordinates in the cropped image are also returned
    """
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

def undo_rotation(img: torch.Tensor, M_cv2, interpolation_mode: str="bilinear"):
    """
    Undoes the provided OpenCV rotation using the provided interpolation mode.

    Args:
        img (torch.Tensor): Image to (un)rotate, shape (B, C, H, W)
        M_cv2: (2, 3) affine transformation matrix used previously with cv2.warpAffine
        interpolation_mode (str): interpolation mode to use for rotating the image. default: "bilinear"

    Returns:
        torch.Tensor: (un)rotated image
    """
    B, C, H, W = img.shape
    device, dtype = img.device, img.dtype

    # bring cv2 matrix into torch
    M = torch.tensor(M_cv2, device=device, dtype=dtype)
    M = M.unsqueeze(0).repeat(B, 1, 1)

    # invert affine
    M_inv = K.geometry.transform.invert_affine_transform(M)

    # original image corners (pixel coords)
    corners = torch.tensor(
        [[[0, 0, 1],
          [W, 0, 1],
          [W, H, 1],
          [0, H, 1]]],
        device=device,
        dtype=dtype
    ).transpose(1, 2)

    warped = M_inv @ corners
    xs, ys = warped[:, 0], warped[:, 1]

    min_x, max_x = xs.min(), xs.max()
    min_y, max_y = ys.min(), ys.max()

    new_W = int(torch.ceil(max_x - min_x).item())
    new_H = int(torch.ceil(max_y - min_y).item())

    # shift so everything is visible
    M_inv[:, 0, 2] -= min_x
    M_inv[:, 1, 2] -= min_y

    # rotate
    out = K.geometry.transform.warp_affine(
        img,
        M_inv,
        dsize=(new_H, new_W),
        mode=interpolation_mode,
        padding_mode="zeros",
        align_corners=False
    )

    out = torch.clamp(out, 0.0, 1.0) # clamp back to [0,1] to remove small overshoots

    return out

def match_sizes_resize(img1: torch.Tensor, img2: torch.Tensor, mask1: torch.Tensor=None, mask2: torch.Tensor=None, size_factor: int=None):
    """
    Pads images to the same size and resizes them to a smaller size. If masks are provided, they are equally padded and resized.
    If no size factor is provided, one is determined automatically based on image size.

    Args:
        img1, img2 (torch.Tensor): images to resize 
        mask1, mask2 (torch.Tensor):  Optional masks to resizes the same as images
        size_factor (int): Optional factor by which to shrink image. A value of 2 results in images with half the original height and width

    Returns:
        img1, img2, (mask1, mask2): resized images (and masks)
    """
    if (mask1 is None) != (mask2 is None):
        raise ValueError("Provide both mask1 and mask2, or neither.")
    
    # pad images to same size
    height = max(img1.shape[-2], img2.shape[-2])
    width = max(img1.shape[-1], img2.shape[-1])

    padder = K.augmentation.PadTo((height, width))

    img1 = padder(img1)
    img2 = padder(img2)   

    # determine resizing factor
    if size_factor is None:
        total = height * width * 1e-6
        if total < 1.5:
            size_factor = 1
        elif total < 6:
            size_factor = 2
        elif total < 13.5:
            size_factor = 3
        else:
            size_factor = 4

    # resize images
    H = int(height/size_factor)
    W = int(width/size_factor) 

    img1 = K.geometry.resize(img1, (H, W), antialias=True)
    img2 = K.geometry.resize(img2, (H, W), antialias=True)

    img1 = torch.clamp(img1, 0.0, 1.0)
    img2 = torch.clamp(img2, 0.0, 1.0)

    if mask1 is None:
        return img1, img2 

    # pad and resize masks
    mask1 = padder(mask1)
    mask2 = padder(mask2)

    mask1 = K.geometry.resize(mask1, (H, W), antialias=False, interpolation='nearest')
    mask2 = K.geometry.resize(mask2, (H, W), antialias=False, interpolation='nearest')

    return img1, img2, mask1, mask2

def match_sizes_resize_batch(imgs: List[torch.Tensor], masks: List[torch.Tensor]=None, size_factor: int=None):
    """
    Pads images to the same size and resizes them to a smaller size. If masks are provided, they are equally padded and resized.
    If no size factor is provided, one is determined automatically based on image size.

    Args:
        imgs (List[torch.Tensor]): list of images to resize 
        mask1, mask2 (List[torch.Tensor]):  Optional list of masks to resizes the same as images
        size_factor (int): Optional factor by which to shrink image. A value of 2 results in images with half the original height and width

    Returns:
        imgs, (masks): lists of resized images (and masks)
    """    
    # filter out None tensors for size computation
    valid_imgs = [img for img in imgs if img is not None]
    if len(valid_imgs) == 0:
        # if all images are None, can just return them as is
        return imgs if masks is None else (imgs, masks)
    
    heights = [img.shape[-2] for img in valid_imgs]
    widths = [img.shape[-1] for img in valid_imgs]

    height = max(heights)
    width = max(widths)

    # pad images to have same size
    padder = K.augmentation.PadTo((height, width))

    for i, img in enumerate(imgs):
        if img is None: # skip None images
            continue

        imgs[i] = padder(img)

    # determine resizing scale factor
    if size_factor is None:
        total = height * width * 1e-6
        if total < 1.5:
            size_factor = 1
        elif total < 6:
            size_factor = 2
        elif total < 13.5:
            size_factor = 3
        else:
            size_factor = 4


    H = int(height/size_factor)
    W = int(width/size_factor) 

    # resize images
    for i, img in enumerate(imgs):
        if img is None: # skip None images
            continue

        img = K.geometry.resize(img, (H, W), antialias=True)
        img = torch.clamp(img, 0.0, 1.0)
        imgs[i] = img

    if masks is None:
        return imgs

    # process masks
    for i, mask in enumerate(masks):
        if mask is None: # skip None mask
            continue
        masks[i] = padder(mask)

    for i, mask in enumerate(masks):
        if mask is None: # skip None mask
            continue
        mask = K.geometry.resize(mask, (H, W), antialias=False, interpolation='nearest')
        mask = torch.clamp(mask, 0.0, 1.0)
        masks[i] = mask

    return imgs, masks

# ----- Loading / Conversion -----

def pil_to_kornia(pil_img):
    """Converta a PIL image to a Kornia/torch image tensor"""
    np_img = np.array(pil_img)
    tensor_img = K.image_to_tensor(np_img).float() / 255.0 
    return tensor_img.unsqueeze(0)

def convert_image_to_tensor(img):
    """Converts image of different possible file types to a Kornia/torch image tensor"""
    if img is None:
        return img
    elif type(img) == torch.Tensor:
        if img.dim == 3:
            img.unsqueeze(0) # add batch dim
        return img
    elif type(img) == np.ndarray:
        img = K.image_to_tensor(img)
        if img.dim() == 3:
            img = img.unsqueeze(0)
        # convert to [0,1] range
        if img.max() < 1+1e-3:
            img = torch.clamp(img, 0.0, 1.0)
        if img.max() > 1:
            img = img/255.0
        return img
    else:
        return pil_to_kornia(img)

def convert_img_tensor_to_numpy(img_tensor: torch.Tensor):
    """Converts an image tensor to a numpy image"""
    img_np = (
        img_tensor[0]                      # select batch
        .permute(1, 2, 0)         # CHW → HWC (or squeeze for grayscale)
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32) 
    )
    return img_np

# ----- Generic Utility Functions -----

def weighted_average(tensor: torch.Tensor, weights: torch.Tensor):
    """Computes the weighted average of the input tensor using the provided weights."""
    weighted_sum = torch.sum(tensor*weights)
    weight_sum = torch.sum(weights)
    weighted_average = weighted_sum / weight_sum
    return weighted_average

def invert_list(l: list, last_ind: int, first_ind: int=1):
    """
    Given a list [a, b, c, d, e, f, ...], (locally) inverts the list and returns the list slice of [first_ind, ..., last_ind] 
    of the *non-inverted* list. 

    E.g. for first_ind=1 and last_ind=4, returns [e, d, c, b]

    If last_ind=-1, all elements after (and including) first_ind are used. This is equivalent to using last_ind=len(l)-1

    """
    n = len(l)
    if last_ind == -1:
        last_ind = n-1
    l_copy = list(reversed(l)) # to avoid altering the original list, make a local copy
    if first_ind == 0:
        return l_copy[n-1-last_ind:]
    else: 
        return l_copy[n-1-last_ind: -first_ind]
    # l.reverse()
    # return l[n-1-last_ind: -first_ind]

def group_by_argmax(values: torch.Tensor, groups: torch.Tensor):
    """
    Groups the values by group, determines the max for each group and returns the index of those values.

    Args:
        values: Values of which we want to find the max-per-group
        group: Tensor indicating which value belongs to which group

    Returns:
        list of indices of the max value for each group
    """
    # map group ids to consecutive values  0..num_unique-1
    unique_groups, inverse = torch.unique(groups, return_inverse=True)
    num_ids = len(unique_groups)

    # compute max value per group
    max_vals = torch.full((num_ids,), float('-inf'), device=values.device)
    max_vals = max_vals.scatter_reduce(
        0,
        inverse,
        values,
        reduce="amax",
        include_self=True
    )

    # find indices where value is the max
    is_max_mask = (values == max_vals[inverse])

    idx = torch.arange(len(values), device=values.device) # full list of indices
    idx_masked = torch.where(is_max_mask, idx, torch.full_like(idx, -1)) # non-max indices are set to -1

    # filter out indices of max per group
    argmax_idx = torch.full((num_ids,), -1, dtype=torch.long, device=values.device)
    argmax_idx = argmax_idx.scatter_reduce(
        0,
        inverse,
        idx_masked,
        reduce="amax",
        include_self=True
    )
    
    return argmax_idx


# ----- Transforms -----

def affine(img: torch.Tensor, rot_angle_deg: float=0, fx: float=0, fy: float=0, scale: float=1.0, interpolation_mode: str='bilinear'):
    """
    Warps an image using an affine transformation.

    Args:
        img (torch.Tensor): image to be warped
        rot_angle_deg (float): angle by which to rotate the image (in degrees)
        fx (float): translation along the x axis, in percent of the image width
        fy (float): translation along the y axis, in percent of the image height
        scale (float): scale factor by which to scale the image
        interpolation_mode (str): mode to use for interpolation. Default: 'bilinear'

    Returns:
        torch.Tensor: warped image
    """
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
    return K.geometry.transform.warp_affine(img, matrix[:,:2,:], dsize=(H, W), mode=interpolation_mode)

def adjust_color(img: torch.Tensor, brightness: float=0.0, contrast: float=0.0, saturation: float=1.0):
    """
    Adjusts the brightness, contrast and saturation of an image.

    Args:
        img (torch.Tensor): image to be warped
        brightness (float): value by which to increase brightness. Default: 0
        contrast (float): value by which to adjust contrast. Default: 0
        saturation (float): value by which to adjust saturation. Default: 1

    Returns:
        torch.Tensor: color adjusted image
    """
    # brightness ∈ [-1, 1], contrast > 0
    img = K.enhance.adjust_brightness(img, brightness)
    img = K.enhance.adjust_contrast(img, contrast+1e-6)
    img = K.enhance.adjust_saturation(img, saturation)
    # K.enhance.adjust_gamma()
    return img

def gaussian_blur(img: torch.Tensor, kernel_size: float=0.0, sigma: float=0.0):
    """
    Blurs an image using gaussian blur

    Args:
        img (torch.Tensor): image to be warped
        kernel_size (float): size of blurring filter. Default: 0
        sigma (float): magnitude of blurring. Default: 0

    Returns:
        torch.Tensor: blurred image
    """
    # add batch dim if necessary
    if img.dim() == 3:
        img = img.unsqueeze(0)

    blur = K.filters.GaussianBlur2d((kernel_size, kernel_size), (sigma, sigma))

    return blur(img)

def add_gaussian_noise(img: torch.Tensor, sigma: float=0.0):
    """
    Adds gaussian noise to an image.

    Args:
        img (torch.Tensor): image to be warped
        sigma (float): magnitude of noise. Default: 0

    Returns:
        torch.Tensor: noisy image
    """
    noise = torch.randn_like(img)  
    if sigma != 1.0:  
        noise *= sigma
    return (img + noise).clamp(0,1)


def transform_img(img: torch.Tensor, transform_name: str, magnitude: float):
    """
    Applies the indicated transformation of the given magnitude to the image.

    Args:
        img (torch.Tensor): image to transform
        transform_name (str): indicates what type of transform to apply. One of 'Rotation', 'Translation', 'Scale', 'Brightness', 'Contrast', 'Saturation', 'Gamma', 'Hue', 'Gaussian Noise', 'Gaussian Blur'
        magnitude (float): magnitude/intensity of the transformation

    Returns:
        torch.Tensor: transformed image
    """
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
   
    return img


def get_affine_transform(center: torch.Tensor, rot_angle_deg: float=0, tx: float=0, ty: float=0, scale: float=1.0, B: int=1, device: str='cpu', dtype=torch.float32):
    """
    Determines the affine transformation matrix that will apply the specified affine transformation.

    Args:
        center: center around which to rotate
        rot_angle_deg: angle (in degrees) by which to rotate
        tx: translation along x axis
        ty: translation along y axis 
        scale: scaling factor
        B: desired batch dimension
        device: device of transformation matrix
        dtype: dtype of transformation matrix

    Returns:
        affine transformation matrix
    """
    angle = torch.tensor([rot_angle_deg], dtype=dtype, device=device).repeat(B)
    scale = torch.tensor([scale, scale], dtype=dtype, device=device).repeat(B,1)
    translation = torch.tensor([[tx, ty]], dtype=dtype, device=device).repeat(B,1)
    return K.geometry.transform.get_affine_matrix2d(translation, center, scale, angle)

def affine_warp_points(points: torch.Tensor, M: torch.Tensor):
    """Apply transformation M to points"""
    return K.geometry.linalg.transform_points(M, points)

def affine_warp_image(img: torch.Tensor, matrix: torch.Tensor, output_size: Tuple[int, int], interpolation: str="bilinear"):
    """ 
    Apply affine transformation matrix to image.

    Args:
        img: image to transform
        matrix: affine transformation matrix
        output_size: size of output image 
        interpolation: mode for interpolation. Default: "bilinear"

    Returns:
        torch.Tensor: transformed image
    """
    return K.geometry.transform.warp_affine(img, matrix[:,:2,:], dsize=output_size, mode=interpolation)

def compute_transformed_bounds(matrix: torch.Tensor, H: int, W: int):
    """
    Computes the bounds of of an image of size (H,W) after applying an affine transformation.
    """
    B = matrix.shape[0]
    device = matrix.device

    # corner tensor
    corners = torch.tensor(
        [[0, 0], [W, 0], [W, H], [0, H]],
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0).repeat(B, 1, 1)  # (B,4,2)

    # transform corners
    warped_corners = affine_warp_points(corners, matrix)

    xmin = warped_corners[..., 0].min(dim=1).values
    ymin = warped_corners[..., 1].min(dim=1).values
    xmax = warped_corners[..., 0].max(dim=1).values
    ymax = warped_corners[..., 1].max(dim=1).values

    return xmin, ymin, xmax, ymax

def expand_affine_from_corners(matrix: torch.Tensor, H: int, W: int):
    """
    Updates an affine transformation (and image size) so that no image content falls outside of the image canvas by tracking corner movement.
    
    Args:
        matrix: affine transformation matrix to apply (and update)
        H: height of image to transform
        W: width of image to transform

    Returns:
        updated affine transformation matrix
        new height
        new width
    """
    
    xmin, ymin, xmax, ymax = compute_transformed_bounds(matrix, H, W)

    new_W = torch.ceil(xmax - xmin).to(torch.int64)
    new_H = torch.ceil(ymax - ymin).to(torch.int64)

    # translation to shift into positive canvas
    shift = torch.zeros_like(matrix)
    shift[:, 0, 2] = -xmin
    shift[:, 1, 2] = -ymin
    shift[:, 2, 2] = 1.0

    B = matrix.shape[0]
    shift = torch.eye(3, device=matrix.device, dtype=matrix.dtype).unsqueeze(0).repeat(B, 1, 1)
    shift[:, 0, 2] = -xmin
    shift[:, 1, 2] = -ymin

    matrix_new = shift @ matrix

    return matrix_new, new_H, new_W

def extract_mask_boundary(mask: torch.Tensor):
    """
    Extract the boundary points of a mask.
    
    Args:
        mask: mask whose boundary should be extracted. Shape: (B,1,H,W)
    
    Returns: 
        list of (Ni,2) tensors of boundary point coordinates
    """
    B = mask.shape[0]
    boundaries = []

    # simple morphological gradient
    kernel = torch.ones(3, 3, device=mask.device)

    eroded = K.morphology.erosion(mask.float(), kernel) # shrink mask a little
    boundary = mask - eroded  # only edges remain

    # create list of boundary points
    for b in range(B):
        ys, xs = torch.where(boundary[b, 0] > 0)
        pts = torch.stack([xs, ys], dim=1).float()
        boundaries.append(pts)

    return boundaries

def expand_affine_from_contour(matrix: torch.Tensor, mask: torch.Tensor):
    """
    Updates an affine transformation (and image size) so that no image content falls outside of the image canvas by tracking movement of boundary points. The image canvas is expanded so that all foreground pixels are preserved.
    
    Args:
        matrix: affine transformation matrix to apply (and update). Shape: (B,3,3)
        mask: mask indicating foreground and background. Shape: (B, 1, H, W)

    Returns:
        updated affine transformation matrix
        new height
        new width
    """
    B = matrix.shape[0]
    device = matrix.device
    dtype = matrix.dtype

    # get contours for each mask
    contours = extract_mask_boundary(mask)

    xmin_list, ymin_list = [], []
    xmax_list, ymax_list = [], []

    # iterate through batch
    for b in range(B):
        pts = contours[b]

        pts = pts.unsqueeze(0)  # (1,N,2)
        matrix_batch = matrix[b:b+1]

        # warp points according to transformation
        warped = K.geometry.linalg.transform_points(matrix_batch, pts)[0]

        # save min/max in x/y dimension
        xmin_list.append(warped[:, 0].min())
        ymin_list.append(warped[:, 1].min())
        xmax_list.append(warped[:, 0].max())
        ymax_list.append(warped[:, 1].max())

    xmin = torch.stack(xmin_list)
    ymin = torch.stack(ymin_list)
    xmax = torch.stack(xmax_list)
    ymax = torch.stack(ymax_list)

    # compute height and width after transformation
    new_W = torch.ceil(xmax - xmin).to(torch.int64)
    new_H = torch.ceil(ymax - ymin).to(torch.int64)

    # create proper translation
    shift = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).repeat(B, 1, 1)
    shift[:, 0, 2] = -xmin
    shift[:, 1, 2] = -ymin

    matrix_new = shift @ matrix # add translation to affine warp

    return matrix_new, new_H, new_W

def expand_affine_from_joint_contour(matrix: torch.Tensor, masks: List[torch.Tensor]):
    """
    Updates an affine transformation (and image size) so that no image content falls outside of the image canvas for any of the listed masks by tracking movement of boundary points. The image canvas is expanded so that all foreground pixels are preserved.
    
    Args:
        matrix: affine transformation matrix to apply (and update). Shape: (B,3,3)
        masks: list of masks [(B,1,H,W), ...] indicating foreground and background of individual images of the same shape

    Returns:
        updated affine transformation matrix
        new height
        new width
    """

    B = matrix.shape[0]
    device = matrix.device
    dtype = matrix.dtype

    xmin_all, ymin_all = [], []
    xmax_all, ymax_all = [], []

    # collect all ROI points per batch
    pts_per_batch = [[] for _ in range(B)]

    for mask in masks:
        # get boundary points
        pts_list = extract_mask_boundary(mask)
        for b in range(B):
            pts_per_batch[b].append(pts_list[b])

    for b in range(B):
        pts = torch.cat(pts_per_batch[b], dim=0)  # union of ROIs

        pts = pts.unsqueeze(0)  # (1,N,2)
        warped = K.geometry.linalg.transform_points(matrix[b:b+1], pts)[0]

        # save min/max in x/y dimension
        xmin_all.append(warped[:, 0].min() - 0.5)
        ymin_all.append(warped[:, 1].min() - 0.5)
        xmax_all.append(warped[:, 0].max() + 0.5)
        ymax_all.append(warped[:, 1].max() + 0.5)

    xmin = torch.stack(xmin_all)
    ymin = torch.stack(ymin_all)
    xmax = torch.stack(xmax_all)
    ymax = torch.stack(ymax_all)

    # compute height and width after transformation
    new_W = torch.ceil(xmax - xmin).to(torch.int64)
    new_H = torch.ceil(ymax - ymin).to(torch.int64)


    # create proper translation
    shift = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).repeat(B, 1, 1)
    shift[:, 0, 2] = -xmin
    shift[:, 1, 2] = -ymin

    matrix_new = shift @ matrix # add translation to affine warp

    return matrix_new, new_H, new_W

def affine_warp_expand(imgs: List[torch.Tensor], masks: List[torch.Tensor]=None, pts_list: List[torch.Tensor]=None, rot_angle_deg: float=0, fx: float=0, fy: float=0, scale: float=1.0, return_matrix: bool=False):
    """
    Applies the specified transformation to all images so that no image content is lost. If masks are provided, the image canvas is expanded so that all foreground is preserved. Otherwise the canvas is expanded based on the movement of image corners.
    If masks are provided, the same transformation is applied to them using nearest-neighbor interpolation.
    If points are provided, the same transformation is applied.
    For images, masks, and points either a single tensor or a list of tensors can be provided. If a list is provided, the image canvas is (collectively) expanded so that the foreground of all images is maintained
    
    Args:
        imgs: single image tensor (B,3,H,W) or list of images [(B,3,H,W), ...] of the same shape
        masks: Optional. single tensor (B,1,H,W) or list of masks [(B,1,H,W), ...] indicating foreground and background of the images
        pts_list: Optional. single tensor or list of tensors or point coordinates.
        rot_angle_deg: angle by which to rotate the image (in degrees)
        fx: translation along the x axis, in percent of the image width
        fy: translation along the y axis, in percent of the image height
        scale: scale factor by which to scale the image
        return_matrix: if true, the transformation matrix is returned with the keyword 'matrix'

    Returns:
        dictionary containing all outputs
            - 'imgs': warped images
            - 'masks': warped masks (only returned if masks were provided)
            - 'pts': warped points (only returned if points were provided)
            - 'matrix': affine transformation matrix that produced the warped images etc. (only returned if return_matrix==True)
    """
    
    # add batch dim if necessary
    if type(imgs) == torch.Tensor:
        imgs = [imgs]
    for img in imgs:
        if img.dim() == 3:
            img = img.unsqueeze(0)
    if type(masks) == torch.Tensor:
        masks = [masks]
    if type(pts_list) == torch.Tensor:
        pts_list = [pts_list]
    
    B,C,H,W = imgs[0].shape
    device = imgs[0].device
    dtype = imgs[0].dtype

    # set up affine transformation matrix
    tx = W*fx/100
    ty = H*fy/100
    angle = torch.tensor([rot_angle_deg], dtype=dtype, device=device).repeat(B)
    scale = torch.tensor([scale, scale], dtype=dtype, device=device).repeat(B,1)
    center = torch.tensor([[img[0].shape[-1]/2, img[0].shape[-2]/2]], dtype=dtype, device=device).repeat(B,1)
    translation = torch.tensor([[tx, ty]], dtype=dtype, device=device).repeat(B,1)
    matrix = K.geometry.transform.get_affine_matrix2d(translation, center, scale, angle)

    # determine new matrix and canvas size
    if masks is not None:
        new_matrix, new_H, new_W = expand_affine_from_joint_contour(matrix, masks)
    else:
        new_matrix, new_H, new_W = expand_affine_from_corners(matrix, H, W)

    output_size = (new_H,new_W)

    # warp images
    imgs_out = []
    for img in imgs:
        imgs_out.append( affine_warp_image(img, new_matrix, output_size) )
    if len(imgs_out)==1:
        imgs_out = imgs_out[0]
    out_dict = {"imgs": imgs_out}

    # warp masks
    if masks is not None:
        masks_out = []
        for mask in masks:
            masks_out.append( affine_warp_image(mask, new_matrix, output_size, interpolation="nearest") )
        if len(masks_out)==1:
            masks_out = masks_out[0]
        out_dict.update({"masks": masks_out})

    # warp points
    if pts_list is not None:
        pts_out = []
        for pts in pts_list:
            pts_out.append( affine_warp_points(pts, new_matrix) )
        if len(pts_out)==1:
            pts_out = pts_out[0]
        out_dict.update({"pts": pts_out})

    if return_matrix:
        out_dict.update({"matrix": new_matrix})

    return out_dict


def find_roi_rotation(mask: torch.Tensor):
    """Fits a rotated minimum-area bounding rectangle to the ROI/foreground of the mask and returns the angle of rotation"""
    bdry_pts = extract_mask_boundary(mask)[0] # get list of boundary points
    bdry_pts_np = bdry_pts.cpu().numpy().astype(np.float32) # convert to np for opencv

    # get the minimum-area bounding rectangle
    rect = cv2.minAreaRect(bdry_pts_np) # ((cx, cy), (y, x), angle)
    # angle is in degrees, range [-90, 0)


    # angle = rect[2]
    # y,x = rect[1]
    # cx, cy = rect[0]
    (cx, cy), (w, h), angle = rect
    if h > w:
        angle += 90  # rotate so long side along x-axis

    return torch.tensor(angle)


def check_orientation(kpts1: torch.Tensor, kpts2: torch.Tensor, num_samples: int = 100, threshold: float=0.3):
    """
    Heuristic to determine whether a 180° rotation is present between images. Exploits that LoFTR returns poor quality inconsistent matches by analyzing the orientation of sampled triangles between images. 

    Args:
        kpts1 (Tensor): list of keypoint coordinates in the first image
        kpts2 (Tensor): list of keypoint coordinates in the second image
        num_samples (int): number of triangles to sample
        threshold (float): threshold below which we suspect a rotation between images. 

    Returns:
        bool: If True, the images have the same orientation with no rotation between them. If False, we suspect there's a 180° rotation between images. 
    """
    N = kpts1.shape[0]
    if N < 3:
        return 0.0

    device = kpts1.device

    # choose index triplets
    if N <= 25:
        idx = torch.tensor(list(combinations(range(N), 3)), device=device)
    else:
        idx = torch.rand(num_samples, N, device=device).topk(3, dim=1).indices # generates random numbers and returns indices of top 3 largest


    # get triangle coords
    a1, b1, c1 = kpts1[idx[:,0]], kpts1[idx[:,1]], kpts1[idx[:,2]] # triangle vertices in img0
    a2, b2, c2 = kpts2[idx[:,0]], kpts2[idx[:,1]], kpts2[idx[:,2]] # triangle vertices in img1

    def signed_area(a, b, c):
        # cross product (b-a) x (c-a)
        return (b[:,0]-a[:,0])*(c[:,1]-a[:,1]) - \
               (b[:,1]-a[:,1])*(c[:,0]-a[:,0])

    # compute signed triangle area
    s1 = signed_area(a1, b1, c1)
    s2 = signed_area(a2, b2, c2)

    # filter based on scale 
    scale = kpts1.std() + kpts2.std()
    eps = 1e-4 * scale # set min area threshold dependent on spread of keypoints

    valid = (s1.abs() > eps) & (s2.abs() > eps)

    if valid.sum() == 0:
        return 0.0

    # filter out degenerate triangles (e.g. colinear vertices)
    s1 = s1[valid]
    s2 = s2[valid]

    agreement = torch.sign(s1) * torch.sign(s2)

    # weighting by area
    weights = (s1.abs() + s2.abs())

    score = (agreement * weights).sum() / weights.sum()

    if score.item() < threshold:
        print("Rotation between Images suspected.")
    return (score.item() >= threshold)


class RandomHomography:
    """
    Class for generating a random homography, then applying said homography to images, masks or points.
    """
    def __init__(
        self,
        height: int,
        width: int,
        distortion_scale: float=0.5,
        device: str="cpu",
        dtype=torch.float32,
    ):
        """
        Initializes class by generating a random homography.

        Args:
            height: height of image to be transformed
            width:  width of image to be transformed
            distortion_scale: degree of the distortion
            device: device of homography
            dtype: dtype of homography
        """
        self.height = height
        self.width = width

        self.aug = K.augmentation.RandomPerspective(
            distortion_scale=distortion_scale,
            p=1.0,
            same_on_batch=True,
            keepdim=True,
        ).to(device=device, dtype=dtype)

        # sample once on dummy input to fix parameters
        dummy = torch.zeros(1, 1, height, width, device=device, dtype=dtype)
        self.aug(dummy)
        self.params = self.aug._params

        self.H, self.H_inv = self._compute_matrices()
        

    def _compute_matrices(self):
        """Computes transformation matrix based on the transformation parameters, as well as its inverse."""
        start = self.params["start_points"]  # (1, 4, 2)
        end = self.params["end_points"]      # (1, 4, 2)

        H = K.geometry.transform.get_perspective_transform(start, end)
        H_inv = torch.inverse(H)

        return H, H_inv

    # -----------------------
    # Image warping
    # -----------------------
    def warp_image(self, x: torch.Tensor):
        """Apply homography to an image"""
        return self.aug(x, params=self.params)

    def warp_image_inverse(self, x: torch.Tensor):
        """Apply inverse of homography to an image"""
        return self.aug.inverse(x, params=self.params)


    # -----------------------
    # Mask warping
    # -----------------------
    def warp_mask(self, mask: torch.Tensor):
        """Apply homography to a mask. Uses nearest neighbor interpolation."""
        return KT.warp_perspective(
            mask.float(), # warp expects float
            self.H,
            dsize=(self.height, self.width),
            mode="nearest",                   
            align_corners=False,
        )#.long()

    def warp_mask_inverse(self, mask: torch.Tensor):
        """Apply inverse homography to a mask. Uses nearest neighbor interpolation."""
        return KT.warp_perspective(
            mask.float(),                      
            self.H_inv,
            dsize=(self.height, self.width),
            mode="nearest",                  
            align_corners=False,
        )#.long()

    # -----------------------
    # Points warping
    # -----------------------
    def warp_points(self, points: torch.Tensor):
        """Apply homography to an batch of points (B, N, 2)"""
        pts_dim = points.dim()
        if pts_dim == 2:
            points = points.unsqueeze(0)
        B = points.shape[0]
        H = self.H.expand(B, -1, -1)
        out = K.geometry.linalg.transform_points(H, points)
        if pts_dim == 2:
            out = out.squeeze(0)
        return out


    def warp_points_inverse(self, points: torch.Tensor):
        """Apply inverse homography to an batch of points (B, N, 2)"""
        pts_dim = points.dim()
        if pts_dim == 2:
            points = points.unsqueeze(0)
        B = points.shape[0]
        H = self.H_inv.expand(B, -1, -1)
        out = K.geometry.linalg.transform_points(H, points)
        if pts_dim == 2:
            out = out.squeeze(0)
        return out
        
    # matrices -----------------------
    def matrix(self):
        """Returns transformation matrix"""
        return self.H

    def inverse_matrix(self):
        """Returns inverse transformation matrix"""
        return self.H_inv