import kornia as K
import kornia.geometry.transform as KT
import numpy as np
import torch
from typing import List


def crop_coords_zero_borders(mask: torch.Tensor):
    """
    mask: (B, C, H, W), values 0/1 (or nonzero = foreground)
    returns: list of cropped tensors, one per batch element
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

    r_idx = torch.where(rows)[0]
    c_idx = torch.where(cols)[0]

    rmin, rmax = r_idx[0], r_idx[-1]
    cmin, cmax = c_idx[0], c_idx[-1]

    return rmin, rmax, cmin, cmax


def crop_coords_zero_borders_batch(mask: torch.Tensor):
    """
    mask: (B, C, H, W), values 0/1 (or nonzero = foreground)
    returns: list of cropped tensors, one per batch element
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

def undo_rotation(img, M_cv2, interpolation_mode: str="bilinear"):
    """
    img: (B, C, H, W)
    M_cv2: (2, 3) affine used previously with cv2.warpAffine
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
    if (mask1 is None) != (mask2 is None):
        raise ValueError("Provide both mask1 and mask2, or neither.")
    
    # resize
    height = max(img1.shape[-2], img2.shape[-2])
    width = max(img1.shape[-1], img2.shape[-1])

    padder = K.augmentation.PadTo((height, width))

    img1 = padder(img1)
    img2 = padder(img2)   

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

    img1 = K.geometry.resize(img1, (H, W), antialias=True)
    img2 = K.geometry.resize(img2, (H, W), antialias=True)

    img1 = torch.clamp(img1, 0.0, 1.0)
    img2 = torch.clamp(img2, 0.0, 1.0)

    if mask1 is None:
        return img1, img2 

    mask1 = padder(mask1)
    mask2 = padder(mask2)

    mask1 = K.geometry.resize(mask1, (H, W), antialias=False, interpolation='nearest')
    mask2 = K.geometry.resize(mask2, (H, W), antialias=False, interpolation='nearest')

    return img1, img2, mask1, mask2

def match_sizes_resize_batch(imgs: List[torch.Tensor], masks: List[torch.Tensor]=None, size_factor: int=None):
    # filter out None tensors for size computation
    valid_imgs = [img for img in imgs if img is not None]
    if len(valid_imgs) == 0:
        # if all images are None, can just return them as is
        return imgs if masks is None else (imgs, masks)
    
    heights = [img.shape[-2] for img in valid_imgs]
    widths = [img.shape[-1] for img in valid_imgs]

    height = max(heights)
    width = max(widths)

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

    for i, img in enumerate(imgs):
        if img is None: # skip None images
            continue

        img = K.geometry.resize(img, (H, W), antialias=True)
        img = torch.clamp(img, 0.0, 1.0)
        imgs[i] = img

    if masks is None:
        return imgs

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

def pil_to_kornia(pil_img):
    np_img = np.array(pil_img)
    tensor_img = K.image_to_tensor(np_img).float() / 255.0 
    return tensor_img.unsqueeze(0)


def convert_image_to_tensor(img):
    if img is None:
        return img
    elif type(img) == torch.Tensor:
        if img.dim == 3:
            img.unsqueeze(0)
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

def convert_img_tensor_to_numpy(img_tensor):
    img_np = (
        img_tensor[0]                      # select batch
        .permute(1, 2, 0)         # CHW → HWC (or squeeze for grayscale)
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32) 
    )
    return img_np

def weighted_average(tensor, weights):
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
    # K.enhance.adjust_gamma()
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


def get_affine_transform(center, rot_angle_deg=0, tx=0, ty=0, scale=1.0, B=1, device='cpu', dtype=torch.float32):
    angle = torch.tensor([rot_angle_deg], dtype=dtype, device=device).repeat(B)
    scale = torch.tensor([scale, scale], dtype=dtype, device=device).repeat(B,1)
    translation = torch.tensor([[tx, ty]], dtype=dtype, device=device).repeat(B,1)
    return K.geometry.transform.get_affine_matrix2d(translation, center, scale, angle)

def affine_warp_points(points, M):

    return K.geometry.linalg.transform_points(M, points)

def affine_warp_image(img, matrix, output_size, interpolation="bilinear"):

    return K.geometry.transform.warp_affine(img, matrix[:,:2,:], dsize=output_size, mode=interpolation)

def compute_transformed_bounds(matrix, H, W):
    B = matrix.shape[0]
    device = matrix.device

    corners = torch.tensor(
        [[0, 0], [W, 0], [W, H], [0, H]],
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0).repeat(B, 1, 1)  # (B,4,2)

    warped_corners = affine_warp_points(corners, matrix)

    xmin = warped_corners[..., 0].min(dim=1).values
    ymin = warped_corners[..., 1].min(dim=1).values
    xmax = warped_corners[..., 0].max(dim=1).values
    ymax = warped_corners[..., 1].max(dim=1).values

    return xmin, ymin, xmax, ymax

def expand_affine_from_corners(matrix, H, W):
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

def extract_mask_boundary(mask):
    """
    mask: (B,1,H,W)
    returns: list of (Ni,2) tensors
    """
    B = mask.shape[0]
    boundaries = []

    # simple morphological gradient
    kernel = torch.ones(3, 3, device=mask.device)

    eroded = K.morphology.erosion(mask.float(), kernel)
    boundary = mask - eroded  # edges

    for b in range(B):
        ys, xs = torch.where(boundary[b, 0] > 0)
        pts = torch.stack([xs, ys], dim=1).float()
        boundaries.append(pts)

    return boundaries

def subsample_points(points, max_points=500):
    if points.shape[0] <= max_points:
        return points
    idx = torch.randperm(points.shape[0], device=points.device)[:max_points]
    return points[idx]

def expand_affine_from_contour(A, mask):
    B = A.shape[0]
    device = A.device
    dtype = A.dtype

    contours = extract_mask_boundary(mask)

    xmin_list, ymin_list = [], []
    xmax_list, ymax_list = [], []

    for b in range(B):
        # pts = subsample_points(contours[b])  # (N,2)
        pts = contours[b]

        pts = pts.unsqueeze(0)  # (1,N,2)
        Ab = A[b:b+1]

        # warp points according to transformation
        warped = K.geometry.linalg.transform_points(Ab, pts)[0]

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

    A_new = shift @ A # add translation to affine warp

    return A_new, new_H, new_W

def expand_affine_from_joint_contour(A, masks):
    """
    A: (B,3,3)
    masks: list of masks [(B,1,H,W), ...]

    returns xmin, ymin, xmax, ymax (B,)
    """
    B = A.shape[0]
    device = A.device
    dtype = A.dtype

    xmin_all, ymin_all = [], []
    xmax_all, ymax_all = [], []

    # collect all ROI points per batch
    pts_per_batch = [[] for _ in range(B)]

    for mask in masks:
        pts_list = extract_mask_boundary(mask)
        for b in range(B):
            pts_per_batch[b].append(pts_list[b])

    for b in range(B):
        pts = torch.cat(pts_per_batch[b], dim=0)  # union of ROIs

        pts = pts.unsqueeze(0)  # (1,N,2)
        warped = K.geometry.linalg.transform_points(A[b:b+1], pts)[0]

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

    A_new = shift @ A # add translation to affine warp

    return A_new, new_H, new_W

# def affine_warp_expand(img, mask: torch.Tensor=None, pts: torch.Tensor=None, rot_angle_deg=0, fx=0, fy=0, scale=1.0):
#     # add batch dim if necessary
#     if img.dim() == 3:
#         img = img.unsqueeze(0)
    
#     B,C,H,W = img.shape
#     device = img.device
#     tx = W*fx/100
#     ty = H*fy/100
#     angle = torch.tensor([rot_angle_deg], dtype=img.dtype, device=img.device).repeat(B)
#     scale = torch.tensor([scale, scale], dtype=img.dtype, device=img.device).repeat(B,1)
#     center = torch.tensor([[img.shape[-1]/2, img.shape[-2]/2]], dtype=img.dtype, device=img.device).repeat(B,1)
#     translation = torch.tensor([[tx, ty]], dtype=img.dtype, device=img.device).repeat(B,1)
#     matrix = K.geometry.transform.get_affine_matrix2d(translation, center, scale, angle)


#     if mask is not None:
#         matrix, new_H, new_W = expand_affine_from_contour(matrix, mask)
#     else:
#         matrix, new_H, new_W = expand_affine_from_corners(matrix, H, W)
#     output_size = (new_H,new_W)

#     img_warped = affine_warp_image(img, matrix, output_size) #K.geometry.transform.warp_affine(img, matrix[:,:2,:], dsize=output_size)
#     out_dict = {"img": img_warped}
#     if mask is not None:
#         mask_warped = affine_warp_image(mask, matrix, output_size, interpolation="nearest")
#         out_dict.update({"mask": mask_warped})

#     if pts is not None:
#         pts_warped = affine_warp_points(pts, matrix)
#         out_dict.update({"pts": pts_warped})

#     return out_dict

def affine_warp_expand(imgs: torch.Tensor, masks: torch.Tensor=None, pts_list: List[torch.Tensor]=None, rot_angle_deg=0, fx=0, fy=0, scale=1.0, return_matrix: bool=False):
    # add batch dim if necessary
    if type(imgs) == torch.Tensor:
        imgs = [imgs]
    for img in imgs:
        if img.dim() == 3:
            img = img.unsqueeze(0)
    if type(masks) == torch.Tensor:
        masks = [masks]
    
    B,C,H,W = imgs[0].shape
    device = imgs[0].device
    dtype = imgs[0].dtype
    tx = W*fx/100
    ty = H*fy/100
    angle = torch.tensor([rot_angle_deg], dtype=dtype, device=device).repeat(B)
    scale = torch.tensor([scale, scale], dtype=dtype, device=device).repeat(B,1)
    center = torch.tensor([[img[0].shape[-1]/2, img[0].shape[-2]/2]], dtype=dtype, device=device).repeat(B,1)
    translation = torch.tensor([[tx, ty]], dtype=dtype, device=device).repeat(B,1)
    matrix = K.geometry.transform.get_affine_matrix2d(translation, center, scale, angle)


    if masks is not None:
        new_matrix, new_H, new_W = expand_affine_from_joint_contour(matrix, masks)
    else:
        new_matrix, new_H, new_W = expand_affine_from_corners(matrix, H, W)

    output_size = (new_H,new_W)

    imgs_out = []
    for img in imgs:
        imgs_out.append( affine_warp_image(img, new_matrix, output_size) )
    if len(imgs_out)==1:
        imgs_out = imgs_out[0]
    out_dict = {"imgs": imgs_out}

    if masks is not None:
        masks_out = []
        for mask in masks:
            masks_out.append( affine_warp_image(mask, new_matrix, output_size, interpolation="nearest") )
        if len(masks_out)==1:
            masks_out = masks_out[0]
        out_dict.update({"masks": masks_out})

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