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


