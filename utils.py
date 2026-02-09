import kornia as K
import kornia.geometry.transform as KT
import numpy as np
import torch


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


