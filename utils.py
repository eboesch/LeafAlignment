import os
import cv2
import kornia as K
import kornia.feature as KF
import kornia.geometry.transform as KT
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import torch
import torch.nn.functional as F
from kornia_moons.viz import draw_LAF_matches
import skimage as ski
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import normalized_mutual_information as nmi


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
    if type(img) == torch.Tensor:
        return img
    elif type(img) == np.ndarray:
        return K.image_to_tensor(img)
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

# ------------- metrics ------------------------------------------

def mse(img1, img2, reduction='mean'):
    img1 = convert_image_to_tensor(img1)
    img2 = convert_image_to_tensor(img2)

    mse_loss = torch.nn.MSELoss(reduction=reduction)
    return mse_loss(img1, img2)

def ncc(img1, img2, reduction='mean'):
    """
    Compute Normalized Cross-Correlation (NCC) between two images.

    Args:
        img1, img2: Input images (H x W x C or tensors).
        reduction: 'mean' for scalar NCC over the entire image,
                   'none' for per-pixel NCC map.

    Returns:
        Scalar NCC if reduction='mean', else a tensor of the same shape as input.
    """
    img1 = convert_image_to_tensor(img1)
    img2 = convert_image_to_tensor(img2)

    # Convert to grayscale
    img1 = K.color.rgb_to_grayscale(img1)
    img2 = K.color.rgb_to_grayscale(img2)

    img1_centered = img1 - img1.mean()
    img2_centered = img2 - img2.mean()
    std1 = torch.std(img1)
    std2 = torch.std(img2)

    if reduction == 'mean':
        # Full-image NCC        
        return torch.sum(img1_centered * img2_centered) / (std1 * std2 * torch.numel(img1) + 1e-13)

    elif reduction == 'none':
        # Per-pixel NCC map
        img1_norm = img1_centered / (std1 + 1e-13)
        img2_norm = img2_centered / (std2 + 1e-13)
        return img1_norm * img2_norm

    else:
        raise ValueError("reduction must be either 'mean' or 'none'")

def local_ncc(img1, img2, window_size=9, reduction='mean'):
    """
    Compute Local Normalized Cross-Correlation (LNCC) between two images.

    Args:
        img1, img2: Input images (H x W x C or tensors).
        window_size: Size of local window (odd integer, e.g., 9).
        reduction: 'mean' for scalar LNCC over the entire image,
                   'none' for per-pixel LNCC map.

    Returns:
        Scalar LNCC if reduction='mean', else a tensor of per-pixel LNCC.
    """
    img1 = convert_image_to_tensor(img1)
    img2 = convert_image_to_tensor(img2)

    # Convert to grayscale if needed
    img1 = K.color.rgb_to_grayscale(img1)
    img2 = K.color.rgb_to_grayscale(img2)

    # Ensure batch and channel dims for conv2d
    if img1.dim() == 2:
        img1 = img1.unsqueeze(0).unsqueeze(0)
        img2 = img2.unsqueeze(0).unsqueeze(0)
    elif img1.dim() == 3:
        img1 = img1.unsqueeze(0)
        img2 = img2.unsqueeze(0)

    padding = window_size // 2
    conv_filter = torch.ones(1, 1, window_size, window_size, device=img1.device)

    # Local sums
    sum1 = F.conv2d(img1, conv_filter, padding=padding)
    sum2 = F.conv2d(img2, conv_filter, padding=padding)

    N = window_size ** 2
    mean1 = sum1 / N
    mean2 = sum2 / N

    # center
    img1_centered = img1 - mean1
    img2_centered = img2 - mean2

    # weighted second moments
    var1 = F.conv2d(img1_centered * img1_centered, conv_filter, padding=padding)
    var2 = F.conv2d(img2_centered * img2_centered, conv_filter, padding=padding)
    cov12 = F.conv2d(img1_centered * img2_centered, conv_filter, padding=padding)


    # Local NCC
    lcc_map = cov12 / (torch.sqrt(var1 * var2) + 1e-13)

    if reduction == 'mean':
        return lcc_map.mean()
    elif reduction == 'none':
        return lcc_map.squeeze(0).squeeze(0)  # remove batch/channel dims
    else:
        raise ValueError("reduction must be 'mean' or 'none'")


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
    # if type(img1) == np.ndarray:
    #     img1 = K.image_to_tensor(img1)
    # if type(img2) == np.ndarray:
    #     img2 = K.image_to_tensor(img2)
    img1 = convert_image_to_tensor(img1)
    img2 = convert_image_to_tensor(img2)

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


def ssim_kornia(img1, img2, window_size=11, reduction='mean'):
    img1 = convert_image_to_tensor(img1)
    img2 = convert_image_to_tensor(img2)


    ssim_map = K.metrics.ssim(img1, img2, window_size, eps=1e-12, padding='same', max_val=1.0)
    if reduction == 'mean':
        return ssim_map.mean(dim=(1, 2, 3)).item()
    elif reduction == 'none':
        return ssim_map
    else:
        raise ValueError("reduction must be either 'mean' or 'none'")


def ssim_skimage(img1, img2, window_size=11, return_img=False):
    img1 = convert_img_tensor_to_numpy(convert_image_to_tensor(img1))
    img2 = convert_img_tensor_to_numpy(convert_image_to_tensor(img2))
    
    data_range = 1.0
    sigma = 1.5
    K1, K2 = 0.01, 0.03

    return ssim(img1, img2, channel_axis=-1, data_range=data_range, win_size=window_size, gaussian_weights=True, sigma=sigma, K1=K1, K2=K2, full=return_img)