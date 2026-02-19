import kornia as K
# import numpy as np
import torch
import torch.nn.functional as F
# import skimage as ski
from skimage.metrics import structural_similarity
from skimage.metrics import normalized_mutual_information
from monai.metrics import compute_hausdorff_distance
from utils import convert_image_to_tensor, convert_img_tensor_to_numpy, weighted_average

def iou(img1, mask1, img2, mask2):
    mask1 = convert_image_to_tensor(mask1).long()
    mask2 = convert_image_to_tensor(mask2).long()
    return K.metrics.mean_iou(mask2, mask1, 2, eps=1e-6)[0,1]

def hausdorff(img1, img1_mask, img2, img2_mask, percentile=97):
    img1_mask = convert_image_to_tensor(img1_mask).long()
    img2_mask = convert_image_to_tensor(img2_mask).long()

    return compute_hausdorff_distance(
        y_pred=img2_mask,
        y=img1_mask,
        distance_metric="euclidean",
        percentile=percentile,
        include_background=False,
    )

def mse(img1, img2, reduction='mean'):
    img1 = convert_image_to_tensor(img1)
    img2 = convert_image_to_tensor(img2)

    mse_loss = torch.nn.MSELoss(reduction=reduction)
    return mse_loss(img1, img2)

def mse_masked(img1, img1_mask, img2, img2_mask, reduction="mean", mask_mode='both'):
    img1 = convert_image_to_tensor(img1)
    img2 = convert_image_to_tensor(img2)
    img1_mask = convert_image_to_tensor(img1_mask)
    img2_mask = convert_image_to_tensor(img2_mask)


    if mask_mode == 'either':
        # consider all pixels where at least one image is valid
        mask = torch.logical_or(img1_mask, img2_mask)
    elif mask_mode == 'both':
        # consider all pixels where both images are valid
        mask = torch.logical_and(img1_mask, img2_mask)
    else:
        raise ValueError(f"Unknown mask_mode {mask_mode}. Expected 'either' or 'both'.")

    if mask.shape[1] == 1:
        mask = K.color.grayscale_to_rgb(mask)
        
    # squared_diff = (img1)
    # mse_loss = torch.nn.MSELoss(reduction=reduction)
    return torch.nn.functional.mse_loss(img1, img2, reduction=reduction, weight=mask)

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
        

def ncc_masked(img1, img1_mask, img2, img2_mask, reduction="mean",  mask_mode='both'):
    img1 = convert_image_to_tensor(img1)
    img2 = convert_image_to_tensor(img2)
    img1_mask = convert_image_to_tensor(img1_mask)
    img2_mask = convert_image_to_tensor(img2_mask)

    # Convert to grayscale
    img1 = K.color.rgb_to_grayscale(img1)
    img2 = K.color.rgb_to_grayscale(img2)

    if mask_mode == 'either':
        # consider all pixels where at least one image is valid
        mask = torch.logical_or(img1_mask, img2_mask)
    elif mask_mode == 'both':
        # consider all pixels where both images are valid
        mask = torch.logical_and(img1_mask, img2_mask)
    else:
        raise ValueError(f"Unknown mask_mode {mask_mode}. Expected 'either' or 'both'.")

    if torch.sum(mask) == 0:
        raise ValueError(f"Combined mask is empty.")

    img1_mean = weighted_average(img1, mask)
    img2_mean = weighted_average(img2, mask)
    img1_centered = img1 - img1_mean
    img2_centered = img2 - img2_mean
    std1 = torch.sqrt(weighted_average(torch.pow(img1_centered, 2), mask))
    std2 = torch.sqrt(weighted_average(torch.pow(img2_centered, 2), mask))

    if reduction == 'mean':
        # Full-image NCC        
        return torch.sum(img1_centered * img2_centered * mask) / (std1 * std2 * torch.sum(mask) + 1e-13)

    elif reduction == 'none':
        # Per-pixel NCC map
        img1_norm = img1_centered / (std1 + 1e-13)
        img2_norm = img2_centered / (std2 + 1e-13)
        return img1_norm * img2_norm * mask

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
    elif img1.dim() == 3:
        img1 = img1.unsqueeze(0)
    if img2.dim() == 2:
        img2 = img2.unsqueeze(0).unsqueeze(0)
    elif img2.dim() == 3:
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


def local_ncc_masked(img1, img1_mask, img2, img2_mask, window_size=9, mask_mode='both', reduction='mean'):
    img1 = convert_image_to_tensor(img1)
    img2 = convert_image_to_tensor(img2)
    img1_mask = convert_image_to_tensor(img1_mask)
    img2_mask = convert_image_to_tensor(img2_mask)

    # Convert to grayscale
    img1 = K.color.rgb_to_grayscale(img1)
    img2 = K.color.rgb_to_grayscale(img2)

    # construct mask
    if mask_mode == 'either':
        # consider all pixels where at least one image is valid
        mask = torch.logical_or(img1_mask, img2_mask)
    elif mask_mode == 'both':
        # consider all pixels where both images are valid
        mask = torch.logical_and(img1_mask, img2_mask)
    else:
        raise ValueError(f"Unknown mask_mode {mask_mode}. Expected 'either' or 'both'.")

    if torch.sum(mask) == 0:
        raise ValueError(f"Combined mask is empty.")

    # cast mask to float
    mask = mask.to(dtype=img1.dtype)

    kernel = torch.ones(1,1,window_size, window_size, device=img1.device)
    padding = window_size//2

    # local sums
    img1_masked = mask * img1
    img2_masked = mask * img2

    sum_mask  = F.conv2d(mask,  kernel, padding=padding)
    sum1  = F.conv2d(img1_masked, kernel, padding=padding)
    sum2  = F.conv2d(img2_masked, kernel, padding=padding)
    
    # local means
    mean1 = sum1 / (sum_mask + 1e-13)
    mean2 = sum2 / (sum_mask + 1e-13)

    # center
    img1_centered = img1 - mean1
    img2_centered = img2 - mean2

    # weighted second moments
    var1 = F.conv2d(mask * img1_centered * img1_centered, kernel, padding=padding)
    var2 = F.conv2d(mask * img2_centered * img2_centered, kernel, padding=padding)
    cov = F.conv2d(mask * img1_centered * img2_centered, kernel, padding=padding)

    # local stds
    std1 = torch.sqrt(var1)
    std2 = torch.sqrt(var2)

    lncc_map = cov / (std1 * std2 + 1e-13)

    # only count pixels where there is enough valid support
    valid = (sum_mask > 0)

    if reduction == 'mean':
        return lncc_map[valid].mean()
    elif reduction == 'none':
        return lncc_map.squeeze(0).squeeze(0)  # remove batch/channel dims
    else:
        raise ValueError("reduction must be 'mean' or 'none'")


def nmi_skimage(img1, img2, bins=100):
    img1 = convert_img_tensor_to_numpy(convert_image_to_tensor(img1))
    img2 = convert_img_tensor_to_numpy(convert_image_to_tensor(img2))

    return normalized_mutual_information(img1, img2, bins=bins)


def nmi(img1, img2, reduction="mean", bins=32, sigma_ratio=1.0, sigma=None, eps=1e-13):
    """
    Parzen-window Normalized Mutual Information (NMI)

    A, B : tensors of same shape, intensities in [0,1]
    W    : same shape, 1 = valid pixel, 0 = invalid
    bins : number of Parzen bins
    sigma: Gaussian kernel width

    Returns: scalar NMI
    """

    img1 = convert_image_to_tensor(img1)
    img2 = convert_image_to_tensor(img2)

    # Convert to grayscale
    img1 = K.color.rgb_to_grayscale(img1)
    img2 = K.color.rgb_to_grayscale(img2)

    # flatten
    img1 = img1.reshape(-1)
    img2 = img2.reshape(-1)

    device = img1.device

    # bin centers
    # shape: (1, bins)
    bin_centers = torch.linspace(0.0, 1.0, bins, device=device).unsqueeze(0)
    bin_width = torch.mean(torch.diff(bin_centers))
    if sigma is None:
        sigma = bin_width * sigma_ratio

    # Parzen soft assignment
    # shape: (N, bins)
    img1 = img1.unsqueeze(1)
    img2 = img2.unsqueeze(1)

    hist1 = torch.exp(-(img1 - bin_centers)**2 / (2 * sigma**2))
    hist2 = torch.exp(-(img2 - bin_centers)**2 / (2 * sigma**2))
    # normalize kernels
    hist1 = hist1 / (hist1.sum(dim=1, keepdim=True) + eps)
    hist2 = hist2 / (hist2.sum(dim=1, keepdim=True) + eps)
    # print(f"P1 alt: {torch.mean(hist1, dim=0)}")

    # joint Parzen histogram
    # shape: (bins, bins)
    P12 = hist1.T @ hist2
    
    # normalize to probabilities
    P12 = P12 / (P12.sum() + eps)

    # marginals
    P1 = torch.mean(hist1, dim=0)
    P2 = torch.mean(hist2, dim=0)    

    # entropies
    H1  = -(P1  * torch.log(P1  + eps)).sum()
    H2  = -(P2  * torch.log(P2  + eps)).sum()
    H12 = -(P12 * torch.log(P12 + eps)).sum()

    # normalized mutual information
    NMI = (H1 + H2) / H12

    return NMI


def nmi_masked(img1, img1_mask, img2, img2_mask, reduction="mean", mask_mode='both', bins=32, sigma_ratio=1.0, sigma=None, eps=1e-7):
    """
    Parzen-window Normalized Mutual Information (NMI)

    A, B : tensors of same shape, intensities in [0,1]
    W    : same shape, 1 = valid pixel, 0 = invalid
    bins : number of Parzen bins
    sigma: Gaussian kernel width

    Returns: scalar NMI
    """

    img1 = convert_image_to_tensor(img1)
    img2 = convert_image_to_tensor(img2)
    img1_mask = convert_image_to_tensor(img1_mask)
    img2_mask = convert_image_to_tensor(img2_mask)

    # Convert to grayscale
    img1 = K.color.rgb_to_grayscale(img1)
    img2 = K.color.rgb_to_grayscale(img2)

    if mask_mode == 'either':
        # consider all pixels where at least one image is valid
        mask = torch.logical_or(img1_mask, img2_mask)
    elif mask_mode == 'both':
        # consider all pixels where both images are valid
        mask = torch.logical_and(img1_mask, img2_mask)
    else:
        raise ValueError(f"Unknown mask_mode {mask_mode}. Expected 'either' or 'both'.")

    if torch.sum(mask) == 0:
        raise ValueError(f"Combined mask is empty.")

    # flatten
    img1 = img1.reshape(-1)
    img2 = img2.reshape(-1)
    mask = mask.reshape(-1)

    device = img1.device

    # bin centers
    # shape: (1, bins)
    bin_centers = torch.linspace(0.0, 1.0, bins, device=device).unsqueeze(0)
    bin_width = torch.mean(torch.diff(bin_centers))
    if sigma is None:
        sigma = bin_width * sigma_ratio

    # Parzen soft assignment
    # shape: (N, bins)
    img1 = img1.unsqueeze(1)
    img2 = img2.unsqueeze(1)
    mask = mask.unsqueeze(1)

    hist1 = torch.exp(-(img1 - bin_centers)**2 / (2 * sigma**2))
    hist2 = torch.exp(-(img2 - bin_centers)**2 / (2 * sigma**2))

    # normalize kernels
    hist1 = hist1 / (hist1.sum(dim=1, keepdim=True) + eps)
    hist2 = hist2 / (hist2.sum(dim=1, keepdim=True) + eps)

    # apply mask 
    hist1 = hist1 * mask
    hist2 = hist2 * mask

    # joint Parzen histogram
    # shape: (bins, bins)
    P12 = hist1.T @ hist2

    # normalize to probabilities
    P12 = P12 / (P12.sum() + eps)

    # marginals
    P1 = torch.sum(hist1, dim=0) / torch.sum(mask) 
    P2 = torch.sum(hist2, dim=0) / torch.sum(mask)    

    # entropies
    H1  = -(P1  * torch.log(P1  + eps)).sum()
    H2  = -(P2  * torch.log(P2  + eps)).sum()
    H12 = -(P12 * torch.log(P12 + eps)).sum()

    # normalized mutual information
    NMI = (H1 + H2) / H12

    return NMI


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

    if img1.dim() == 2:
        img1 = img1.unsqueeze(0).unsqueeze(0)
    elif img1.dim() == 3:
        img1 = img1.unsqueeze(0)
    if img2.dim() == 2:
        img2 = img2.unsqueeze(0).unsqueeze(0)
    elif img2.dim() == 3:
        img2 = img2.unsqueeze(0)


    ssim_map = K.metrics.ssim(img1, img2, window_size, eps=1e-12, padding='same', max_val=1.0)
    if reduction == 'mean':
        return ssim_map.mean(dim=(1, 2, 3))
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

    return structural_similarity(img1, img2, channel_axis=-1, data_range=data_range, win_size=window_size, gaussian_weights=True, sigma=sigma, K1=K1, K2=K2, full=return_img)


def ssim_masked(img1, img1_mask, img2, img2_mask, window_size=11, reduction='mean'):
    img1 = convert_image_to_tensor(img1)
    img2 = convert_image_to_tensor(img2)


    ssim_map = K.metrics.ssim(img1, img2, window_size, eps=1e-12, padding='same', max_val=1.0)
    if reduction == 'mean':
        return ssim_map.mean(dim=(1, 2, 3))
    elif reduction == 'none':
        return ssim_map
    else:
        raise ValueError("reduction must be either 'mean' or 'none'")
