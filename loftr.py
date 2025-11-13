import os
import cv2
import kornia as K
import kornia.feature as KF
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


def loftr_match(img_fix, img_mov):
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
    _, inliers = cv2.findFundamentalMat(mkpts0.cpu().numpy(), mkpts1.cpu().numpy(), cv2.USAC_MAGSAC, 0.5, 0.999, 100000)
    inliers = inliers > 0

    print(f"Total matches: {len(mkpts0)}")
    print(f"Matches with Confidence > 0.5: {torch.sum(confidence > 0.5)}")
    print(f"Inliers: {inliers.sum()} ({inliers.sum()/len(mkpts0):.2%})")

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

def tps_skimage(keypts_fix, keypts_mov, confidence, thrsld, img_mov):
    """
    Applies TPS to register moving image to fixed image. Keypoints are filtered by confidence.

    Returns: transformed moving image and transform function
    """

    # kornia and torch expect C x H x W, while skimage expects H x W x C
    img_mov_reordered = K.tensor_to_image(img_mov)

    img_fix_mks = keypts_fix[confidence > thrsld]
    img_mov_mks = keypts_mov[confidence > thrsld]


    tps = ski.transform.ThinPlateSplineTransform()
    tps.estimate(img_fix_mks, img_mov_mks) # estimate transform from img_fix -> img_mov
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
    

