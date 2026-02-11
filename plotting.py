import kornia as K
import kornia.feature as KF
# from kornia_moons.viz import draw_LAF_matches
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as mpatches
import numpy as np
import torch

from utils import convert_image_to_tensor 


# def plot_matches_moons(img_fix, keypts_fix, img_mov, keypts_mov, inliers, N_show: int=100, inliers_only: bool=True, vertical: bool=True):
#     """
#     Plots matches between images.

#     Args:
#         img_fix:    fixed image
#         keypts_fix: coordinates of keypoints in fixed image 
#         img_mov:    moving image
#         keypts_mov: coordinates of keypoints in moving image 
#         inliers:     array containing classification as inlier of each match
#         N_show (int):   how many matches to plot 
#         inliers_only (bool):    whether to plot only matches classified as inliers
#         vertical (bool):    whether to plot images vertically stacked.
#     """
#     # select points to plot
#     if inliers_only:
#         inliers_t = torch.as_tensor(inliers).flatten() # flatten inliers array and convert to tensor
#         inlier_idx = torch.nonzero(inliers_t).flatten() # extract indices of True values

#         N_show = min(N_show, inliers.sum())
#         rand_idx = torch.randperm(len(inlier_idx))[:N_show]  
#         show_idx = inlier_idx[rand_idx] # plot subset of inliers
#     else:
#         N_show = min(N_show, len(keypts_fix))
#         show_idx = torch.randperm(len(keypts_fix))[:N_show] # plot subset of *all* matches
    
#     keypts_fix_show = keypts_fix[show_idx]
#     keypts_mov_show = keypts_mov[show_idx]
#     inliers_show = inliers[show_idx]

#     # plot
#     fig, ax = draw_LAF_matches(
#         KF.laf_from_center_scale_ori(
#             keypts_fix_show.view(1, -1, 2),
#             torch.ones(keypts_fix_show.shape[0]).view(1, -1, 1, 1),
#             torch.ones(keypts_fix_show.shape[0]).view(1, -1, 1),
#         ),
#         KF.laf_from_center_scale_ori(
#             keypts_mov_show.view(1, -1, 2),
#             torch.ones(keypts_mov_show.shape[0]).view(1, -1, 1, 1),
#             torch.ones(keypts_mov_show.shape[0]).view(1, -1, 1),
#         ),
#         torch.arange(keypts_fix_show.shape[0]).view(-1, 1).repeat(1, 2),
#         K.tensor_to_image(img_fix),
#         K.tensor_to_image(img_mov),
#         inliers_show,
#         draw_dict={"inlier_color": (0.2, 1, 0.2), "tentative_color": None, "feature_color": (0.2, 0.5, 1), "vertical": vertical},
#         return_fig_ax=True
#     )

#     return fig, ax
#     # fig.savefig('LeafAlignment/loftr_test.png')
#     # fig.show()

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

    if type(img_fix) == torch.Tensor:
        # img_fix = K.color.rgb_to_grayscale(img_fix)
        img_fix = K.tensor_to_image(img_fix)
    if type(img_mov) == torch.Tensor:
        # img_mov = K.color.rgb_to_grayscale(img_mov)
        img_mov = K.tensor_to_image(img_mov)

    if vertical:
        img_pair = np.concatenate([img_fix, img_mov], axis=0)
    else:
        img_pair = np.concatenate([img_fix, img_mov], axis=1)


    # Prepare figure
    if vertical:
        fig, ax = plt.subplots(figsize=(8,8)) # for vertical plot
    else:
        fig, ax = plt.subplots(figsize=(12,6)) # for horizontal plot
    
    ax.imshow(img_pair, cmap='gray')

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
    inliers_show = inliers[show_idx].astype(float)

    # set up colormap
    cmap = cm.winter
    colors = cmap(inliers_show)
    
    # Draw matches
    if vertical:
        for (x0, y0), (x1, y1), color in zip(keypts_fix_show, keypts_mov_show, colors):
            ax.scatter([x0, x1 ], [y0, y1 + img_fix.shape[0]], color=color, s=2)
            ax.plot([x0, x1 ], [y0, y1+ img_fix.shape[0]], color=color, linewidth=1)
    else:
        for (x0, y0), (x1, y1), color in zip(keypts_fix_show, keypts_mov_show, colors):
            ax.scatter([x0, x1 + img_fix.shape[1]], [y0, y1], color=color, s=1)
            ax.plot([x0, x1 + img_fix.shape[1]], [y0, y1], color=color, linewidth=1)

    ax.axis('off')

    legend_handles = [
        mpatches.Patch(color=cmap(0.0), label='Outlier'),
        mpatches.Patch(color=cmap(1.0), label='Inlier')
    ]

    if not inliers_only:
        if vertical:
            fig.legend(handles=legend_handles, loc='center right', bbox_to_anchor=(1.05, 0.5))#, 0.5, 0.5))
        else:
            fig.legend(handles=legend_handles, loc='center right', bbox_to_anchor=(1, 0.5))#, 0.5, 0.5))
    # plt.tight_layout()
    
    return fig, ax


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
    cbar = plt.colorbar(sm, ax=ax, orientation='horizontal')
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


def plot_overlay(img_fix, img_mov, title=None):
    """
    ideally the fixed image is in gray scale
    """
    # kornia and torch expect C x H x W, while skimage & matplotlib expect H x W x C
    if type(img_fix) == torch.Tensor:
        if img_fix.shape[1] == 3:
            # convert rgb image to grayscale
            img_fix = K.color.rgb_to_grayscale(img_fix)
        img_fix = K.tensor_to_image(img_fix)
    if type(img_mov) == torch.Tensor:
        img_mov = K.tensor_to_image(img_mov)

    if title == None:
        title = "Overlay: fixed (gray) + moving (hot)"

    fig = plt.figure(figsize=(12,6))
    plt.imshow(img_fix, cmap='gray')
    plt.imshow(img_mov, cmap='hot', alpha=0.5)
    plt.title(title)
    plt.axis('off')
    plt.show()

    return fig

def plot_image_pair(img1, img2, img1_ind: int=None, img2_ind: int = 2, title: str=None, title_offset: float=0.86):
    if type(img1) == torch.Tensor:
        img1 = K.tensor_to_image(img1)
    if type(img2) == torch.Tensor:
        img2 = K.tensor_to_image(img2)

    if img1_ind is None:
        img1_ind = 1
    if img2_ind is None:
        img2_ind = 2

    fig, axs = plt.subplots(1, 2, figsize=(12,6))
    axs[0].imshow(img1)
    axs[0].set_title(f"Image {img1_ind}")
    # axs[0].axes('off')

    axs[1].imshow(img2)
    axs[1].set_title(f"Image {img2_ind}")
    if title is not None:
        fig.suptitle(title, fontsize=22, y=title_offset)
    plt.tight_layout()
    
    return fig, axs
