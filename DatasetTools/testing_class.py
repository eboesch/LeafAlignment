from importlib import reload
import DatasetTools.LeafImageSeries
reload(DatasetTools.LeafImageSeries)
from DatasetTools.LeafImageSeries import LeafDataset
from DatasetTools.LeafImageSeries import KeypointEditor

# load an image sequence
leaf = LeafDataset(
    base_dir='C:/Users/anjonas/PycharmProjects/sympathique-wheat',
    # leaf_uid="ESWW0090057_18", 
    leaf_uid="ESWW0070020_1", 
    load=('images', 'rois', 'tforms', 'target_masks', 'leaf_masks', 'instance_masks', 'det_masks', 'seg_masks', 'target_images', "keypoints")
)


leaf.combine_masks()  # combine detection and segmentation masks

leaf.get_roi_leaf_mask()

leaf.show_series(interval=1000, show=('roi_leaf_images', 'roi_leaf_masks'))

# show a frame from the series
leaf.show_frame(3, show=('instance_masks', 'target_images'))

# show the full sequence
leaf.show_series(interval=1000, show=('instance_masks', 'target_images'))

# repeat the warping of the original image with the original output
leaf.warp_images()

# # compare warped image with the original output
# leaf.show_series(interval=1000, show=('warped_images', 'target_images'))

# # repeat the warping of the original mask with the original output
# leaf.combine_masks()  # combine detection and segmentation masks
# # leaf.show_frame(5, show=('symptoms_masks'))
# leaf.warp_masks()
# # leaf.show_frame(5, show=('warped_masks'))

# # check that result is consistent
# leaf.show_series(interval=1000, show=('warped_masks', 'target_masks'))

# # create triangulation plots
# leaf.plot_triangulation(show=True)

# # show warped images with triangulation
# leaf.show_series_triplot(interval=1000)

# # filter triangles according to their aspect ratio
# leaf.filter_tform_shape(AR_threshold=6)  # tforms get replaced!
# leaf.plot_triangulation(show=True)
# leaf.show_series_triplot(interval=1000)

# # warp images using the filtered tforms
# leaf.warp_images()  # warped images get replaced!

# # warp masks using the filtered tforms
# leaf.warp_masks()  # warped masks get replaced!

# # check result
# leaf.show_series(interval=1000, show=('warped_masks', 'warped_images'))

# # show warped images with shape outlier triangles excluded
# leaf.show_series(interval=1000, show=('warped_images', 'targets'))

# ===================================================================================================

# # filter on differences between the affine transformation matrices
# leaf.filter_tform_diff(plot=True)

# # create triangulation plots
# leaf.plot_triangulation(show=True)

# # warp images using the filtered tforms
# leaf.warp_images(use="filter_diff")

# # show warped images with shape outlier triangles excluded
# leaf.show_series(interval=1000, show=('warped_images', 'targets'))

# ===================================================================================================

# edited keypoints and export edited keypoints
leaf.review_keypoints_series(show='keypoints')

# reload the sequence and inspect result
leaf = LeafDataset(
    base_dir='C:/Users/anjonas/PycharmProjects/sympathique-wheat',
    leaf_uid="ESWW0090057_18", 
    load=('images', 'target_images',"edited_keypoints")
)
leaf.review_keypoints_series(show='edited_keypoints')

# where needed, these edited keypoint coordinates can be used to re-align the images

# ===================================================================================================

