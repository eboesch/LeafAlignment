import os
from pathlib import Path
import re
import json
import pickle
from PIL import Image
import DatasetTools.utils as utils
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.animation import FuncAnimation
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.colors import ListedColormap, BoundaryNorm
import numpy as np
import cv2
import copy
import skimage
from skimage.transform import PiecewiseAffineTransform
from skimage.transform import AffineTransform
from scipy.spatial import Delaunay
from tqdm import tqdm
import kornia as K


class KeypointEditor:
    def __init__(self, image, keypoints, target_image=None):
        self.image = image
        self.target_image = target_image
        self.keypoints = [tuple(kp) for kp in keypoints]
        self.fig, self.axes = plt.subplots(1, 2, figsize=(12, 6), sharex=False, sharey=False)
        self.ax_main = self.axes[0]
        self.ax_target = self.axes[1] if self.target_image is not None else None

        self.scatter = None
        self._draw_keypoints()
        self.cid = self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        self.fig.suptitle("Left-click: add, Right-click: remove nearest. Zoom & pan are synced.")

        plt.tight_layout()
        plt.show()

    def _draw_keypoints(self):
        self.ax_main.clear()
        self.ax_main.imshow(self.image, cmap='gray')
        self.ax_main.set_title("Original + Keypoints")

        if self.target_image is not None:
            self.ax_target.clear()
            self.ax_target.imshow(self.target_image, cmap='gray')
            self.ax_target.set_title("Warped Target Image")

        if self.keypoints:
            x, y = zip(*self.keypoints)
            self.scatter = self.ax_main.scatter(x, y, c='red', s=40, marker='x')

        self.fig.canvas.draw_idle()

    def onclick(self, event):

        if event.inaxes != self.ax_main:
            return
            
        if self.fig.canvas.manager.toolbar.mode != '':  # 'zoom rect', 'pan/zoom', etc.
            return

        x_click, y_click = event.xdata, event.ydata

        if event.button == 1:  # Left-click: Add point
            self.keypoints.append((x_click, y_click))
            print(f"Added: {(x_click, y_click)}")

        elif event.button == 3:  # Right-click: Remove nearest point
            if not self.keypoints:
                return

            dists = [np.hypot(x - x_click, y - y_click) for x, y in self.keypoints]
            idx = int(np.argmin(dists))
            removed = self.keypoints.pop(idx)
            print(f"Removed: {removed}")

        self._draw_keypoints()


class LeafDataset:

    def __init__(self, base_dir, leaf_uid=None, load=('images', 'tforms', 'rois', 'target_masks', 'target_images'), verbose=False):
        if verbose:
            print("Initializing dataset...")
        self.base_dir = base_dir
        self.leaf_uid = leaf_uid
        self.series = utils.get_series(path_images=os.path.join(base_dir, "raw", "*", "*"), leaf_uid=leaf_uid, verbose=verbose)[0]
        self.image_uids = [os.path.basename(p) for p in self.series]
        self.output_base = os.path.join(base_dir, "processed", self.leaf_uid)
        self.output_reg = os.path.join(base_dir, "processed", "reg", self.leaf_uid)
        self.output_ts = os.path.join(base_dir, "processed", "ts")
        self.shift_affine = np.array([[1, 0, 10000], [0, 1, 10000], [0, 0, 1]])

        # Initialize data containers
        self.images = None
        self.tforms = None
        self.rois = None
        self.keypoints = None
        self.edited_keypoints = None
        self.target_masks = None
        self.warped_masks = None
        self.instance_masks = None
        self.leaf_masks = None
        self.det_masks = None
        self.seg_masks = None
        self.symptoms_masks = None
        self.roi_leaf_images = None
        self.roi_leaf_masks = None
        self.target_images = None
        self.warped_images = None

        self.verbose = verbose
        if verbose:
            print("Loading requested values...")
        self._load_requested(load, verbose=verbose)

    def _extract_leaf_uid(self, path):
        return re.search(r'(ESWW00\d+_\d+)', path).group(1)

    def _load_requested(self, load, verbose=False):
        if 'images' in load:
            if verbose:
                print("Loading images...")
            # self.images = [K.io.load_image(img_path, K.io.ImageLoadType.RGB32)[None, ...] for img_path in self.series]
            self.images = [Image.open(p) for p in self.series]
        
        if 'cropped_images' in load:
            if verbose:
                print("Loading cropped images...")
            crop_dir = os.path.join(self.output_reg, "crop")
            self.cropped_images = self._load_images_from_dir(crop_dir)
            # self.images = [K.io.load_image(img_path, K.io.ImageLoadType.RGB32)[None, ...] for img_path in self.series]
            # self.images = [Image.open(p) for p in self.series]

        if 'tforms' in load:
            if verbose:
                print("Loading transforms...")
            roi_dir = os.path.join(self.output_base, "roi")
            self.tforms = []

            for path in self.series:
                name = os.path.splitext(os.path.basename(path))[0]
                path_tform = os.path.join(roi_dir, f"{name}_tform_piecewise.pkl")
                
                if os.path.exists(path_tform):
                    with open(path_tform, 'rb') as f:
                        self.tforms.append(pickle.load(f))
                else:
                    print(f"Warning: tform not found for {name}")
                    self.tforms.append(None)
                    
        if 'rois' in load:
            if verbose:
                print("Loading ROIs...")
            roi_dir = os.path.join(self.output_reg, "roi")
            self.rois = []
            for path in self.series:
                name = os.path.splitext(os.path.basename(path))[0]
                path_roi = os.path.join(roi_dir, f"{name}.json")
                if os.path.exists(path_roi):
                    with open(path_roi, 'r') as f:
                        self.rois.append(json.load(f))
                else:
                    print(f"Warning: ROI not found for {name}")
                    self.rois.append(None)

        if 'target_masks' in load:
            if verbose:
                print("Loading Target Masks...")
            target_mask_dir = os.path.join(self.output_reg, "mask_aligned", "piecewise")
            self.target_masks = self._load_images_from_dir(target_mask_dir)

        if 'instance_masks' in load:
            instance_mask_dir = os.path.join(self.output_ts, self.leaf_uid, "instance_mask" )
            self.instance_masks = self._load_images_from_dir(instance_mask_dir)

        if 'leaf_masks' in load:
            if verbose:
                print("Loading Leaf Masks...")
            mask_dir = os.path.join(self.output_ts, self.leaf_uid, "leaf_mask")
            self.leaf_masks = self._load_images_from_dir(mask_dir)

        if 'det_masks' in load:
            if verbose:
                print("Loading Detection Masks...")
            mask_dir = os.path.join(self.output_reg, "predictions", "symptoms_det", "pred")
            self.det_masks = self._load_images_from_dir(mask_dir)

        if 'seg_masks' in load:
            if verbose:
                print("Loading Segmentation Masks...")
            mask_dir = os.path.join(self.output_reg, "predictions", "symptoms_seg", "pred")
            self.seg_masks = self._load_images_from_dir(mask_dir)

        if 'symptom_masks' in load:
            if verbose:
                print("Generating Symptom Masks...")
            self.combine_masks()

        if 'roi_leaf_masks' in load:
            if verbose:
                print("Generating ROI Leaf Masks...")
            if self.symptoms_masks == None:
                self.combine_masks()
            self.get_roi_leaf_mask()

        if 'target_images' in load:
            if verbose:
                print("Loading target images...")
            target_dir = os.path.join(self.output_reg, "result", "piecewise")
            self.target_images = self._load_images_from_dir(target_dir)

        if 'keypoints' in load:
            if verbose:
                print("Loading keypoints...")
            self.keypoints = []
            kpts_dir = os.path.join(self.output_reg, "keypoints")
            for i, path in enumerate(self.series):
                # kpts_dir = os.path.join(os.path.dirname(path), "runs", "pose", "predict", "labels")
                name = os.path.splitext(os.path.basename(path))[0]
                path_kpts = os.path.join(kpts_dir, f"{name}.txt")
                if os.path.exists(path_kpts):
                    shape = np.asarray(self.images[i]).shape
                    coords = utils.get_keypoints(file_path=path_kpts, shape=shape)
                    self.keypoints.append(coords)
                else:
                    print(f"Warning: keypoints not found for {name}")
                    self.keypoints.append(None)

        if 'edited_keypoints' in load:
            kpts_dir = os.path.join(self.output_base, "keypoints", "edited")
            self.edited_keypoints = []
            for path in self.series:
                name = os.path.splitext(os.path.basename(path))[0]
                path_kpts = os.path.join(kpts_dir, f"{name}.txt")
                if os.path.exists(path_kpts):
                    coords = [tuple(pt) for pt in np.loadtxt(path_kpts, delimiter=",")]
                    self.edited_keypoints.append(coords)
                else:
                    print(f"Warning: edited keypoints not found for {name}")

    def _load_images_from_dir(self, dir_path):
        result = []
        for path in self.series:
            name = os.path.splitext(os.path.basename(path))[0]
            img_path = os.path.join(dir_path, f"{name}.png")
            if not os.path.exists(img_path):
                img_path = os.path.join(dir_path, f"{name}.JPG")
            if os.path.exists(img_path):
                result.append(Image.open(img_path))
            else:
                print(f"Warning: file not found for {name}")
                result.append(None)
        return result
    
    def sort_tforms(self):

        self.tforms_sorted = []

        for i in tqdm(range(len(self.images)), desc="Processing series"):

            tf = self.tforms[i]
            image_uid = self.image_uids[i]

            if tf is None:
                print(f"Warning: No transformation found for {image_uid}")
                self.tforms_sorted.append(None)  # Append placeholder
                continue

            # source points and triangulation
            src_pts = tf._tesselation.points
            tri = Delaunay(src_pts)
            triangles = src_pts[tri.simplices]
            centroids = np.mean(triangles, axis=1)  # shape (n_triangles, 2)
            sorted_indices = np.argsort(centroids[:, 0])  # sort by x-coordinate
            
            # Deep copy and reorder all relevant attributes
            new_tf = copy.deepcopy(tf)
            new_tf.affines = [tf.affines[i] for i in sorted_indices]

            self.tforms_sorted.append(new_tf)

    def show_frame(self, i, show=('images', 'masks', 'target_images')):
        """Display selected elements for a specific frame index.
        
        Parameters:
            i (int): Index of the frame to display.
            show (tuple or str): Elements to show, e.g., ('masks', 'target_images')
        """

        if isinstance(show, str):
            show = (show,)  # allow a single string

        available_data = {
            key: value for key, value in self.__dict__.items()
            if not key.startswith('_') and value is not None
        }

        valid_show = [s for s in show if available_data.get(s) is not None]

        if not valid_show:
            raise ValueError("None of the requested elements are available.")

        fig, axs = plt.subplots(1, len(valid_show), sharex=True, sharey=True)
        if len(valid_show) == 1:
            axs = [axs]

        for ax, element in zip(axs, valid_show):
            data_list = available_data[element]
            if i >= len(data_list) or data_list[i] is None:
                img = np.zeros((100, 100))  # fallback if missing
            else:
                img = data_list[i]
            ax.imshow(img)
            ax.set_title(f"{element.capitalize()} - frame {i}")

        fig.suptitle(f"Frame {i+1}", fontsize=16)
        plt.show()


    def show_series(self, interval=500, show=('images', 'masks', 'target_images')):
        """Animate the image series (like a GIF), with zoom preserved between frames.
        
        Parameters:
            interval (int): Delay between frames in milliseconds.
            show (tuple): Elements to show, e.g. ('images', 'masks')
        """
        # Generate a colormap with 100 Viridis colors
        num_colors = 50
        viridis_colors = plt.cm.viridis(np.linspace(0, 1, num_colors))
        cmap = ListedColormap(viridis_colors)

        # Create boundaries so that each integer from 0 to num_colors-1 gets its own bin.
        # Here, label 0 can be considered the background.
        boundaries = np.arange(-0.5, num_colors + 0.5, 1)
        norm = BoundaryNorm(boundaries, cmap.N)

        if isinstance(show, str):
            show = (show,)

        fig, axs = plt.subplots(1, len(show), sharex=True, sharey=True)
        if len(show) == 1:
            axs = [axs]

        n_frames = len(self.image_uids)

        def find_first_valid_frame(element):
            data = getattr(self, element)
            for d in data:
                if d is not None:
                    return np.asarray(d)
            return np.zeros((100, 100))  # fallback if all are None

        # Track zoom state per axis
        zoom_state = [None for _ in show]
        images = []

        # Initialize axes with imshow and store the image objects
        for j, element in enumerate(show):
            img_data = find_first_valid_frame(element)
            im = axs[j].imshow(img_data)
            images.append(im)
            axs[j].set_title(f"{element} - frame 0")

        fig.suptitle(f"Frame 1/{n_frames}", fontsize=16)

        def on_xlim_changed(event_ax):
            for j, ax in enumerate(axs):
                if ax == event_ax:
                    zoom_state[j] = (ax.get_xlim(), ax.get_ylim())

        for ax in axs:
            ax.callbacks.connect('xlim_changed', on_xlim_changed)

        def get_frame(i, element):
            data = getattr(self, element)
            if data and data[i] is not None:
                return np.asarray(data[i])
            else:
                return None

        def update(i):
            for j, element in enumerate(show):
                img_data = get_frame(i, element)
                if img_data is not None:
                    if element == "instance_masks":
                        # Ensure the mask data is an integer array
                        img_data = img_data.astype(int)
                        # Use the categorical colormap and normalization defined above
                        images[j].set_data(img_data)
                        images[j].set_cmap(cmap)
                        images[j].set_norm(norm)
                    else:
                        images[j].set_data(img_data)
                else:
                    images[j].set_data(np.zeros_like(images[j].get_array()))
                axs[j].set_title(f"{element} - frame {i}")
                if zoom_state[j] is not None:
                    axs[j].set_xlim(zoom_state[j][0])
                    axs[j].set_ylim(zoom_state[j][1])

        ani = FuncAnimation(fig, update, frames=n_frames, interval=interval, repeat=True)
        plt.show()

    def review_keypoints(self, i, show='keypoints'):
        # Only initialize the list once
        if not hasattr(self, 'edited_keypoints') or not self.edited_keypoints:
            self.edited_keypoints = [None] * len(self.images)

        image = self.images[i]

        # Try to get edited keypoints if they exist
        if self.edited_keypoints[i] is not None and show == "edited_keypoints":
            keypoints = self.edited_keypoints[i]
        elif self.edited_keypoints[i] is not None and show == "edited_keypoints":
            print(f"Warning: No edited keypoints for frame {i}, using original keypoints.")
            keypoints = self.keypoints[i]
        else:
            keypoints = self.keypoints[i]

        editor = KeypointEditor(image, keypoints, target_image=self.target_images[i])

        # Save edited keypoints
        self.edited_keypoints[i] = editor.keypoints


    def review_keypoints_series(self, show='keypoints'):

        output_path = Path(self.output_base) / "keypoints" / "edited"
        output_path.mkdir(parents=True, exist_ok=True)
    
        for i, path in enumerate(self.series):

            print(f"Frame {i}")
            self.review_keypoints(i, show=show)

            # Save the edited keypoints to a .txt file
            keypoints = self.edited_keypoints[i]
            if keypoints:
                save_path = output_path / f"{Path(path).stem}.txt"
                np.savetxt(save_path, np.array(keypoints), fmt="%.3f", delimiter=",")
            else:
                print(f"No keypoints to save for frame {i}.")

            cont = input("Next frame? [Enter to continue, q to quit] ")
            if cont.strip().lower() == 'q':
                break

        
    def warp_images(self):

        self.warped_images = []

        for i in tqdm(range(len(self.images)), desc="Processing series"):

            # get elements
            pw_warped = None  # re-initilize for each iteration
            image_uid = self.image_uids[i]
            img = np.asarray(self.images[i])
            
            # rotate 
            roi = self.rois[i]
            M_img= np.asarray(roi["rotation_matrix"])
            rows, cols = img.shape[0], img.shape[1]
            img_rot = cv2.warpAffine(img, M_img, (cols, rows))
            
            # crop
            box = np.asarray(roi["bounding_box"])
            crop = img_rot[box[0][1]:box[2][1], box[0][0]:box[1][0]]

            # get target dimensions from initial frame
            if i == 0:
                h, w = crop.shape[:2]
            
            # warp
            tform = self.tforms[i]
            if tform is not None:
                pw_warped = skimage.transform.warp(crop, tform, output_shape =(h, w))
                pw_warped = skimage.util.img_as_ubyte(pw_warped)
            else:
                print(f"Warning: No transformation found for {image_uid}")
                pw_warped = crop
            self.warped_images.append(pw_warped)

    def combine_masks(self):

        self.symptoms_masks = []

        for i in tqdm(range(len(self.images)), desc="Processing series"):

            det_mask_ = np.asarray(self.det_masks[i])
            det_mask_ = np.where(det_mask_ == 0, det_mask_, det_mask_ + 4)
            seg_mask_ = np.asarray(self.seg_masks[i])
            self.symptoms_masks.append(np.where(det_mask_ == 0, seg_mask_, det_mask_))

    def get_roi_leaf_mask(self):

        self.roi_leaf_masks = []
        self.roi_leaf_images = []

        for i in tqdm(range(len(self.images)), desc="Processing series"):

            # get elements
            roi = self.rois[i]
            if roi is None:
                if self.verbose:
                    print(f"No ROI data for index {i}")
                self.roi_leaf_masks.append(None)
                self.roi_leaf_images.append(None)
                continue
            bbox = np.asarray(roi["bounding_box"])
            box = np.intp(bbox)
            rot = np.asarray(roi['rotation_matrix'])
            
            rows, cols = np.asarray(self.images[i]).shape[:2]
            _, mh = map(int, np.mean(box, axis=0))
            # target = np.asarray(self.target_images[i])
            symptoms_mask = self.symptoms_masks[i]

            # full mask
            full_mask = np.zeros((rows, cols)).astype("float32")
            full_mask[mh - 1024:mh + 1024, :] = symptoms_mask

            # binarize
            leaf_mask = np.where(full_mask == 0, 0, 1).astype("float32")
            
            # rotate mask
            leaf_mask_rot = cv2.warpAffine(leaf_mask, rot, (cols, rows))

            # crop roi
            roi_mask = leaf_mask_rot[box[0][1]:box[2][1], box[0][0]:box[1][0]]

            # image
            image = np.asarray(self.images[i])

            # rotate image
            leaf_image_rot = cv2.warpAffine(image, rot, (cols, rows))

            # crop image roi
            roi_image = leaf_image_rot[box[0][1]:box[2][1], box[0][0]:box[1][0]]

            self.roi_leaf_masks.append(roi_mask)
            self.roi_leaf_images.append(roi_image)

            # # get leaf mask
            # leaf_mask = utils.remove_points_from_mask(mask=full_mask, classes=kpt_cls)

            # # rotate mask
            # segmentation_mask_rot = cv2.warpAffine(segmentation_mask, rot, (cols, rows))

            # # crop roi
            # roi = segmentation_mask_rot[box[0][1]:box[2][1], box[0][0]:box[1][0]]

    def warp_masks(self, kpt_cls=[5,6], n_cls=6):

        self.warped_masks = []

        for i in tqdm(range(len(self.images)), desc="Processing series"):

            # get elements
            roi = self.rois[i]
            bbox = np.asarray(roi["bounding_box"])
            box = np.intp(bbox)
            rot = np.asarray(roi['rotation_matrix'])
            symptoms_mask = self.symptoms_masks[i]
            rows, cols = np.asarray(self.images[i]).shape[:2]
            _, mh = map(int, np.mean(box, axis=0))
            target = np.asarray(self.target_images[i])

            # full mask
            full_mask = np.zeros((rows, cols)).astype("uint8")
            full_mask[mh - 1024:mh + 1024, :] = symptoms_mask

            # remove points
            segmentation_mask = utils.remove_points_from_mask(mask=full_mask, classes=kpt_cls)

            # rotate mask
            segmentation_mask_rot = cv2.warpAffine(segmentation_mask, rot, (cols, rows))

            # crop roi
            roi = segmentation_mask_rot[box[0][1]:box[2][1], box[0][0]:box[1][0]]

            # warp roi (except for the first image in the series)
            
            tform = self.tforms[i]
            if tform is not None:
                lm = np.stack([roi, roi, roi], axis=2)
                warped = skimage.transform.warp(lm, tform, output_shape=target.shape)
                warped = skimage.util.img_as_ubyte(warped[:, :, 0])
            else:
                warped = roi

            # Transform points, add to mask ====================================================================================

            # warp points
            if tform is not None:
                complete = utils.rotate_translate_warp_points(
                    mask=full_mask,
                    classes=kpt_cls,
                    rot=rot,
                    box=box,
                    tf=tform,
                    target_shape=target.shape,
                    warped=warped,
                )
            else:
                complete = warped

            # Output ===========================================================================================================

            # transform to ease inspection
            complete = (complete.astype("uint32")) * 255 / n_cls
            self.warped_masks.append(complete.astype("uint8"))


    def filter_tform_shape(self, AR_threshold):

        for i in tqdm(range(len(self.images)), desc="Processing series"):

            tf = self.tforms[i]
            image_uid = self.image_uids[i]

            if tf is None:
                print(f"Warning: No transformation found for {image_uid}")
                continue

            # source points and triangulation
            src_pts = tf._tesselation.points
            tri = Delaunay(src_pts)
            triangles = src_pts[tri.simplices]

            num_kept = 0
            for j, triangle in enumerate(triangles):
                ar = utils.triangle_aspect_ratio(triangle)
                if ar > AR_threshold:
                    tf.affines[j].params[:, :] = self.shift_affine
                else:
                    num_kept += 1

            kept_ratio = num_kept / len(triangles)
            print(f"{image_uid}: Kept {num_kept} of {len(triangles)} triangles ({kept_ratio:.1%})")

            # Directly update the transformation in-place
            self.tforms[i] = tf


    def filter_tform_diff(self, diff_threshold=500, window_size=10, max_iterations=2, plot=False):
        
        self.tforms_filter_diff = []

        def is_replaced(affine_matrix, shift_affine, tol=1e-3):
            return np.allclose(affine_matrix, shift_affine, atol=tol)

        for i in tqdm(range(len(self.images)), desc="Processing series"):
            tf = self.tforms[i]
            image_uid = self.image_uids[i]

            if tf is None:
                print(f"Warning: No transformation found for {image_uid}")
                self.tforms_filter_diff.append(None)
                continue

            iteration = 0
            replaced_in_iteration = True

            while iteration < max_iterations and replaced_in_iteration:
                iteration += 1
                replaced_in_iteration = False

                # --- Get source points and triangulation ---
                src_pts = tf._tesselation.points
                tri = Delaunay(src_pts)
                triangles = src_pts[tri.simplices]
                centroids = np.mean(triangles, axis=1)  # shape (n_triangles, 2)

                # --- Get sorted indices by x coordinate ---
                sorted_indices = np.argsort(centroids[:, 0])

                # --- Reorder affine matrices according to sorted triangles ---
                matrices = [tf.affines[j].params.copy() for j in sorted_indices]
                mat_stack = np.stack(matrices)
                valid_mask = np.array([not is_replaced(m, self.shift_affine) for m in matrices])

                diff_norms = []

                for j in range(len(matrices)):
                    start = max(0, j - window_size // 2)
                    end = min(len(matrices), j + window_size // 2 + 1)

                    window_indices = list(range(start, end))
                    valid_in_window = [idx for idx in window_indices if valid_mask[idx]]

                    if not valid_in_window:
                        diff_norms.append(0)
                        continue

                    window = mat_stack[valid_in_window]
                    median_mat = np.median(window, axis=0)
                    norm = np.linalg.norm(matrices[j] - median_mat, ord='fro')
                    diff_norms.append(norm)

                    if valid_mask[j] and norm > diff_threshold:
                        matrices[j][:, :] = self.shift_affine
                        replaced_in_iteration = True

                # --- Reassign filtered matrices back to tf.affines in original order ---
                affines_filtered = [None] * len(matrices)
                for idx_sorted, original_idx in enumerate(sorted_indices):
                    affines_filtered[original_idx] = AffineTransform(matrix=matrices[idx_sorted])

                tf.affines = affines_filtered

                if plot:
                    plt.plot(diff_norms, marker='x', label=f'Iteration {iteration}')
                    plt.axhline(y=diff_threshold, color='r', linestyle='--', label='Threshold')
                    plt.title(f"Diff Norms - Frame {i} Iteration {iteration}")
                    plt.xlabel("Triangle Index (sorted)")
                    plt.ylabel("Frobenius Norm")
                    plt.legend()
                    plt.show(block=True)

                if not replaced_in_iteration:
                    print(f"No new outliers found in iteration {iteration} for frame {i}")

            self.tforms_filter_diff.append(tf)

    def plot_triangulation(self, save_path=None, show=False):
        """
        Plots the triangles with their indices on top of the source image and stores the plot object.

        Parameters:
            save_path (str, optional): If provided, saves the plot to this path.
            show (bool): Whether to display the plot immediately.

        Stores:
            self.triangle_plots (list): Stores figure objects for reuse.
        """

        def is_replaced(affine_matrix, shift_affine, tol=1e-3):
            return np.allclose(affine_matrix, shift_affine, atol=tol)
        
        shift_affine = np.array([[1, 0, 10000],
                         [0, 1, 10000],
                         [0, 0, 1]])
        
        if not hasattr(self, 'tri_plots'):
            self.tri_plots = [None] * len(self.images)

        for i in tqdm(range(len(self.images)), desc="Processing series"):

            tf = self.tforms[i]
            target = self.target_images[i]
            image_uid = self.image_uids[i]

            if tf is None:
                print(f"Warning: No transformation found for {image_uid}")
                self.tri_plots.append(None)  # Append placeholder
                continue

            src_pts = tf._tesselation.points
            tri = Delaunay(src_pts)
            triangles = src_pts[tri.simplices]

            # Determine which triangles are soft-removed
            valid_mask = [not is_replaced(tform.params, shift_affine) for tform in tf.affines]

            # Filter out soft-removed triangles before sorting
            valid_triangles = [triangles[i] for i, valid in enumerate(valid_mask) if valid]

            centroids = np.mean(valid_triangles, axis=1)  # shape (n_triangles, 2)
            sorted_indices = np.argsort(centroids[:, 0])  # sort by x-coordinate  

            # Deep copy and reorder
            new_tri = copy.deepcopy(valid_triangles)
            new_tri = [valid_triangles[i] for i in sorted_indices]

            fig, ax = plt.subplots(figsize=(8, 8))
            ax.imshow(target)
            for j, triangle in enumerate(new_tri):
                polygon = Polygon(triangle, edgecolor='white', facecolor='none', linewidth=1)
                ax.add_patch(polygon)
                centroid = triangle.mean(axis=0)
                ax.text(*centroid, str(j), color='yellow', fontsize=8, ha='center', va='center')
            
            ax.axis('off')

            canvas = FigureCanvas(fig)
            canvas.draw()
            buf = canvas.buffer_rgba()
            img = np.asarray(buf, dtype=np.uint8)
            img = img.reshape(canvas.get_width_height()[::-1] + (4,))
            img = img[..., :3]
            self.tri_plots[i] = img

            if save_path:
                fig.savefig(save_path, bbox_inches='tight')

            if show:
                plt.show()
            else:
                plt.close(fig)

    def show_series_triplot(self, interval=500):
        """
        Animate the triangle plots across the image series.

        Parameters:
            interval (int): Delay between frames in milliseconds.
        """
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation
        import numpy as np

        if not hasattr(self, 'tri_plots') or self.tri_plots is None:
            raise ValueError("No tri_plots attribute found or it's None. Run create_triplots(show=False) first.")

        tri_plots = self.tri_plots
        n_frames = len(tri_plots)

        # Find a valid frame for shape reference
        for d in tri_plots:
            if d is not None:
                H, W, *_ = np.asarray(d).shape
                break
        else:
            H, W = 100, 100  # fallback

        fig, ax = plt.subplots()
        im = ax.imshow(np.zeros((H, W, 3), dtype=np.uint8))  # initialize blank image
        ax.set_title("Triangle Plot - Frame 0")

        def update(i):
            plot_img = tri_plots[i]
            if plot_img is not None:
                im.set_data(plot_img)
            else:
                im.set_data(np.zeros((H, W, 3), dtype=np.uint8))  # blank if missing
            ax.set_title(f"Triangle Plot - Frame {i + 1}")
            return [im]

        ani = FuncAnimation(fig, update, frames=n_frames, interval=interval, repeat=True)
        plt.show()
