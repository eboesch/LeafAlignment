import os
import csv
import kornia as K
import numpy as np
import torch
from tqdm import tqdm

from utils import convert_image_to_tensor
from metrics import mse_masked, local_ncc_masked, nmi_masked, ssim_masked, iou, hausdorff
from DatasetTools.LeafImageSeries import LeafDataset
from registration import fetch_registered_image_mask_pair

import warnings
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    module="monai"
)




if __name__ == "__main__":
    # out_file = "results/test.csv"
    out_file = "results/registration_eval3.csv"

    metrics = {'MSE': mse_masked, 'NCC': local_ncc_masked, 'MI': nmi_masked, 'SSIM': ssim_masked, 'IoU': iou, 'Hausdorff': hausdorff}

    METHODS = ["Piecewise Affine", "LoFTR + TPS ROI"]#, "LoFTR + TPS ROI with Markers"]
    data_to_load = []
    if "Piecewise Affine" in METHODS:
        data_to_load.extend(['target_images', 'target_masks'])
    if "LoFTR + TPS ROI" in METHODS or "LoFTR + TPS ROI with Markers" in METHODS:
        data_to_load.extend(['rois', 'images', 'keypoints', 'leaf_masks'])
    if "LoFTR + TPS Full" in METHODS or "LoFTR + TPS Full with Markers" in METHODS:
        data_to_load.extend(['seg_masks', 'cropped_images'])

    # fetch list of all leaf uids
    base_dir = '../leaf-image-sequences'
    assert os.path.exists(base_dir+'/raw'), "Base directory empty" 
    uid_dir = base_dir + '/processed/reg'
    leaf_uids = [
        name for name in os.listdir(uid_dir)
        if os.path.isdir(os.path.join(uid_dir, name))
    ]   


    # track which leaves have been evaluated for which metrics
    processed_keys = set()
    if os.path.exists(out_file):
        with open(out_file, newline="", mode="r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = (row["leaf_uid"], row["registration_method"])
                processed_keys.add(key)

    with open(out_file, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['leaf_uid', 'registration_method', 'fixed_image', 'moving_image']+list(metrics.keys()))
        
        # write header only if file is new
        if f.tell() == 0:
            writer.writeheader()

        # iterate through all leaves
        for uid in leaf_uids:

            # skip leaves that have been evaluated for all methods
            keys_to_check = [ (uid, method) for method in METHODS ]
            if all(key in processed_keys for key in keys_to_check):
                print(f"Leaf {uid} has been processed for all registration methods")
                continue
            
            # load leaf
            leaf = LeafDataset(
                base_dir=base_dir,
                leaf_uid=uid, 
                load=(data_to_load),
                verbose=False
            )  

            fixed_img_indices = [0] #,1]
            moving_img_indices = np.arange(1,len(leaf.image_uids))

            for method in METHODS:

                # skip leaves that have already been evaluated for this method
                key = (uid, method)
                if key in processed_keys:
                    print(f"Leaf {uid} has already been evaluated for {method} registration.")
                    continue
                else:
                    print(f"Evaluating {method} registration on leaf {uid}...")

                # iterate through fixed and moving indices
                for fixed_img_ind in fixed_img_indices:
                    for moving_img_ind in tqdm(moving_img_indices):

                        fixed_img, moving_img, fixed_mask, moving_mask = fetch_registered_image_mask_pair(leaf, fixed_img_ind, moving_img_ind, method)
                        
                        eval_res = {'leaf_uid': uid, 'registration_method': method, 'fixed_image': fixed_img_ind, 'moving_image': moving_img_ind}

                        if (fixed_img is None) or (moving_img is None) or (fixed_mask is None) or (moving_mask is None):
                            print(f"Error: missing data for leaf {uid}")
                            for metric_name, metric_func in metrics.items():
                                eval_res.update({metric_name: None})    
                            writer.writerow(eval_res)
                            continue                    

                        # evaluate for all metrics
                        for metric_name, metric_func in metrics.items():
                            val = metric_func(fixed_img, fixed_mask, moving_img, moving_mask)
                            eval_res.update({metric_name: val.item()})

                        # write data to output
                        writer.writerow(eval_res)
