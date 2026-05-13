# Deep Learning for Leaf Image Sequence Alignment of Field Crops
Tracking individual disease symptoms of crop leaves over time provides valuable insights into the progression of these diseases. Registering leaf images from different time points simplifies measuring the growth and emergence of individual disease symptoms over time.

We develop a landmark-based image registration method for non-rigid deformation of low-texture wheat leaves, without making use of artificial reference points. We use the feature matching method LoFTR to detect correspondences between leaf images. To validate correctness of the matches, we track matches across a three-image cycle and measure their consistency. The consistent matches then serve as landmarks for estimating a regularized thin plate spline transform that is applied to the moving image.

The resulting method does not rely on artificial markers and is thus able to process the full leaf area. Due to lack of ground truth and changes to the appearance of the leaf, quantitative evaluation of the method is challenging. Qualitatively, the method achieves satisfactory results in cases with only limited changes to the leaf, but is prone to artifacts and unstable transformations when the appearance changes drastically. The leading cause for faulty registrations are incorrect landmarks, demanding further investigation into verification of match quality.

## Quickstart

A RenkuLab Project is available at `https://renkulab.io/p/elena.boesch/leafalignment`

The Data Connectors of the project should provide access to the dataset, with no need for downloading the dataset.

The main evaluation can be run in the notebook `series_reg_eval.ipynb` in the section `Main Evaluation`. A different configuration can be chosen either by editing the config file `configs/default_s100.yaml` or by specifying the path the a differnt config file as the argument of `load_config()`. By default all three sequencing styles are tested on all leaves specified in `test_set.txt`. 

A more convenient setup will be provided soon.


 
## Brief Overview of Repo

`series_reg_eval.ipynb` :  Notebook used for primary evaluation  
`Final_Eval.ipynb` : Notebook used for analysis of experiment results and generating plots  
`registration.py` : contains all registration functions  
`loftr.py` : contains all functions relating to LoFTR and TPS, including filtering and warp consistency  
`masking.py` : contains masking and pre-processing functions  
`metrics.py` : contains all evaluation metrics  
`plotting.py` : contains utility functions for plotting images
`utils.py` : contains general utility functions  

`configs/` : directory containing base configurations for different experiments
`test_set.txt` : list of leaf uids comprising our test set

`LoFTR_Test.ipynb` : LoFTR testing ground  
`Masking.ipynb` : Masking testing ground  
`Match_subsampling.ipynb` : match subsampling/filtering  testing ground  
`Metrics.ipynb` : metrics testing ground  
`Robustness.ipynb` : testing robustness of LoFTR  
`Series_Registration.ipynb` : sequencing  testing ground  
`Testing.ipynb` : used for generating figures in thesis  
`TPS_Ransac.ipynb` :  testing ground for TPS Ransac  
`TPS_Testing.ipynb` : testing ground for different TPS implementations  
`Warp_Consistency.ipynb`: warp consistency  testing ground  

Other files are deprecated. Please be patient while I continue to clean up and document this repository.






## Attribution
The directory `DatasetTools` and the utility functions it contains were developed by Jonas Anderegg. The full repository is available at git@github.com:and-jonas/sympathique-wheat.git