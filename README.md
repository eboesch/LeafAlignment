# LeafAlignment

A repository for non-rigid leaf image sequence registration of field-grown crop leaves.

## Abstract

Tracking individual disease symptoms of crop leaves over time provides valuable insights into the progression of these diseases. Registering leaf images from different time points simplifies measuring the growth and emergence of individual disease symptoms over time.

We develop a landmark-based image registration method for non-rigid deformation of low-texture wheat leaves, without making use of artificial reference points. We use the feature matching method LoFTR to detect correspondences between leaf images. To validate correctness of the matches, we track matches across a three-image cycle and measure their consistency. The consistent matches then serve as landmarks for estimating a regularized thin plate spline transform that is applied to the moving image.

The resulting method does not rely on artificial markers and is thus able to process the full leaf area. Due to lack of ground truth and changes to the appearance of the leaf, quantitative evaluation of the method is challenging. Qualitatively, the method achieves satisfactory results in cases with only limited changes to the leaf, but is prone to artifacts and unstable transformations when the appearance changes drastically. The leading cause for faulty registrations are incorrect landmarks, demanding further investigation into verification of match quality.

---

## Features

- Landmark-based registration using learned feature matching (LoFTR)
- Match validation via 'warp consistency'
- Regularized thin plate spline transforms for non-rigid deformation
- Support for multiple experiment configurations via YAML
- Evaluation and visualization notebooks for registration quality and robustness
- Detailed documentation in the shape of a Masters Thesis

---

## Getting Started

### 1. Clone the repository

```powershell
git clone https://github.com/eboesch/LeafAlignment.git
cd LeafAlignment
```

### 2. Create the environment

This project uses `conda` and installs Python packages from `temp/requirements.txt`.

```powershell
conda env create -f environment.yml
conda activate base
```

If you prefer pip only, install dependencies from `temp/requirements.txt` manually.

### 3. Installing the code

The project code can be installed in "editable mode", which allows notebooks and scripts to import the package from anywhere in the repository while immediately reflecting local code changes, without needing to reinstall the package.

After creating and activating the project environment, run:

```powershell
pip install -e .
```

You can then import functions from anywhere in the repository:

```python
from leafalignment.registration import fetch_registered_image_mask_seq
```


### 4. Usage

The most relevant functions and how to use them are introduced in `Showcase.ipynb`. For detailed documentation of the method and results, see the Thesis PDF in `documentation/`.

The main results of this project were obtained using the notebook `evaluation/Series_Registration_Evaluation.ipynb`, and then evaluated and visualized using `evaluation/Final_Eval.ipynb`. The numeric results are stored in `results/`, in `.csv` files according to experiments.

---

## Repository Structure

- `src/leafalignment/`
  - `registration.py`: registration pipelines
  - `loftr.py`: LoFTR matching, filtering, and warp consistency checks
  - `masking.py`: image masking and preprocessing utilities
  - `metrics.py`: evaluation metrics 
  - `plotting.py`: visualization helpers for images and results
  - `utils.py`: general utilities
  - `DatasetTools/`: dataset handling utilities, originally developed by Jonas Anderegg
- `configs/`: YAML experiment configs for different registration settings
- `evaluation/`: evaluation notebooks and scripts
- `development/`: notebooks used for algorithm testing and prototyping. Many are deprecated.
- `figures/`: diagrams and visualization assets
- `results/`: generated evaluation output files
- `log_data/` — logging and experiment metadata files
- `environment.yml` — conda environment definition
- `pyproject.toml` — project/package metadata
- `temp/requirements.txt` — pip dependency list

---

## Dataset access

The dataset is not included in this repository, but is freely accessible [here](https://libdrive.ethz.ch/index.php/s/Agn94FpGxtKyLkd). In the [original RenkuLab environment](https://renkulab.io/p/elena.boesch/leafalignment), dataset access is provided through data connectors.

If you are working locally, ensure your dataset loader paths match the dataset location available to your notebooks.



## Main Results

- Our approach works best when leaf appearance changes are moderate.
- Large changes in leaf texture, lighting, or morphology can lead to poor landmark matches and thus unstable warps and artifacts
- Improving match validation remains a key direction for future work.
- Quantitative Evaluation is a major challenge.




## Attribution

The `src/leafalignment/DatasetTools` utilities were developed by Jonas Anderegg and originate from `git@github.com:and-jonas/sympathique-wheat.git`.
