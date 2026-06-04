## Disclaimer
The notebooks in this directory were used over the course of the project to develop and test new methods. Therefore, many methods and functions used in these notebooks are deprecated. meaning that e.g. more recent methodologies exist, function signaturers have changed, etc. Thus, we do not guarantee that all code present in these notebooks is still functional. This is especially true regarding any methods using Skimage for TPS, as this approach was discarded due to being very slow.

For a showcase of the most relevant functions, please see `Showcase.ipynb`.


## Overview

- `LoFTR_Test.ipynb` : testing LoFTR and first TPS implementation
- `Masking.ipynb` : developing masking and preprocessing functions  
- `Match_subsampling.ipynb` : different methods for match filtering
- `Metrics.ipynb` : implementation and minor analysis of metrics  
- `Quality Criterion.ipynb` : testing different criteria for semi-sequential registration
- `Series_Registration.ipynb` : how to register an entire sequence  
- `TPS_Testing.ipynb` : testing different TPS implementations and analyizing TPS robustness
- `Warp_Consistency.ipynb`: developing warp consistency  