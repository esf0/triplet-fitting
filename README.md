# Triplet Fitting

This repository provides tools for fitting models to datasets using triplet data structures. It includes integration with MATLAB for specific computational tasks.

## Installation

### Requirements

To install the required dependencies, use:

```bash
pip install -r requirements.txt
```

### Using MATLAB

If you want to use the original MATLAB engine, follow these steps:

1. **Installing the MATLAB Library**

   - First, find your MATLAB root folder by opening MATLAB and running the command `matlabroot`. This command will give you the path to the root folder of MATLAB.
   
   - Next, open your terminal. If you are using Windows, you can do this by pressing `Windows + R`, typing `cmd`, and pressing `Enter`.
   
   - In the terminal, navigate to the MATLAB Python engine directory by running the following command (make sure to replace `matlabroot` with the path you found):

     ```bash
     cd matlabroot\extern\engines\python
     ```

   - Finally, run the following command to install the MATLAB Python library:

     ```bash
     python3 setup.py install
     ```

   This will install the MATLAB Python engine and allow you to run MATLAB code from within your Python environment.

### Running the Code

To fit models, use the `fit_all` function from the `triplet_fitting.fit` module.

### Dependencies

- pandas
- numpy
- scipy
- sklearn
- tqdm
- matlab.engine
- matplotlib
- seaborn
- plotly
- hpcom

### Citations

#### Citation for BRMLtoolkit

This code uses the BRMLtoolkit from the book *Bayesian Reasoning and Machine Learning* by David Barber. The book is available in hardcopy from Cambridge University Press. The publishers have kindly agreed to allow the online version to remain freely accessible.

If you wish to cite the book, please use the following BibTeX citation:

```bibtex
@BOOK{barberBRML2012,
author = {Barber, D.},
title= {{Bayesian Reasoning and Machine Learning}},
publisher = {{Cambridge University Press}},
year = 2012}
```

#### Citation for This Code

If you wish to cite this code, please use the following BibTeX citation:

```bibtex
@software{sedov_triplet_fitting_2024,
  author       = {Sedov, E.},
  title        = {Triplet Fitting},
  version      = {0.1.1},
  year         = 2024,
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.13628611},
  url          = {https://doi.org/10.5281/zenodo.13628611}
}
```
